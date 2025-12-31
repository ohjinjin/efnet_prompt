import math
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm

from basicsr.utils import get_root_logger

from einops import rearrange
import numbers
from timm.models.layers import DropPath, trunc_normal_, to_2tuple

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. '
                             'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


def flow_warp(x,
              flow,
              interp_mode='bilinear',
              padding_mode='zeros',
              align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h).type_as(x),
        torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(
        x,
        vgrid_scaled,
        mode=interp_mode,
        padding_mode=padding_mode,
        align_corners=align_corners)

    # TODO, what if align_corners=False
    return output


def resize_flow(flow,
                size_type,
                sizes,
                interp_mode='bilinear',
                align_corners=False):
    """Resize a flow according to ratio or shape.

    Args:
        flow (Tensor): Precomputed flow. shape [N, 2, H, W].
        size_type (str): 'ratio' or 'shape'.
        sizes (list[int | float]): the ratio for resizing or the final output
            shape.
            1) The order of ratio should be [ratio_h, ratio_w]. For
            downsampling, the ratio should be smaller than 1.0 (i.e., ratio
            < 1.0). For upsampling, the ratio should be larger than 1.0 (i.e.,
            ratio > 1.0).
            2) The order of output_size should be [out_h, out_w].
        interp_mode (str): The mode of interpolation for resizing.
            Default: 'bilinear'.
        align_corners (bool): Whether align corners. Default: False.

    Returns:
        Tensor: Resized flow.
    """
    _, _, flow_h, flow_w = flow.size()
    if size_type == 'ratio':
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == 'shape':
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(
            f'Size type should be ratio or shape, but got type {size_type}.')

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(
        input=input_flow,
        size=(output_h, output_w),
        mode=interp_mode,
        align_corners=align_corners)
    return resized_flow


# TODO: may write a cpp file
def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class Mutual_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Mutual_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        

    def forward(self, x, y):

        assert x.shape == y.shape, 'The shape of feature maps from image and event branch are not equal!'

        b,c,h,w = x.shape

        q = self.q(x) # image
        k = self.k(y) # event
        v = self.v(y) # event
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


##########################################################################
## Event-Image Channel Attention (EICA)
class EventImage_ChannelAttentionTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super(EventImage_ChannelAttentionTransformerBlock, self).__init__()

        self.norm1_image = LayerNorm(dim, LayerNorm_type)
        self.norm1_event = LayerNorm(dim, LayerNorm_type)
        self.attn = Mutual_Attention(dim, num_heads, bias)
        # mlp
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * ffn_expansion_factor)
        self.ffn = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

    def forward(self, image, event):
        # image: b, c, h, w
        # event: b, c, h, w
        # return: b, c, h, w
        assert image.shape == event.shape, 'the shape of image doesnt equal to event'
        b, c , h, w = image.shape
        fused = image + self.attn(self.norm1_image(image), self.norm1_event(event)) # b, c, h, w

        # mlp
        fused = to_3d(fused) # b, h*w, c
        fused = fused + self.ffn(self.norm2(fused))
        fused = to_4d(fused, h, w)

        return fused

##########################################################################
## Prompt guided module
class PromptGuided_ChannelAttentionTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super(PromptGuided_ChannelAttentionTransformerBlock, self).__init__()

        self.norm1_image_prompt = LayerNorm(dim, LayerNorm_type)
        self.norm1_event = LayerNorm(dim, LayerNorm_type)
        self.attn = Mutual_Attention(dim, num_heads, bias)
        # mlp
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * ffn_expansion_factor)
        self.ffn = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

    def forward(self, image_prompt, event):
        # image_edge: b, c, h, w
        # event: b, c, h, w
        # return: b, c, h, w
        assert image_prompt.shape == event.shape, 'the shape of image_prompt doesnt equal to event'
        b, c , h, w = image_prompt.shape
        fused = image_prompt + self.attn(self.norm1_image_prompt(image_prompt), self.norm1_event(event)) # b, c, h, w

        # mlp
        fused = to_3d(fused) # b, h*w, c
        fused = fused + self.ffn(self.norm2(fused))
        fused = to_4d(fused, h, w)

        return fused


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

##########################################################################
## customed attention
##########################################################################
class Mutual_Attention_prompt(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Mutual_Attention_prompt, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        

    def forward(self, x, y, prompt_feature):

        assert x.shape == y.shape, 'The shape of feature maps from image and event branch are not equal!'

        b,c,h,w = x.shape
#         print("check :::::::::::::::c", b,c,h,w)
#         print("check2 :::::::::::::::prompt", prompt_feature.shape)

        q = self.q(x) # image
        k = self.k(y) # event
        v = self.v(y) # event
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        prompt = rearrange(prompt_feature, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         print("CHECK PROMTPLL111::::", prompt.shape)
#         prompt = prompt.repeat(1, 1, 1, self.num_heads)
#         print("CHECK PROMTPLL::::", prompt.shape)
#         print("CHECK VVVVV::::", v.shape)
        v = v * prompt

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

##########################################################################
## customed attention_sum->mul
##########################################################################
class Mutual_Attention_prompt2(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Mutual_Attention_prompt2, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        

    def forward(self, x, y, prompt_local, prompt_global):

        assert x.shape == y.shape, 'The shape of feature maps from image and event branch are not equal!'

        b,c,h,w = x.shape
#         print("check :::::::::::::::c", b,c,h,w)
#         print("check2 :::::::::::::::prompt", prompt_local.shape)

        q = self.q(x) # image
        k = self.k(y) # event
        v = self.v(y) # event
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        prompt_l = rearrange(prompt_local, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        prompt_g = rearrange(prompt_global, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         print("CHECK PROMTPLL111::::", prompt.shape)
#         prompt = prompt.repeat(1, 1, 1, self.num_heads)
#         print("CHECK PROMTPLL::::", prompt.shape)
#         print("CHECK VVVVV::::", v.shape)
        v = (v + prompt_g) * prompt_l

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

##########################################################################
## customed Event-Image Channel Attention (EICA) using prompt
class EventImage_w_prompt_ChannelAttentionTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super(EventImage_w_prompt_ChannelAttentionTransformerBlock, self).__init__()

        self.norm1_image = LayerNorm(dim, LayerNorm_type)
        self.norm1_event = LayerNorm(dim, LayerNorm_type)
        self.attn = Mutual_Attention_prompt(dim, num_heads, bias)
        # mlp
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * ffn_expansion_factor)
        self.ffn = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

    def forward(self, image, event, prompt_local):
        # image: b, c, h, w
        # event: b, c, h, w
        # return: b, c, h, w
        assert image.shape == event.shape, 'the shape of image doesnt equal to event'
        b, c , h, w = image.shape

        prompt_feature = prompt_local
        fused = image + self.attn(self.norm1_image(image), self.norm1_event(event), prompt_feature) # b, c, h, w

        # mlp
        fused = to_3d(fused) # b, h*w, c
        fused = fused + self.ffn(self.norm2(fused))
        fused = to_4d(fused, h, w)

        return fused

##########################################################################
## customed Edge-aware Sharpening module with prompt
class EdgeAwareSharpening_w_prompt_ChannelAttentionTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super(EdgeAwareSharpening_w_prompt_ChannelAttentionTransformerBlock, self).__init__()

        self.norm1_image_edge = LayerNorm(dim, LayerNorm_type)
        self.norm1_event = LayerNorm(dim, LayerNorm_type)
        self.attn = Mutual_Attention_prompt(dim, num_heads, bias)
        # mlp
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * ffn_expansion_factor)
        self.ffn = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

    def forward(self, image_edge, event, prompt_local):
        # image_edge: b, c, h, w
        # event: b, c, h, w
        # return: b, c, h, w
        assert image_edge.shape == event.shape, 'the shape of image_edge doesnt equal to event'
        b, c , h, w = image_edge.shape
        
        prompt_feature = prompt_local
        fused = image_edge + self.attn(self.norm1_image_edge(image_edge), self.norm1_event(event), prompt_feature) # b, c, h, w

        # mlp
        fused = to_3d(fused) # b, h*w, c
        fused = fused + self.ffn(self.norm2(fused))
        fused = to_4d(fused, h, w)

        return fused

##########################################################################
## Edge-aware Sharpening module
class EdgeAwareSharpening_ChannelAttentionTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super(EdgeAwareSharpening_ChannelAttentionTransformerBlock, self).__init__()

        self.norm1_image_edge = LayerNorm(dim, LayerNorm_type)
        self.norm1_event = LayerNorm(dim, LayerNorm_type)
        self.attn = Mutual_Attention(dim, num_heads, bias)
        # mlp
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * ffn_expansion_factor)
        self.ffn = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

    def forward(self, image_edge, event):
        # image_edge: b, c, h, w
        # event: b, c, h, w
        # return: b, c, h, w
        assert image_edge.shape == event.shape, 'the shape of image_edge doesnt equal to event'
        b, c , h, w = image_edge.shape
        fused = image_edge + self.attn(self.norm1_image_edge(image_edge), self.norm1_event(event)) # b, c, h, w

        # mlp
        fused = to_3d(fused) # b, h*w, c
        fused = fused + self.ffn(self.norm2(fused))
        fused = to_4d(fused, h, w)

        return fused
    
##########################################################################
## customed Event-Image Channel Attention (EICA) using prompt
class EventImage_w_promptLG_ChannelAttentionTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super(EventImage_w_promptLG_ChannelAttentionTransformerBlock, self).__init__()

        self.norm1_image = LayerNorm(dim, LayerNorm_type)
        self.norm1_event = LayerNorm(dim, LayerNorm_type)
        self.attn = Mutual_Attention_prompt(dim, num_heads, bias)
        # mlp
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * ffn_expansion_factor)
        self.ffn = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

    def forward(self, image, event, prompt_local, prompt_global):
        # image: b, c, h, w
        # event: b, c, h, w
        # return: b, c, h, w
        assert image.shape == event.shape, 'the shape of image doesnt equal to event'
        b, c , h, w = image.shape

        prompt_feature = prompt_local
        fused = image + self.attn(self.norm1_image(image), self.norm1_event(event), prompt_feature) + prompt_global # b, c, h, w

        # mlp
        fused = to_3d(fused) # b, h*w, c
        fused = fused + self.ffn(self.norm2(fused))
        fused = to_4d(fused, h, w)

        return fused

##########################################################################
## customed Event-Image Channel Attention (EICA) using prompt_reverse operation
class EventImage_w_promptLG_reverse_ChannelAttentionTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super(EventImage_w_promptLG_reverse_ChannelAttentionTransformerBlock, self).__init__()

        self.norm1_image = LayerNorm(dim, LayerNorm_type)
        self.norm1_event = LayerNorm(dim, LayerNorm_type)
        self.attn = Mutual_Attention_prompt2(dim, num_heads, bias)
        # mlp
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * ffn_expansion_factor)
        self.ffn = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

    def forward(self, image, event, prompt_local, prompt_global):
        # image: b, c, h, w
        # event: b, c, h, w
        # return: b, c, h, w
        assert image.shape == event.shape, 'the shape of image doesnt equal to event'
        b, c , h, w = image.shape

        fused = image + self.attn(self.norm1_image(image), self.norm1_event(event), prompt_local, prompt_global) # b, c, h, w

        # mlp
        fused = to_3d(fused) # b, h*w, c
        fused = fused + self.ffn(self.norm2(fused))
        fused = to_4d(fused, h, w)

        return fused



################################
#  Prompt local
################################
class PromptMapGenBlock(nn.Module):
    # prompt_len=5, in_ch=128, prompt_dim=128
    def __init__(self, prompt_len=32, in_ch=128, prompt_dim=128, stride=1):
        super(PromptMapGenBlock,self).__init__()
#         self.SegNet = SegmentationNet(lin_dim_=in_ch, prompt_len_=prompt_len)
        self.prompt_param = nn.Parameter(torch.rand(prompt_len, prompt_dim))
        self.conv3x3 = nn.Conv2d(prompt_dim, in_ch, kernel_size=3, stride=stride, padding=1, bias=False)

    def forward(self,x):
        
        B, C, H, W = x.shape
#         print("CHECK JINJIN::::::x", x.shape)  # CHECK JINJIN::::::x torch.Size([4, 2, 256, 256])

        # x_flatten = torch.randn(
        #     x.permute(0, 2, 3, 1).reshape(B*H*W, -1).shape,
        #     device=x.device,
        #     dtype=x.dtype
        # )#x.permute(0, 2, 3, 1).reshape(B*H*W, -1)
        x_flatten = x.permute(0, 2, 3, 1).reshape(B*H*W, -1)
        # prompt_param = torch.randn(
        #     self.prompt_param.shape,
        #     device=x.device,
        #     dtype=x.dtype
        # )#self.prompt_param
        prompt_param = self.prompt_param
        prompt_feature = torch.matmul(x_flatten, prompt_param)
#         if input_colorname == None:
# #             seg_map = self.SegNet(x)
#             seg_map_ = seg_map.permute(0, 2, 3, 1).reshape(B*H*W, -1)        
#             prompt_feature = torch.matmul(seg_map_, self.prompt_param)
#         else:
#             seg_map = F.interpolate(input_colorname, (H,W), mode='bilinear').to(x.device) #.cuda()
#             seg_map_ = seg_map.permute(0, 2, 3, 1).reshape(B*H*W, -1)        
#             prompt_feature = torch.matmul(seg_map_, self.prompt_param)
        
        prompt_feature_ = prompt_feature.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        prompt_feature_ = self.conv3x3(prompt_feature_)

        # if H == 64:
        #     save_image(prompt_feature_.squeeze(0).unsqueeze(1), 'pred2.png')
        # # print(prompt_feature_.mean())

        return prompt_feature_#, seg_map



################################
#  Prompt local init방법 다양하게..
################################
class PromptMapGenBlock_diverse_init(nn.Module):
    # prompt_len=5, in_ch=128, prompt_dim=128
    def __init__(self, prompt_len=32, in_ch=128, prompt_dim=128, stride=1, init_mode='xavier_uniform', column_gain=None, signed_gain=False, seed=None):
        super(PromptMapGenBlock_diverse_init,self).__init__()
#         self.SegNet = SegmentationNet(lin_dim_=in_ch, prompt_len_=prompt_len)
        self.prompt_param = nn.Parameter(torch.empty(prompt_len, prompt_dim))
        self.conv3x3 = nn.Conv2d(prompt_dim, in_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.reset_parameters(init_mode=init_mode, column_gain=column_gain, signed_gain=signed_gain, seed=seed)

    
    @torch.no_grad()
    def reset_parameters(self, init_mode='xavier_uniform', column_gain=None, signed_gain=False, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        W = self.prompt_param              # (in_ch, prompt_dim)
        fin, fout = self.in_ch, self.prompt_dim

        # 1) 기본 대칭 분포 초기화들 (음수 포함)
        if init_mode == 'xavier_uniform':
            nn.init.xavier_uniform_(W, gain=1.0)               # U[-a, a]
        elif init_mode == 'xavier_normal':
            nn.init.xavier_normal_(W, gain=1.0)                # N(0, σ^2)
        elif init_mode == 'kaiming_uniform':
            nn.init.kaiming_uniform_(W, a=math.sqrt(5), mode='fan_in', nonlinearity='linear')  # U[-a,a]
        elif init_mode == 'kaiming_normal':
            nn.init.kaiming_normal_(W, a=math.sqrt(5), mode='fan_in', nonlinearity='linear')   # N(0, σ^2)
        elif init_mode == 'orthogonal':
            nn.init.orthogonal_(W, gain=1.0)                   # 직교; 요소는 ± 섞임
        elif init_mode == 'spherical':
            W.normal_(0, 1); W /= (W.norm(dim=0, keepdim=True) + 1e-12)   # 각 컬럼 L2=1, ± 섞임
        elif init_mode == 'normal':
            std = math.sqrt(2.0 / (fin + fout))
            W.normal_(0.0, std)                                # N(0, σ^2)
        elif init_mode == 'uniform_sym':
            bound = 1.0 / math.sqrt(fin)
            W.uniform_(-bound, bound)                          # U[-b, b]
        # 2) 다양성 실험용 분포들 (음수 포함)
        elif init_mode == 'rademacher':
            # 각 요소가 P(±1)=0.5인 Rademacher * 스케일
            W.bernoulli_(0.5); W.mul_(2).sub_(1.0)             # ±1
            W.mul_(1.0 / math.sqrt(fin))                       # fan-in 기준 스케일
        elif init_mode == 'laplace':
            # 라플라스(쌍곡선) 분포: 뾰족하고 꼬리가 김 → 희소성/강한 다양성
            dist = torch.distributions.Laplace(loc=0.0, scale=1.0 / math.sqrt(fin))
            W.copy_(dist.sample(W.shape))
        elif init_mode == 'gauss_mix':
            # 가우시안 두 개의 혼합(더 다양한 모드)
            W1 = torch.empty_like(W).normal_(0.0, 1.0 / math.sqrt(fin))
            W2 = torch.empty_like(W).normal_(0.0, 2.0 / math.sqrt(fin))    # 더 큰 분산
            mask = torch.empty_like(W).bernoulli_(0.3)                      # 30%는 큰 분산
            W.copy_(mask * W2 + (1 - mask) * W1)
        elif init_mode == 'zeros':
            W.zero_()                                         # 음수는 아님(0), 실험용
        else:
            raise ValueError(f"unknown init_mode: {init_mode}")

        # 3) 컬럼별 스케일 다양화 (로그-균등) + (옵션) 부호 랜덤
        if column_gain is not None:
            gmin, gmax = column_gain
            assert gmin > 0 and gmax > gmin
            log_g = torch.empty(1, self.prompt_dim, device=W.device).uniform_(math.log(gmin), math.log(gmax))
            gains = log_g.exp()  # (1, prompt_dim) > 0
            if signed_gain:
                # Rademacher(±1)를 곱해 컬럼별로 부호까지 뒤섞기
                signs = torch.empty(1, self.prompt_dim, device=W.device).bernoulli_(0.5).mul_(2).sub_(1.0)
                gains = gains * signs
            W.mul_(gains)   # 컬럼별 스케일/부호 적용


    def forward(self,x):
        
        B, C, H, W = x.shape
#         print("CHECK JINJIN::::::x", x.shape)  # CHECK JINJIN::::::x torch.Size([4, 2, 256, 256])

        # x_flatten = torch.randn(
        #     x.permute(0, 2, 3, 1).reshape(B*H*W, -1).shape,
        #     device=x.device,
        #     dtype=x.dtype
        # )#x.permute(0, 2, 3, 1).reshape(B*H*W, -1)
        x_flatten = x.permute(0, 2, 3, 1).reshape(B*H*W, -1)
        # prompt_param = torch.randn(
        #     self.prompt_param.shape,
        #     device=x.device,
        #     dtype=x.dtype
        # )#self.prompt_param
        prompt_param = self.prompt_param
        prompt_feature = torch.matmul(x_flatten, prompt_param)
#         if input_colorname == None:
# #             seg_map = self.SegNet(x)
#             seg_map_ = seg_map.permute(0, 2, 3, 1).reshape(B*H*W, -1)        
#             prompt_feature = torch.matmul(seg_map_, self.prompt_param)
#         else:
#             seg_map = F.interpolate(input_colorname, (H,W), mode='bilinear').to(x.device) #.cuda()
#             seg_map_ = seg_map.permute(0, 2, 3, 1).reshape(B*H*W, -1)        
#             prompt_feature = torch.matmul(seg_map_, self.prompt_param)
        
        prompt_feature_ = prompt_feature.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        prompt_feature_ = self.conv3x3(prompt_feature_)

        # if H == 64:
        #     save_image(prompt_feature_.squeeze(0).unsqueeze(1), 'pred2.png')
        # # print(prompt_feature_.mean())

        return prompt_feature_#, seg_map


################################
#  Prompt local with offset of dcn
################################
class PromptMapGenBlockDCN(nn.Module):
    # prompt_len=5, in_ch=128, prompt_dim=128
    def __init__(self, prompt_len=32, in_ch=128, prompt_dim=128, stride=1):
        super(PromptMapGenBlockDCN,self).__init__()
#         self.SegNet = SegmentationNet(lin_dim_=in_ch, prompt_len_=prompt_len)
        self.prompt_param = nn.Parameter(torch.rand(prompt_len, prompt_dim))
        self.conv3x3 = nn.Conv2d(prompt_dim, in_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.offset = nn.Conv2d(in_ch, 18, kernel_size=3, padding=1, bias=True)

    def forward(self,x):
        
        B, C, H, W = x.shape
#         print("CHECK JINJIN::::::x", x.shape)  # CHECK JINJIN::::::x torch.Size([4, 2, 256, 256])

        # x_flatten = torch.randn(
        #     x.permute(0, 2, 3, 1).reshape(B*H*W, -1).shape,
        #     device=x.device,
        #     dtype=x.dtype
        # )#x.permute(0, 2, 3, 1).reshape(B*H*W, -1)
        x_flatten = x.permute(0, 2, 3, 1).reshape(B*H*W, -1)
        # prompt_param = torch.randn(
        #     self.prompt_param.shape,
        #     device=x.device,
        #     dtype=x.dtype
        # )#self.prompt_param
        prompt_param = self.prompt_param
        prompt_feature = torch.matmul(x_flatten, prompt_param)
#         if input_colorname == None:
# #             seg_map = self.SegNet(x)
#             seg_map_ = seg_map.permute(0, 2, 3, 1).reshape(B*H*W, -1)        
#             prompt_feature = torch.matmul(seg_map_, self.prompt_param)
#         else:
#             seg_map = F.interpolate(input_colorname, (H,W), mode='bilinear').to(x.device) #.cuda()
#             seg_map_ = seg_map.permute(0, 2, 3, 1).reshape(B*H*W, -1)        
#             prompt_feature = torch.matmul(seg_map_, self.prompt_param)
        
        prompt_feature_ = prompt_feature.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        prompt_feature_ = self.conv3x3(prompt_feature_)
        offset = self.offset(prompt_feature_)

        # if H == 64:
        #     save_image(prompt_feature_.squeeze(0).unsqueeze(1), 'pred2.png')
        # # print(prompt_feature_.mean())

        return prompt_feature_, offset#, seg_map

################################
#  Prompt global
################################
class PromptMapGenBlock1D(nn.Module):
    # 사용 예: prompt_len=5, in_ch=128, prompt_dim=128
    def __init__(self, prompt_len=5, in_ch=128, prompt_dim=128):
        super(PromptMapGenBlock1D, self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(prompt_len, prompt_dim))
        self.fc = nn.Linear(prompt_dim, in_ch, bias=False)  # prompt_len → in_ch

    def forward(self, x, H, W):
        # x: (B, C)
        B, C = x.shape

        # (B, C) @ (prompt_len, C)^T = (B, prompt_len)
        prompt_feature = torch.matmul(x, self.prompt_param)  # (B, prompt_len)
        
        out = self.fc(prompt_feature)  # (B, in_ch)
        # (B, in_ch, 1, 1) → (B, in_ch, H, W)
        out = out.unsqueeze(-1).unsqueeze(-1)       # (B, in_ch, 1, 1)
        out = out.repeat(1, 1, H, W)                # (B, in_ch, H, W)
        return out
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, y, H=None, W=None):
        # x: image
        # y: event
        assert x.dim()==3, x.shape
        assert x.shape == y.shape
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            y_ = y.permute(0, 2, 1).reshape(B, C, H, W)
            y_ = self.sr(y_).reshape(B, C, -1).permute(0, 2, 1)
            y_ = self.norm(y_)
            kv = self.kv(y_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(y).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

