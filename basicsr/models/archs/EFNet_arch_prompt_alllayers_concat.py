'''
EFNet
@inproceedings{sun2022event,
      author = {Sun, Lei and Sakaridis, Christos and Liang, Jingyun and Jiang, Qi and Yang, Kailun and Sun, Peng and Ye, Yaozu and Wang, Kaiwei and Van Gool, Luc},
      title = {Event-Based Fusion for Motion Deblurring with Cross-modal Attention},
      booktitle = {European Conference on Computer Vision (ECCV)},
      year = 2022
      }
'''

import torch
import torch.nn as nn
import math
from basicsr.models.archs.arch_util import PromptGuided_ChannelAttentionTransformerBlock, PromptMapGenBlock
from torch.nn import functional as F
from PIL import Image
import os

def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer

def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

## Supervised Attention Module
## https://github.com/swz30/MPRNet
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size=3, bias=True):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img

class PixelwisePromptDist(nn.Module):
    """
    입력:  (B, C_in, H, W)
    출력:  (B, prompt_num, H, W)  # 픽셀별 softmax → 각 픽셀에서 채널합=1
    """
    def __init__(self, in_channels: int, prompt_num: int, relu_slope: float = 0.2):
        super().__init__()
        self.prompt_num = prompt_num

        # 다운샘플 제거 → stride=1 통일
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(relu_slope, inplace=True),

            nn.Conv2d(64, 128, 3, 1, 1),   # stride=1
            nn.BatchNorm2d(128),
            nn.LeakyReLU(relu_slope, inplace=True),

            nn.Conv2d(128, 256, 3, 1, 1),  # stride=1
            nn.BatchNorm2d(256),
            nn.LeakyReLU(relu_slope, inplace=True),

            nn.Conv2d(256, 512, 3, 1, 1),  # stride=1
            nn.BatchNorm2d(512),
            nn.LeakyReLU(relu_slope, inplace=True),
        )

        # 픽셀별 prompt_num개 로짓 생성
        self.head = nn.Conv2d(512, prompt_num, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)     # (B, 512, H, W)
        logits = self.head(feat)    # (B, P, H, W)

        # softmax → 픽셀별 채널 분포
        logits = logits - logits.max(dim=1, keepdim=True).values
        probs = F.softmax(logits, dim=1)  # (B, P, H, W)

        return probs

class EFNet(nn.Module):
    def __init__(self, in_chn=3, ev_chn=6, wf=64, depth=3, fuse_before_downsample=True, relu_slope=0.2, num_heads=[1,2,4]):
        super(EFNet, self).__init__()
        self.prompt_num = 32
        self.prompt_weight = PixelwisePromptDist(in_channels=ev_chn+in_chn, prompt_num=self.prompt_num, relu_slope=relu_slope)
        
        self.depth = depth
        self.fuse_before_downsample = fuse_before_downsample
        self.num_heads = num_heads
        self.down_path_1 = nn.ModuleList()
        self.down_path_2 = nn.ModuleList()
        self.conv_01 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.conv_02 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        # event
        self.down_path_ev = nn.ModuleList()
        self.conv_ev1 = nn.Conv2d(ev_chn, wf, 3, 1, 1)

        prev_channels = self.get_input_chn(wf)
        for i in range(depth):
            downsample = True if (i+1) < depth else False 

            self.down_path_1.append(UNetConvBlock(prev_channels, (2**i) * wf, downsample, relu_slope, prompt_num=self.prompt_num, stride_promptweight=(2**i), num_heads=self.num_heads[i]))
            self.down_path_2.append(UNetConvBlock(prev_channels, (2**i) * wf, downsample, relu_slope, use_emgc=downsample))
            # ev encoder
            if i < self.depth:
                self.down_path_ev.append(UNetEVConvBlock(prev_channels, (2**i) * wf, downsample , relu_slope))

            prev_channels = (2**i) * wf

        self.up_path_1 = nn.ModuleList()
        self.up_path_2 = nn.ModuleList()
        self.skip_conv_1 = nn.ModuleList()
        self.skip_conv_2 = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path_1.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope))
            self.up_path_2.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope))
            self.skip_conv_1.append(nn.Conv2d((2**i)*wf, (2**i)*wf, 3, 1, 1))
            self.skip_conv_2.append(nn.Conv2d((2**i)*wf, (2**i)*wf, 3, 1, 1))
            prev_channels = (2**i)*wf
        self.sam12 = SAM(prev_channels)

        self.cat12 = nn.Conv2d(prev_channels*2, prev_channels, 1, 1, 0)
        self.last = conv3x3(prev_channels, in_chn, bias=True)

    def forward(self, x, event, mask=None):
        image = x

        # B, C, H, W = event.shape
        prompt_weight_map = self.prompt_weight(torch.cat([event, image], dim=1))
#         B, N, H, W = prompt_weight_map.shape
# #         print("CHEKKKKBBBBCCCCCCCCNNNNNHHHHWWWWWWWW::",prompt_weights.shape)
# #         print("CHEKKKKBBBBCCCCCCCCNNNNNHHHHWWWWWWWW::",prompt_weights[0,:,0].shape)

#         counter = 0
#         # 디렉토리가 존재하는 경우 연번호 추가
#         while os.path.exists(f"/home/work/data/code/EFNet_original/EFNet/experiments/result_weight_nhwc_dilation/{counter}"):
# #             dir_path = os.path.join(base_path, f"{each_batch}_{counter}")
#             counter += 1

#         # 각 배치에 대해 이미지로 저장
#         for each_batch in range(B):
#             # 폴더 생성
#             os.makedirs(f"/home/work/data/code/EFNet_original/EFNet/experiments/result_weight_nhwc_dilation/{counter}/{each_batch}/", exist_ok=True)
#             print(f"WEIGHTMAP::: /home/work/data/code/EFNet_original/EFNet/experiments/result_weight_nhwc_dilation/{counter}/{each_batch}/")
#             # 텐서를 PIL 이미지로 변환
#             imgs = prompt_weight_map[each_batch].cpu().numpy()  # (N, H, W)
#             for _ in range(N):
#                 img = imgs[_]
# #                 print("CHECKJJINJIN:::::::", img.shape)
#                 img = (img * 255).astype('uint8')  # 그레이스케일 값 범위를 0-255로 조정
#                 img = Image.fromarray(img)
#                 img.save(f"/home/work/data/code/EFNet_original/EFNet/experiments/result_weight_nhwc_dilation/{counter}/{each_batch}/prompt_weight_{_}.png")

        ev = []
        #EVencoder
        e1 = self.conv_ev1(event)
        for i, down in enumerate(self.down_path_ev):
            if i < self.depth-1:
                e1, e1_up = down(e1, self.fuse_before_downsample)
                if self.fuse_before_downsample:
                    ev.append(e1_up)
                else:
                    ev.append(e1)
            else:
                e1 = down(e1, self.fuse_before_downsample)
                ev.append(e1)

        #stage 1
        x1 = self.conv_01(image)
        encs = []
        decs = []
        masks = []
        for i, down in enumerate(self.down_path_1):
            if (i+1) < self.depth:

                x1, x1_up = down(x1, prompt_weight_map, event_filter=ev[i], merge_before_downsample=self.fuse_before_downsample)
                encs.append(x1_up)

                if mask is not None:
                    masks.append(F.interpolate(mask, scale_factor = 0.5**i))
            
            else:
                x1 = down(x1, prompt_weight_map, event_filter=ev[i], merge_before_downsample=self.fuse_before_downsample)


        for i, up in enumerate(self.up_path_1):
            x1 = up(x1, self.skip_conv_1[i](encs[-i-1]))
            decs.append(x1)
        sam_feature, out_1 = self.sam12(x1, image)

        #stage 2
        x2 = self.conv_02(image)
        x2 = self.cat12(torch.cat([x2, sam_feature], dim=1))
        blocks = []
        for i, down in enumerate(self.down_path_2):
            if (i+1) < self.depth:
                if mask is not None:
                    x2, x2_up = down(x2, encs[i], decs[-i-1], mask=masks[i])
                else:
                    x2, x2_up = down(x2, encs[i], decs[-i-1])
                blocks.append(x2_up)
            else:
                x2 = down(x2)

        for i, up in enumerate(self.up_path_2):
            x2 = up(x2, self.skip_conv_2[i](blocks[-i-1]))

        out_2 = self.last(x2)
        out_2 = out_2 + image

        return [out_1, out_2]

    def get_input_chn(self, in_chn):
        return in_chn

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, prompt_num=None, stride_promptweight=None, use_emgc=False, num_heads=None): # cat
        super(UNetConvBlock, self).__init__()
        self.prompt_num = prompt_num
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_emgc = use_emgc
        self.num_heads = num_heads

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)        

        if downsample and use_emgc:
            self.emgc_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_dec = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_enc_mask = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_dec_mask = nn.Conv2d(out_size, out_size, 3, 1, 1)

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

        if self.num_heads is not None:
            self.downsample_prompt = conv_down(out_size, out_size, bias=False)
            self.image_event_transformer = PromptGuided_ChannelAttentionTransformerBlock(out_size, num_heads=self.num_heads, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias')
            # 1x1 conv to reduce channel size after concatenation
            self.conv1d = nn.Conv2d(out_size * 2, out_size, kernel_size=1)
            self.prompt_localblock = PromptMapGenBlock(prompt_len=self.prompt_num, in_ch=out_size, prompt_dim=out_size, stride=stride_promptweight)
        

    def forward(self, x, prompt_weight_map=None, enc=None, dec=None, mask=None, event_filter=None, merge_before_downsample=True):
        out = self.conv_1(x)

        out_conv1 = self.relu_1(out)
        out_conv2 = self.relu_2(self.conv_2(out_conv1))

        out = out_conv2 + self.identity(x)

        if enc is not None and dec is not None and mask is not None:
            assert self.use_emgc
            out_enc = self.emgc_enc(enc) + self.emgc_enc_mask((1-mask)*enc)
            out_dec = self.emgc_dec(dec) + self.emgc_dec_mask(mask*dec)
            out = out + out_enc + out_dec        
            
        if event_filter is not None and merge_before_downsample:
            # b, c, h, w = out.shape
            prompt_local = self.prompt_localblock(prompt_weight_map)
            out = self.image_event_transformer(self.conv1d(torch.cat([out, prompt_local], 1)), event_filter) 
             
        if self.downsample:
            out_down = self.downsample(out)
            if not merge_before_downsample: 
                out_prompt_down = self.downsample_prompt(prompt_local)
                out_down = self.image_event_transformer(self.conv1d(torch.cat([out_down, out_prompt_down], 1)), event_filter) 

            return out_down, out

        else:
            if merge_before_downsample:
                return out
            else:
                out = self.image_event_transformer(self.conv1d(torch.cat([out, prompt_local], 1)), event_filter)


class UNetEVConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_emgc=False):
        super(UNetEVConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_emgc = use_emgc

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        self.conv_before_merge = nn.Conv2d(out_size, out_size , 1, 1, 0) 
        if downsample and use_emgc:
            self.emgc_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_dec = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_enc_mask = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_dec_mask = nn.Conv2d(out_size, out_size, 3, 1, 1)

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

    def forward(self, x, merge_before_downsample=True):
        out = self.conv_1(x)

        out_conv1 = self.relu_1(out)
        out_conv2 = self.relu_2(self.conv_2(out_conv1))

        out = out_conv2 + self.identity(x)
             
        if self.downsample:

            out_down = self.downsample(out)
            
            if not merge_before_downsample: 
            
                out_down = self.conv_before_merge(out_down)
            else : 
                out = self.conv_before_merge(out)
            return out_down, out

        else:

            out = self.conv_before_merge(out)
            return out


class UNetUpBlock(nn.Module):

    def __init__(self, in_size, out_size, relu_slope):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(in_size, out_size, False, relu_slope)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out


if __name__ == "__main__":
    pass
