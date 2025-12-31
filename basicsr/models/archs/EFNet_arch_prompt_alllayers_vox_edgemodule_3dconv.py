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
from basicsr.models.archs.arch_util import EdgeAwareSharpening_w_prompt_ChannelAttentionTransformerBlock, PromptMapGenBlock
from torch.nn import functional as F
from PIL import Image
import os

def scer_to_voxel_general(event: torch.Tensor) -> torch.Tensor:
    """
    SCER (B, C, H, W) -> voxel-like (B, C, H, W)
    규칙(채널 수 C에 대해 일반화):
      - 가운데 두 채널 [mid_left, mid_right]는 그대로 복사
      - 그보다 왼쪽 채널 i (< mid_left):  edge[i] = event[i] - event[i+1]   # forward diff
      - 그보다 오른쪽 채널 i (> mid_right): edge[i] = event[i] - event[i-1] # backward diff

    예) C=6이면 mid_left=2, mid_right=3 이므로
        edge[0]=e[0]-e[1], edge[1]=e[1]-e[2], edge[2]=e[2], edge[3]=e[3],
        edge[4]=e[4]-e[3], edge[5]=e[5]-e[4]
    """
    assert event.dim() == 4, "event must be (B, C, H, W)"
    B, C, H, W = event.shape
    assert C >= 2, "C must be >= 2"

    # 가운데 두 채널 인덱스
    mid_left  = C // 2 - 1
    mid_right = C // 2

    edge = torch.zeros_like(event)

    # 왼쪽 구간: i in [0, mid_left-1]  -> forward diff
    if mid_left > 0:
        edge[:, :mid_left] = event[:, :mid_left] - event[:, 1:mid_left+1]

    # 가운데 두 채널: 그대로 복사
    edge[:, mid_left:mid_right+1] = event[:, mid_left:mid_right+1]

    # 오른쪽 구간: i in [mid_right+1, C-1] -> backward diff
    if mid_right + 1 < C:
        edge[:, mid_right+1:] = event[:, mid_right+1:] - event[:, mid_right:-1]

    return edge


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
        self.prompt_num=32
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

            self.down_path_1.append(UNetConvBlock(prev_channels, (2**i) * wf, downsample, relu_slope, edge_chn=ev_chn, prompt_num=self.prompt_num, stride_promptweight=(2**i), num_heads=self.num_heads[i]))
            self.down_path_2.append(UNetConvBlock(prev_channels, (2**i) * wf, downsample, relu_slope, edge_chn=ev_chn, use_emgc=downsample))
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

        edge = scer_to_voxel_general(event)#torch.cat([event[:,2,:,:].unsqueeze(1), event[:,3,:,:].unsqueeze(1)], dim=1)
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

                x1, x1_up = down(x1, edge, prompt_weight_map, event_filter=ev[i], merge_before_downsample=self.fuse_before_downsample)
                encs.append(x1_up)

                if mask is not None:
                    masks.append(F.interpolate(mask, scale_factor = 0.5**i))
            
            else:
                x1 = down(x1, edge, prompt_weight_map, event_filter=ev[i], merge_before_downsample=self.fuse_before_downsample)


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
    def __init__(self, in_size, out_size, downsample, relu_slope, edge_chn=2, prompt_num=None, stride_promptweight=None, use_emgc=False, num_heads=None): # cat
        super(UNetConvBlock, self).__init__()
        self.prompt_num = prompt_num
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_emgc = use_emgc
        self.num_heads = num_heads
        self.stride_promptweight = stride_promptweight

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
            # ── Edge feature extraction: 2D Conv → 3D Conv로 교체 ───────────────────────────
            # t축만 stride를 크게 줄 것이므로 모듈 정의는 stride=(n,n,n)로 두고,
            # forward에서 F.conv3d로 런타임에 t_stride만 오버라이드
            self.k_t = 3
            self.t_pad = self.k_t // 2
            self.conv_edge3d = nn.Conv3d(in_channels = edge_chn, out_channels=out_size, kernel_size=(self.k_t,3,3), stride=stride_promptweight, padding=(self.t_pad,1,1), bias=True)
            # self.conv_edge = nn.Conv2d(edge_chn, out_size, kernel_size=3, padding=1, stride=stride_promptweight, bias=True)
            self.downsample_edge = conv_down(out_size, out_size, bias=False)
            self.conv1d = nn.Conv2d(out_size * 2, out_size, kernel_size=1)
            self.downsample_prompt = conv_down(out_size, out_size, bias=False)
            self.image_event_transformer = EdgeAwareSharpening_w_prompt_ChannelAttentionTransformerBlock(out_size, num_heads=self.num_heads, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias')
            self.prompt_localblock = PromptMapGenBlock(prompt_len=self.prompt_num, in_ch=out_size, prompt_dim=out_size, stride=stride_promptweight)

    def _edge_3d_project(self, edge):
        """
        edge: (B, Ce, H, W) 또는 (B, Ce, D, H, W)
        3D Conv에서 t_stride = D_in 으로 설정하여 D_out = 1이 되도록 만든 후,
        (B, Co, H', W')로 투영하여 반환.
        """
        if edge.dim() == 4:
            edge_5d = edge.unsqueeze(2)                 # (B, Ce, 1, H, W)
        elif edge.dim() == 5:
            edge_5d = edge                               # (B, Ce, D, H, W)
        else:
            raise ValueError(f"[edge] expected 4D/5D, got {edge.shape}")

        B, Ce, D_in, H, W = edge_5d.shape
        t_stride = D_in if D_in > 0 else 1               # D_out=1 보장

        out_3d = F.conv3d(
            edge_5d,
            self.conv_edge3d.weight,
            self.conv_edge3d.bias,
            stride=(t_stride, self.stride_promptweight, self.stride_promptweight),                     # t만 크게, x/y=1
            padding=(self.t_pad, 1, 1),
            dilation=self.conv_edge3d.dilation,
            groups=self.conv_edge3d.groups,
        )                                                # (B, Co, 1, H', W')

        out_2d = out_3d.squeeze(2)                       # (B, Co, H', W')
        return out_2d
        

    def forward(self, x, edge=None, prompt_weight_map=None, enc=None, dec=None, mask=None, event_filter=None, merge_before_downsample=True):
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
            # out_edge = self.conv_edge(edge)
            out_edge = self._edge_3d_project(edge)
            prompt_local = self.prompt_localblock(prompt_weight_map)
            
            out = self.image_event_transformer(self.conv1d(torch.cat([out, out_edge], 1)), event_filter, prompt_local) 
             
        if self.downsample:
            out_down = self.downsample(out)
            if not merge_before_downsample: 
                out_edge_down = self.downsample_edge(out_edge)
                out_prompt_down = self.downsample_prompt(prompt_local)
                out_down = self.image_event_transformer(self.conv1d(torch.cat([out_down,out_edge_down],1)), event_filter, out_prompt_down) 

            return out_down, out

        else:
            if merge_before_downsample:
                return out
            else:
                out = self.image_event_transformer(self.conv1d(torch.cat([out,out_edge], 1)), event_filter, prompt_local)


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
