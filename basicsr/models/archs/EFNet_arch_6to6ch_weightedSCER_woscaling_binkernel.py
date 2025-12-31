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
from basicsr.models.archs.arch_util import EventImage_ChannelAttentionTransformerBlock
from torch.nn import functional as F
import os
import matplotlib.pyplot as plt
import numpy as np

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

class DynamicChannelMixer(nn.Module):
    def __init__(self, in_ch=3, hidden=32, out_ch=3, norm='softmax'):
        super().__init__()
        # context를 뽑는 간단한 헤드 (원하면 더 깊게 가능)
        self.ctx = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_ch*in_ch, 1)  # (B, 36, H, W)
        )
        self.out_ch = out_ch
        self.norm = norm

    def forward(self, x):               # x: (B, 6, H, W)
        B, C, H, W = x.shape            # C=6
        w = self.ctx(x)                 # (B, 36, H, W)
        w = w.view(B, self.out_ch, C, H, W)       # (B, out=6, in=6, H, W)

        # ---- 가중치 활성화 선택지 ----
        if self.norm == 'softmax':
            w = F.softmax(w, dim=2)          # in축으로 정규화(합=1)
        elif self.norm == 'sigmoid':
            w = torch.sigmoid(w)
        elif self.norm == 'relu':
            w = F.relu(w)
        elif self.norm == 'softplus':
            w = F.softplus(w)
        elif self.norm in ('none', 'raw', None):
            pass  # 활성함수 생략 (그대로 사용)
        else:
            raise ValueError(f"Unknown norm: {self.norm}")

        # einsum으로 O[b,o,h,w] = sum_i w[b,o,i,h,w] * x[b,i,h,w]
        out = torch.einsum('boihw, bihw -> bohw', w, x)
        return out                      # (B, 6, H, W)

class SharedKernelScerVoxelMixer(nn.Module):
    def __init__(self, hidden=16, k=3, bias=True):
        super().__init__()
        pad = k // 2
        self.kernel = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=k, padding=pad, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=k, padding=pad, bias=bias),
        )

    def forward(self, scer: torch.Tensor, voxel: torch.Tensor) -> torch.Tensor:
        B, C, H, W = scer.shape
        assert C == 3 and voxel.shape[1] == 3, "입력 채널 수는 3이어야 합니다."

        # 모든 (s,v) 쌍의 차이: (B, 3, 3, H, W)
        diff = scer.unsqueeze(2) - voxel.unsqueeze(1)

        # 공통 커널 1장으로 일괄 처리
        diff_flat = diff.reshape(B * 9, 1, H, W).contiguous()
        w_flat = self.kernel(diff_flat)
        w = w_flat.view(B, 3, 3, H, W)  # (B, s, v, H, W)

        # voxel_v와 곱해 v축 합산 → s별 누적
        voxel_exp = voxel.unsqueeze(1).expand(B, 3, 3, H, W)  # (B, s, v, H, W)
        scer_hat = (w * voxel_exp).sum(dim=2)  # (B, 3, H, W)
        return scer_hat

        
# class ResidualDCM_Identity(nn.Module):
#     def __init__(self, in_ch=3, hidden=32, norm='softmax'):
#         super().__init__()
#         self.mix = DynamicChannelMixer(in_ch=in_ch, hidden=hidden, out_ch=in_ch, norm=norm)

#     def forward(self, x):
#         y = self.mix(x)          # (B, in_ch, H, W)
#         return x + y             # 동일 채널수면 그냥 더함

# class DynamicChannelMixer(nn.Module):
#     def __init__(self, in_ch=6, hidden=32, out_ch=6, norm='group_softmax', split_idx=3, eps=1e-12):
#         super().__init__()
#         # context head
#         self.ctx = nn.Sequential(
#             nn.Conv2d(in_ch, hidden, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(hidden, out_ch * in_ch, 1)  # (B, out_ch*in_ch, H, W)
#         )
#         self.out_ch = out_ch
#         self.in_ch = in_ch
#         self.norm = norm
#         self.split_idx = split_idx  # 0:split_idx-1, split_idx:in_ch-1
#         self.eps = eps

#     def _group_normalize(self, w):
#         """
#         w: (B, out_ch, in_ch, H, W)
#         두 그룹( [0:split_idx], [split_idx:in_ch] ) 각각에서 in_ch 축 합이 1이 되도록 정규화
#         """
#         B, O, I, H, W = w.shape
#         i0, i1 = 0, self.split_idx
#         i2 = I

#         g1 = w[:, :, i0:i1, :, :]       # (B, O, split, H, W)
#         g2 = w[:, :, i1:i2, :, :]       # (B, O, I-split, H, W)

#         if self.norm in ('group_softmax', 'softmax'):
#             # 각 그룹 내 softmax (양수 & 합=1)
#             g1 = F.softmax(g1, dim=2)
#             g2 = F.softmax(g2, dim=2)
#         elif self.norm in ('group_sum1', 'sum1'):
#             # 가중치 부호를 유지할 필요가 없다면 softplus로 양수화 후 합=1로 정규화
#             g1p = F.softplus(g1)
#             g2p = F.softplus(g2)
#             g1 = g1p / (g1p.sum(dim=2, keepdim=True) + self.eps)
#             g2 = g2p / (g2p.sum(dim=2, keepdim=True) + self.eps)
#         elif self.norm == 'sigmoid':
#             # 시그모이드 후 각 그룹 합=1로 재정규화
#             g1s = torch.sigmoid(g1)
#             g2s = torch.sigmoid(g2)
#             g1 = g1s / (g1s.sum(dim=2, keepdim=True) + self.eps)
#             g2 = g2s / (g2s.sum(dim=2, keepdim=True) + self.eps)
#         else:
#             # 'none' 등: 원래 값 그대로 두되, 각 그룹을 단순 정규화(합=1)만 수행하려면 여기서 처리
#             g1p = F.softplus(g1)
#             g2p = F.softplus(g2)
#             g1 = g1p / (g1p.sum(dim=2, keepdim=True) + self.eps)
#             g2 = g2p / (g2p.sum(dim=2, keepdim=True) + self.eps)

#         return torch.cat([g1, g2], dim=2)  # (B, O, I, H, W)

#     def forward(self, x):
#         # x: (B, in_ch, H, W)
#         B, C, H, W = x.shape
#         assert C == self.in_ch, f"in_ch mismatch: expected {self.in_ch}, got {C}"

#         w = self.ctx(x)                                # (B, out_ch*in_ch, H, W)
#         w = w.view(B, self.out_ch, self.in_ch, H, W)   # (B, O, I, H, W)

#         # ★ 그룹별 합=1 정규화
#         w = self._group_normalize(w)

#         # 그룹 정규화가 반영된 w로 채널-가중합 수행
#         # out[b,o,h,w] = sum_i w[b,o,i,h,w] * x[b,i,h,w]
#         out = torch.einsum('boihw,bihw->bohw', w, x)
#         return out # (B, 6, H, W)

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

class EFNet(nn.Module):
    def __init__(self, in_chn=3, ev_chn=6, wf=64, depth=3, fuse_before_downsample=True, relu_slope=0.2, num_heads=[1,2,4]):
        super(EFNet, self).__init__()
        self.depth = depth
        self.ev_chn = ev_chn
        self.fuse_before_downsample = fuse_before_downsample
        self.num_heads = num_heads
        self.down_path_1 = nn.ModuleList()
        self.down_path_2 = nn.ModuleList()
        self.conv_01 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.conv_02 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        # event
        self.down_path_ev = nn.ModuleList()
        self.conv_ev1 = nn.Conv2d(ev_chn, wf, 3, 1, 1)
        # self.dynamic_scer_left = DynamicChannelMixer(int(self.ev_chn/2), 32, int(self.ev_chn/2))
        # self.dynamic_scer_right = DynamicChannelMixer(int(self.ev_chn/2), 32, int(self.ev_chn/2))
        self.dynamic_scer_left = SharedKernelScerVoxelMixer(hidden=16, k=3)
        self.dynamic_scer_right = SharedKernelScerVoxelMixer(hidden=16, k=3)

        prev_channels = self.get_input_chn(wf)
        for i in range(depth):
            downsample = True if (i+1) < depth else False 

            self.down_path_1.append(UNetConvBlock(prev_channels, (2**i) * wf, downsample, relu_slope, num_heads=self.num_heads[i]))
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
        event_origin = event
        # save_dir = '/home/work/data/ev_repr_vis/'
        # os.makedirs(save_dir, exist_ok=True)
        
        # out_np = event.squeeze(0).detach().cpu().numpy()  # (6, H, W)
        # vmax = np.max(np.abs(out_np))
        # for c in range(out_np.shape[0]):
        #     m = out_np[c]
        #     # NaN/Inf 방지
        #     m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    
        #     # 대칭 정규화 범위 설정: v = max(|min|, |max|)
            
        #     if vmax < 1e-12:
        #         # 전부 0이면 흰색 이미지로 저장
        #         norm = np.zeros_like(m)
        #         vlim = 1.0
        #     else:
        #         norm = m / vmax   # [-1, 1]로 스케일
        #         vlim = 1.0
    
        #     # 저장
        #     plt.figure(figsize=(6, 4), dpi=100)
        #     # bwr: blue-white-red. vmin/vmax를 대칭(-1,1)으로 고정 → 0이 정확히 흰색
        #     plt.imshow(norm, cmap='bwr', vmin=-vlim, vmax=vlim)
        #     plt.axis('off')
        #     fname = os.path.join(save_dir, f"originalSCER_c{c}.png")
        #     if not os.path.exists(fname):
        #         plt.savefig(fname, bbox_inches='tight', pad_inches=0)
        #     plt.close()
        event_voxel_6 = scer_to_voxel_general(event)
        # out_np = event_voxel_6.squeeze(0).detach().cpu().numpy()  # (6, H, W)
        # for c in range(out_np.shape[0]):
        #     m = out_np[c]
        #     # NaN/Inf 방지
        #     m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    
        #     # 대칭 정규화 범위 설정: v = max(|min|, |max|)
        #     # vmax = np.max(np.abs(m))
        #     if vmax < 1e-12:
        #         # 전부 0이면 흰색 이미지로 저장
        #         norm = np.zeros_like(m)
        #         vlim = 1.0
        #     else:
        #         norm = m / vmax   # [-1, 1]로 스케일
        #         vlim = 1.0
    
        #     # 저장
        #     plt.figure(figsize=(6, 4), dpi=100)
        #     # bwr: blue-white-red. vmin/vmax를 대칭(-1,1)으로 고정 → 0이 정확히 흰색
        #     plt.imshow(norm, cmap='bwr', vmin=-vlim, vmax=vlim)
        #     plt.axis('off')
        #     fname = os.path.join(save_dir, f"voxel_c{c}.png")
        #     if not os.path.exists(fname):
        #         plt.savefig(fname, bbox_inches='tight', pad_inches=0)
        #     plt.close()
        # event = torch.cat([event_48[:,0,:,:].unsqueeze(1), event_48[:,8,:,:].unsqueeze(1), event_48[:,16,:,:].unsqueeze(1),  event_48[:,31,:,:].unsqueeze(1), event_48[:,39,:,:].unsqueeze(1), event_48[:,47,:,:].unsqueeze(1)], dim=1)
        
        # event = self.dynamic_scer(event_voxel_6)
        event = torch.cat([self.dynamic_scer_left(event_origin[:,:int(self.ev_chn/2),:,:], event_voxel_6[:,:int(self.ev_chn/2),:,:]), self.dynamic_scer_right(event_origin[:,int(self.ev_chn/2):,:,:], event_voxel_6[:,int(self.ev_chn/2):,:,:])], dim=1)
        # event = event + event_origin
        # out_np = event.squeeze(0).detach().cpu().numpy()  # (6, H, W)
        # for c in range(out_np.shape[0]):
        #     m = out_np[c]
        #     # NaN/Inf 방지
        #     m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    
        #     # 대칭 정규화 범위 설정: v = max(|min|, |max|)
        #     # vmax = np.max(np.abs(m))
        #     if vmax < 1e-12:
        #         # 전부 0이면 흰색 이미지로 저장
        #         norm = np.zeros_like(m)
        #         vlim = 1.0
        #     else:
        #         norm = m / vmax   # [-1, 1]로 스케일
        #         vlim = 1.0
    
        #     # 저장
        #     plt.figure(figsize=(6, 4), dpi=100)
        #     # bwr: blue-white-red. vmin/vmax를 대칭(-1,1)으로 고정 → 0이 정확히 흰색
        #     plt.imshow(norm, cmap='bwr', vmin=-vlim, vmax=vlim)
        #     plt.axis('off')
        #     fname = os.path.join(save_dir, f"dynamicSCER_c{c}.png")
        #     if not os.path.exists(fname):
        #         plt.savefig(fname, bbox_inches='tight', pad_inches=0)
        #     plt.close()
        del(event_voxel_6)
        # del(event_voxel_48)

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

                x1, x1_up = down(x1, event_filter=ev[i], merge_before_downsample=self.fuse_before_downsample)
                encs.append(x1_up)

                if mask is not None:
                    masks.append(F.interpolate(mask, scale_factor = 0.5**i))
            
            else:
                x1 = down(x1, event_filter=ev[i], merge_before_downsample=self.fuse_before_downsample)


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
    def __init__(self, in_size, out_size, downsample, relu_slope, use_emgc=False, num_heads=None): # cat
        super(UNetConvBlock, self).__init__()
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
            self.image_event_transformer = EventImage_ChannelAttentionTransformerBlock(out_size, num_heads=self.num_heads, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias')
        

    def forward(self, x, enc=None, dec=None, mask=None, event_filter=None, merge_before_downsample=True):
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
            out = self.image_event_transformer(out, event_filter) 
             
        if self.downsample:
            out_down = self.downsample(out)
            if not merge_before_downsample: 
                out_down = self.image_event_transformer(out_down, event_filter) 

            return out_down, out

        else:
            if merge_before_downsample:
                return out
            else:
                out = self.image_event_transformer(out, event_filter)


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
