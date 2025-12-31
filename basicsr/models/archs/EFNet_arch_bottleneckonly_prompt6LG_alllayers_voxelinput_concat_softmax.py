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
from basicsr.models.archs.arch_util import EventImage_ChannelAttentionTransformerBlock, PromptGuided_ChannelAttentionTransformerBlock, PromptMapGenBlock, PromptMapGenBlock1D
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
        # probs = logits

        return probs

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights

class ResNet50PromptDist(nn.Module):
    """
    해상도 보존형 ResNet-50: 다운샘플링만 막은 버전
    입력: (B, C_in, H, W)  출력: (B, prompt_num, H, W)
    """
    def __init__(self, in_channels: int, prompt_num: int,
                 pretrained: bool = False, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

        # 1. pretrained ResNet-50 모델 로드
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)

        # 2. Stem 수정: conv1 stride=1, maxpool 제거하여 초기 다운샘플링 방지
        old_conv1 = m.conv1
        m.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                w = old_conv1.weight  # (64, 3, 7, 7)
                if in_channels == 3:
                    m.conv1.weight.copy_(w)
                else:
                    # 입력 채널이 3이 아닐 경우, 기존 가중치의 평균을 내어 확장
                    w_mean = w.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
                    m.conv1.weight.copy_(w_mean.expand(-1, in_channels, -1, -1))
        m.maxpool = nn.Identity()

        # 3. Layer 2, 3, 4의 다운샘플링 방지
        # ResNet-50의 Bottleneck 블록은 conv2에서 다운샘플링이 일어납니다.
        def _no_downsample(layer: nn.Sequential):
            b0 = layer[0]  # 각 layer의 첫 번째 블록
            
            # Bottleneck 블록의 stride는 conv2에 있으므로 conv2의 stride를 1로 변경
            if hasattr(b0, "conv2"):
                b0.conv2.stride = (1, 1)
                
            # 채널 매칭용 downsample conv가 있으면 stride도 1로 변경
            if getattr(b0, "downsample", None) is not None:
                ds0 = b0.downsample[0]
                if isinstance(ds0, nn.Conv2d):
                    ds0.stride = (1, 1)

        _no_downsample(m.layer2)
        _no_downsample(m.layer3)
        _no_downsample(m.layer4)

        # 4. 백본(FC 레이어 제외) 정의
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)  # (B, 64, H, W)
        self.layer1 = m.layer1  # (B, 256, H, W)
        self.layer2 = m.layer2  # (B, 512, H, W)
        self.layer3 = m.layer3  # (B, 1024, H, W)
        self.layer4 = m.layer4  # (B, 2048, H, W)

        # 5. Head 정의: 픽셀별 로짓을 prompt_num 채널로 매핑
        # ResNet-50의 최종 출력 채널은 2048입니다.
        self.head = nn.Conv2d(2048, prompt_num, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)      # (B, 2048, H, W) 해상도 유지

        # Head
        logits = self.head(x)   # (B, P, H, W)

        # Temperature-scaled stable softmax (픽셀별 softmax)
        logits = logits / max(self.temperature, 1e-6)
        logits = logits - logits.max(dim=1, keepdim=True).values # for numerical stability
        return F.softmax(logits, dim=1)

class ResNet18PromptDist(nn.Module):
    """
    해상도 보존형 ResNet18: 다운샘플만 막은 최소 수정 버전
    입력: (B, C_in, H, W)  출력: (B, prompt_num, H, W)
    """
    def __init__(self, in_channels: int, prompt_num: int,
                 pretrained: bool = False, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

        # 1) stem: conv1 stride=1, maxpool 제거
        old_conv1 = m.conv1
        m.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                w = old_conv1.weight  # (64,3,7,7)
                if in_channels == 3:
                    m.conv1.weight.copy_(w)
                else:
                    w_mean = w.mean(dim=1, keepdim=True)  # (64,1,7,7)
                    m.conv1.weight.copy_(w_mean.expand(-1, in_channels, -1, -1))
        m.maxpool = nn.Identity()

        # 2) layer2/3/4의 첫 블록 stride=1로 덮어쓰기 (다운샘플 방지)
        def _no_downsample(layer: nn.Sequential):
            b0 = layer[0]
            # 기본블록의 stride는 conv1에 들어가므로 conv1만 1로
            if hasattr(b0, "conv1"):
                b0.conv1.stride = (1, 1)
            # 채널 매칭용 downsample conv가 있으면 stride도 1로
            if getattr(b0, "downsample", None) is not None:
                ds0 = b0.downsample[0]
                if isinstance(ds0, nn.Conv2d):
                    ds0.stride = (1, 1)

        _no_downsample(m.layer2)
        _no_downsample(m.layer3)
        _no_downsample(m.layer4)

        # 백본(FC 제거)
        self.stem   = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)  # (B,64,H,W)
        self.layer1 = m.layer1  # (B,64,H,W)
        self.layer2 = m.layer2  # (B,128,H,W)
        self.layer3 = m.layer3  # (B,256,H,W)
        self.layer4 = m.layer4  # (B,512,H,W)

        # 픽셀별 로짓 → prompt_num채널
        self.head = nn.Conv2d(512, prompt_num, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)            # (B,512,H,W) 해상도 유지
        logits = self.head(x)         # (B,P,H,W)
        # 픽셀별 softmax
        logits = logits / max(self.temperature, 1e-6)
        logits = logits - logits.max(dim=1, keepdim=True).values
        return F.softmax(logits, dim=1)


class EFNet(nn.Module):
    def __init__(self, in_chn=3, ev_chn=6, wf=64, depth=3, fuse_before_downsample=True, relu_slope=0.2, num_heads=[1,2,4]):
        super(EFNet, self).__init__()
        self.prompt_num = 6
        # self.prompt_weight = PixelwisePromptDist(in_channels=ev_chn+in_chn, prompt_num=self.prompt_num, relu_slope=relu_slope)
        
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
        edge = scer_to_voxel_general(event)

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
        ######################
        # counter = 0
        # while os.path.exists(f"/home/work/data/code/EFNet_original/EFNet/experiments/result_weight_nhwc_dilation_prompt6_softmax/{counter}"):
        #     counter += 1
        ######################
        counter = None
        for i, down in enumerate(self.down_path_1):
            if (i+1) < self.depth:

                x1, x1_up = down(x1, event_filter=ev[i], edge=edge, merge_before_downsample=self.fuse_before_downsample)
                encs.append(x1_up)

                if mask is not None:
                    masks.append(F.interpolate(mask, scale_factor = 0.5**i))
            
            else:
                x1 = down(x1, event_filter=ev[i], edge=edge, merge_before_downsample=self.fuse_before_downsample, prompt=True)


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
            self.image_event_transformer = EventImage_ChannelAttentionTransformerBlock(out_size, num_heads=self.num_heads, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias')

        if self.num_heads is not None:
            self.conv_edge = nn.Conv2d(6, out_size, kernel_size=3, padding=1, stride=stride_promptweight, bias=True)
            self.downsample_edge = conv_down(out_size, out_size, bias=False)
            self.conv1d = nn.Conv2d(out_size * 3, out_size, kernel_size=1)
            if not downsample:
                self.image_event_prompt_transformer = PromptGuided_ChannelAttentionTransformerBlock(out_size, num_heads=self.num_heads, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias')
                self.prompt_weight = ResNet18PromptDist(in_channels=out_size, prompt_num=self.prompt_num, pretrained=False, temperature=1.0)
                self.prompt_localblock = PromptMapGenBlock(prompt_len=self.prompt_num, in_ch=out_size, prompt_dim=out_size, stride=1)
                self.prompt_globalblock = PromptMapGenBlock1D(prompt_len=self.prompt_num, in_ch=out_size, prompt_dim=out_size)
            
             
    def forward(self, x, edge=None, enc=None, dec=None, mask=None, event_filter=None, merge_before_downsample=True, prompt=None):
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
            
             
        if self.downsample: # 1,2 layer면 true
            out_down = self.downsample(out) # cross attn 한걸 downsampling
            if not merge_before_downsample: 
                out_down = self.image_event_transformer(out_down, event_filter) 

            return out_down, out

        else: # bottleneck
            if merge_before_downsample:
                out_edge = self.conv_edge(edge)
                prompt_weight_map = self.prompt_weight(out_edge)
                prompt_weight_map_global = prompt_weight_map.mean(dim=[2,3])

                prompt_local = self.prompt_localblock(prompt_weight_map)
                prompt_global = self.prompt_globalblock(prompt_weight_map_global, prompt_local.shape[2], prompt_local.shape[3])

                out = self.image_event_prompt_transformer(self.conv1d(torch.cat([out, prompt_local, prompt_global], 1)), event_filter)
                return out
            else:
                out_down = self.image_event_transformer(out_down, event_filter)
                


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
