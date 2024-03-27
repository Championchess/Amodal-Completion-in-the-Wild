'''
The input of this architecture is Stable Diffusion feature
which is extracted via DIFT(NeurIPS 2023)
Github link: https://github.com/Tsingularity/dift
'''

# full assembly of the sub-parts to form the complete net
import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_parts import inconv, down, up, outconv
from .. import resnet


import ipdb
import cv2
import numpy as np

class UNetSDM5Skip(nn.Module):
    def __init__(self, in_channels=3, w=4, n_classes=2, use_deform=False):
        super(UNetSDM5Skip, self).__init__()
        self.inc = inconv(in_channels, int(16 * w))
        self.down1 = down(int(16 * w), int(32 * w))
        self.down2 = down(int(32 * w), int(64 * w))
        self.down3 = down(int(64 * w), int(128 * w))
        self.down4 = down(int(128 * w), int(256 * w))
        self.down5 = down(int(256 * w), int(256 * w))

        self.reduce_dim_0 = nn.Sequential(
            nn.Conv2d(1280, 128 * w, kernel_size=1),
            nn.BatchNorm2d(128 * w),
            nn.ReLU(inplace=True))
        self.reduce_dim_1 = nn.Sequential(
            nn.Conv2d(1280, 128 * w, kernel_size=1),
            nn.BatchNorm2d(128 * w),
            nn.ReLU(inplace=True))
        self.reduce_dim_2 = nn.Sequential(
            nn.Conv2d(640, 128 * w, kernel_size=1),
            nn.BatchNorm2d(128 * w),
            nn.ReLU(inplace=True))
        self.reduce_dim_3 = nn.Sequential(
            nn.Conv2d(320, 128 * w, kernel_size=1),
            nn.BatchNorm2d(128 * w),
            nn.ReLU(inplace=True))
            
        self.up0 = up(int(512+256+512 + 256), int(128 * w))
        self.up1 = up(int(256 * w + 256), int(64 * w))
        self.up2 = up(int(128 * w + 256), int(32 * w))
        self.up3 = up(int(64 * w), int(16 * w))
        self.up4 = up(int(32 * w), int(16 * w))
        self.outc = outconv(int(16 * w), n_classes)

        self.use_deform = use_deform

    def forward(self, x, rgb, return_feat=False):

        x1 = self.inc(x) # 32 x 32 x 512 x 512
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3) # 32 x 256 x 64 x 64
        x5 = self.down4(x4) # 32 x 512 x 32 x 32
        x6 = self.down5(x5) # 32 x 512 x 16 x 16

        img_feat_0 = self.reduce_dim_0(rgb[0].permute(0,3,1,2)) # 32 x 256 x 24 x 24
        img_feat_0 = F.interpolate(
            img_feat_0, size=(x6.size(2), x6.size(3)), mode='bilinear') # 32 x 256 x 16 x 16
        img_feat_1 = self.reduce_dim_1(rgb[1].permute(0,3,1,2)) # 32 x 256 x 48 x 48
        img_feat_1 = F.interpolate(
            img_feat_1, size=(x5.size(2), x5.size(3)), mode='bilinear') # 32 x 256 x 32 x 32
        img_feat_2 = self.reduce_dim_2(rgb[2].permute(0,3,1,2)) # 32 x 256 x 96 x 96
        img_feat_2 = F.interpolate(
            img_feat_2, size=(x4.size(2), x4.size(3)), mode='bilinear') # 32 x 256 x 64 x 64
        img_feat_3 = self.reduce_dim_3(rgb[3].permute(0,3,1,2)) # 32 x 256 x 96 x 96
        img_feat_3 = F.interpolate(
            img_feat_3, size=(x3.size(2), x3.size(3)), mode='bilinear') # 32 x 256 x 128 x 128

        cat = torch.cat((x6, img_feat_0), dim=1) # 512 + 256  # 32 x (512 + 256) x 16 x 16
        # ipdb.set_trace()
        x = self.up0(cat, torch.cat((x5, img_feat_1), dim=1))
        x = self.up1(x, torch.cat((x4, img_feat_2), dim=1)) # 32 x 128 x 32 x 32
        x = self.up2(x, torch.cat((x3, img_feat_3), dim=1)) # 32 x 64 x 64 x 64
        x = self.up3(x, x2) # 32 x 32 x 128 x 128
        x = self.up4(x, x1) # 32 x 32 x 256 x 256

        x_ = x
        x = self.outc(x) # 32 x 2 x 256 x 256
        # ipdb.set_trace()

        if return_feat: # false
            return x, x_
        else:
            return x


def unet2sdm5skip(in_channels, **kwargs):
    return UNetSDM5Skip(in_channels, w=2, **kwargs)

