'''
Data processing is from ASBU(ICCV 2021)
Github link: https://github.com/ducminhkhoi/Amodal-Instance-Seg-ASBU
Our code system is based on ASBU(ICCV 2021)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_parts import inconv, down, up, outconv
from .. import resnet

import ipdb
import cv2
import numpy as np

class UNetResNet5Skip(nn.Module):
    def __init__(self, in_channels=3, w=4, n_classes=2, use_deform=False):
        super(UNetResNet5Skip, self).__init__()
        self.inc = inconv(in_channels, int(16 * w))
        self.down1 = down(int(16 * w), int(32 * w))
        self.down2 = down(int(32 * w), int(64 * w))
        self.down3 = down(int(64 * w), int(128 * w))
        self.down4 = down(int(128 * w), int(256 * w))
        self.down5 = down(int(256 * w), int(256 * w))
        self.image_encoder = resnet.resnet18(pretrained=True)
        self.reduce_dim = nn.Sequential(
            nn.Conv2d(self.image_encoder.out_dim, 128 * w, kernel_size=1),
            nn.BatchNorm2d(128 * w),
            nn.ReLU(inplace=True))
        self.up0 = up(int(512+256+512), int(128 * w))
        self.up1 = up(int(256 * w), int(64 * w))
        self.up2 = up(int(128 * w), int(32 * w))
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

        img_feat = self.image_encoder(rgb) # 32 x 512 x 16 x 16
        
        img_feat = self.reduce_dim(img_feat)
        
        img_feat = F.interpolate(
            img_feat, size=(x6.size(2), x6.size(3)), mode='bilinear', align_corners=True) # 32 x 256 x 16 x 16
        
        cat = torch.cat((x6, img_feat), dim=1) # 32 x (512 + 256) x 16 x 16
        
        x = self.up0(cat, x5)
        x = self.up1(x, x4) 
        x = self.up2(x, x3) 
        x = self.up3(x, x2) 
        x = self.up4(x, x1) 

        x_ = x
        x = self.outc(x) 

        if return_feat: # false
            return x, x_
        else:
            return x


def unet2res5skip(in_channels, **kwargs):
    return UNetResNet5Skip(in_channels, w=2, **kwargs)

