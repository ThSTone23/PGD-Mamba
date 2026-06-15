# -*- coding: utf-8 -*-
# coding=utf-8
"""Documentation omitted for release."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(conv => BN => SiLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """Documentation omitted for release."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """Upscale then double conv"""

    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetDenoiser(nn.Module):
    """Documentation omitted for release."""
    def __init__(self, in_channels=2, out_channels=2, base_channels=32, depth=4):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_channels)
        self.downs = nn.ModuleList()
        ch = base_channels
        for _ in range(depth - 1):
            self.downs.append(Down(ch, ch * 2))
            ch *= 2
            
        
        self.up = nn.ModuleList()
        for _ in range(depth - 1):
            
            self.up.append(Up(ch + ch // 2, ch // 2))
            ch //= 2
        self.outc = OutConv(base_channels, out_channels)

        
        self.res_weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        
        x1 = self.inc(x)
        encs = [x1]
        for d in self.downs:
            encs.append(d(encs[-1]))
        
        feat = encs[-1]
        for i, u in enumerate(self.up):
            feat = u(feat, encs[-2 - i])
        residual = self.outc(feat)
        
        return x + self.res_weight * residual
