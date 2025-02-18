# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 23:41:17 2025

@author: JCH
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(GaussianConv2D, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be an odd number."

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2  

        # A: scale factor, sigma 
        self.A = nn.Parameter(1 + torch.randn(out_channels, in_channels, 1, 1))  # (out_ch, in_ch, 1, 1)
        self.sigma = nn.Parameter(self.padding + torch.randn(out_channels, in_channels, 1, 1))  # (out_ch, in_ch, 1, 1)

        coords = torch.arange(kernel_size) - (kernel_size // 2)
        self.register_buffer("coords", coords.view(1, 1, 1, -1))  # 4D (1,1,1,kernel_size)

    def forward(self, x):
        sigma = torch.clamp(self.sigma, min=1e-3)  # sigma가 0이 되는 것을 방지
        weight_1d = self.A * torch.exp(-self.coords**2 / (2 * sigma**2))  # (out_ch, in_ch, 1, kernel_size)
        kernel_2d = torch.matmul(weight_1d.transpose(-1, -2), weight_1d)  # (out_ch, in_ch, kernel_size, kernel_size)
        return F.conv2d(x, kernel_2d, padding=self.padding)

# 사용 예시
batch, in_ch, h, w = 4, 3, 1024, 1024
input_tensor = torch.rand(batch, in_ch, h, w)

out_ch = 8
gaussian_conv = GaussianConv2D(in_channels=in_ch, out_channels=out_ch, kernel_size=5)  # 홀수만 가능
output = gaussian_conv(input_tensor)
print(output.shape)  # (1, 1, 1024, 1024)