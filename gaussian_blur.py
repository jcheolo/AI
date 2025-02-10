# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 22:49:38 2025

@author: JCH
"""


import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision

import numpy as np
import matplotlib.pyplot as plt
import kornia

    

def gaussian_kernel(kernel_size: int, sigma: float):
    """Gaussian Kernel 생성"""
    # 커널 크기는 홀수여야 함
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")
    
    # 1D Gaussian 커널 계산
    kernel = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (kernel / sigma) ** 2)
    kernel = kernel / kernel.sum()  # 정규화

    return kernel

def gaussian_blur(image: torch.Tensor, kernel_size: int, sigma: float):
    """Gaussian Blur"""
    # 1D Gaussian 커널 생성
    kernel_1d = gaussian_kernel(kernel_size, sigma)
    
    # 2D Gaussian 커널 생성 (외적)
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # 4D로 변환 (1, 1, H, W)
    
    # 입력 이미지와 같은 채널 수로 확장
    channels = image.shape[1]
    kernel_2d = kernel_2d.expand(channels, 1, kernel_size, kernel_size)
    
    # padding 계산
    padding = kernel_size // 2
    
    # Convolution 적용
    blurred_image = F.conv2d(image, kernel_2d, padding=padding, groups=channels)
    return blurred_image

def gaussian_blur_separable(image: torch.Tensor, kernel_size: int, sigma: float):
    """Separable Gaussian Blur: 1D Kernel로 Convolution 두 번 수행"""
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")
    padding = kernel_size // 2
    channels = image.shape[1]
    
    # 1D Gaussian Kernel 생성
    kernel_1d = gaussian_kernel(kernel_size, sigma)
    
    # 1D 커널을 4D로 확장
    kernel_1d_h = kernel_1d.reshape(1, 1, kernel_size, 1)  # (1, 1, H, W) 형태
    #kernel_1d_w = kernel_1d.view(1, 1, 1, kernel_size)  # (1, 1, H, W) 형태    
    
    # 채널 수에 맞게 커널 확장
    kernel_1d_h = kernel_1d_h.expand(channels, 1, kernel_size, 1)
    #kernel_1d_w = kernel_1d_w.expand(channels, 1, 1, kernel_size)

    # Reflection Padding 적용 (가로/세로 각각)
    image = F.pad(image, (0, 0, padding, padding), mode='reflect')  # 세로 블러용 패딩
    image = F.conv2d(image, kernel_1d_h, groups=channels)           # 세로 방향 Blur

    kernel_1d_w = kernel_1d_h.reshape(channels, 1, 1, kernel_size)  # (1, 1, H, W) 형태
    image = F.pad(image, (padding, padding, 0, 0), mode='reflect')  # 가로 블러용 패딩
    image = F.conv2d(image, kernel_1d_w, groups=channels)           # 가로 방향 Blur
    return image

def crop_border(tensor, left,right,top,bottom):
    return tensor[:,:,left:-right, top:-bottom]



class GaussianBlurSeparable(nn.Module):
    def __init__(self, kernel_size: int, sigma: float):
        super(GaussianBlurSeparable, self).__init__()
        
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd.")
        
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.padding = kernel_size // 2
        
        kernel_1d = self._create_gaussian_kernel(kernel_size, sigma)
        self.register_buffer("kernel_1d", kernel_1d)

    def _create_gaussian_kernel(self, kernel_size: int, sigma: float):
        kernel = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (kernel / sigma) ** 2)
        kernel = kernel / kernel.sum() 
        return kernel

    def forward(self, x: torch.Tensor):
        channels = x.shape[1]
        
        # 세로 방향 Gaussian Blur
        kernel_h = self.kernel_1d.reshape(1,1,kernel_size,1)
        # x = F.pad(x, (0, 0, self.padding, self.padding), mode='reflect')
        # x = F.conv2d(x, kernel_h.expand(channels, 1, -1, -1), groups=channels)
        x = F.conv2d(x, kernel_h.expand(channels, 1, -1, -1), padding = (self.padding,0), groups=channels)

        # 가로 방향 Gaussian Blur
        kernel_w = kernel_h = self.kernel_1d.reshape(1,1,1,kernel_size) 
        # x = F.pad(x, (self.padding, self.padding, 0, 0), mode='reflect')
        # x = F.conv2d(x, kernel_w.expand(channels, 1, -1, -1), groups=channels)
        x = F.conv2d(x, kernel_w.expand(channels, 1, -1, -1), padding = (0,self.padding), groups=channels)
        return x



class GaussianBlur(nn.Module):
    def __init__(self, kernel_size: int, sigma: float):
        super(GaussianBlur, self).__init__()
        
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd.")
        
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.padding = kernel_size // 2
        
        kernel = self._create_gaussian_kernel(kernel_size, sigma)
        self.register_buffer("kernel", kernel)

    def _create_gaussian_kernel(self, kernel_size: int, sigma: float):
        kernel = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (kernel / sigma) ** 2)
        kernel = kernel / kernel.sum()  
        kernel = torch.outer(kernel, kernel)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        return kernel

    def forward(self, x: torch.Tensor):
        channels = x.shape[1]
        kernel = self.kernel.expand(channels, 1, kernel_size, kernel_size)
        x = F.conv2d(x, kernel, padding = self.padding, groups=channels)
        return x



image = torch.rand(1, 3, 128, 128)  # NCHW: 배치 크기 1, 채널 3, 높이와 너비 128
kernel_size = 3  # 커널 크기 (홀수)
sigma = 1.5      # Sigma 값

# Gaussian Blur 적용
# blur_torch = gaussian_blur(image, kernel_size, sigma)
# blur_torch = gaussian_blur_separable(image, kernel_size, sigma)
# G = GaussianBlurSeparable(kernel_size, sigma)
# blur_torch = G(image)
G = GaussianBlur(kernel_size, sigma)
blur_torch = G(image)
blur_vision = torchvision.transforms.functional.gaussian_blur(image, kernel_size, sigma)
# print(blurred_image.shape)  # (1, 3, 128, 128)

plt.imshow((blur_torch-blur_vision).numpy()[0,0,:,:])
a = crop_border((blur_torch - blur_vision), kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2 )
# a = blur_torch-blur_vision
a = torch.max(torch.abs(a))
print(a)

