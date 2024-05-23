# -*- coding: utf-8 -*-
"""
Created on Fri May 24 00:43:50 2024

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

from conv2d.conv2d import custom_maxpool2d


def gaussian_2d(x, y, mu_x, mu_y, sigma_x, sigma_y):
    return np.exp(-((x - mu_x)**2 / (2 * sigma_x**2) + (y - mu_y)**2 / (2 * sigma_y**2)))

def create_gaussian_image(width, height, mu_x, mu_y, sigma_x, sigma_y):
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    X, Y = np.meshgrid(x, y)
    
    image = gaussian_2d(X, Y, mu_x, mu_y, sigma_x, sigma_y)
    # image /= np.max(image)  # Normalize to [0, 1]
    image = image/image.sum()
    return image

def convolution_2d_with_padding(image, kernel, padding):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    output_height = image_height - kernel_height + 2 * padding + 1
    output_width = image_width - kernel_width + 2 * padding + 1
    
    padded_image = np.pad(image, padding, mode='constant')  # 이미지 주변에 zero-padding 추가
    
    output = np.zeros((output_height, output_width))
    
    for i in range(output_height):
        for j in range(output_width):
            output[i, j] = np.sum(padded_image[i:i+kernel_height, j:j+kernel_width] * kernel)
    
    return output

def create_circle(diameter, size=None):
    radius = diameter//2
    if diameter%2 == 1:
        x = np.linspace(-radius,radius,diameter)
        y = np.linspace(-radius,radius,diameter)
    else:
        x = np.linspace(-radius+0.5,radius-0.5,diameter)
        y = np.linspace(-radius+0.5,radius-0.5,diameter)
    x,y = np.meshgrid(x,y)
    distance = np.sqrt(x*x+y*y)
    if diameter%2 == 1:
        mask = distance <= radius+0.5
    else:
        mask = distance <= radius
    # tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape (1, 1, kernel_size, kernel_size)
    
    if size is not None:
        _mask = np.zeros([size,size])
        center = size//2
        if diameter%2 == 1:
            _mask[center-radius:center+radius+1,center-radius:center+radius+1] = mask[:,:]
        else:
            _mask[center-radius:center+radius,center-radius:center+radius] = mask[:,:]
        mask = _mask
    return mask.astype(float)
    
def crop_border(tensor, left,right,top,bottom):
    return tensor[:,:,left:-right, top:-bottom]
    




width = 11  # 이미지 너비
height = 11  # 이미지 높이
mu_x = 5  # Gaussian 중심 x 좌표
mu_y = 5  # Gaussian 중심 y 좌표
sigma_x = 1  # Gaussian x 축 표준편차
sigma_y = 1  # Gaussian y 축 표준편차

gaussian_image = create_gaussian_image(width, height, mu_x, mu_y, sigma_x, sigma_y)

ori = create_circle(20, 200)
input = create_circle(20, 200)
input = convolution_2d_with_padding(input, gaussian_image, 5)



A = torch.fft.fftshift(torch.fft.fft2(torch.tensor(input)))
temp = torch.zeros_like(A)
shift = 0
temp[shift:width+shift, shift:shift+height]=torch.tensor(gaussian_image)
ker_cen = gaussian_image.shape[-1]//2
temp = torch.roll(temp, shifts=(-ker_cen,-ker_cen), dims=(-2,-1))
# temp[100-5:100+6,100-5:100+6]=torch.tensor(gaussian_image)
B = torch.fft.fftshift(torch.fft.fft2(temp))

C = A/B
D = torch.fft.ifft2(torch.fft.ifftshift(C))
# plt.imshow(abs(D)-ori)

c=1.25
n = 2
x = np.linspace(-n, n, 2*n+1)*c
y = np.linspace(-n, n, 2*n+1)*c
# full coordinate arrays
xx, yy = np.meshgrid(x, y)
zz = np.sqrt(xx**2 + yy**2)
kernel = np.sinc(zz/np.pi)
kernel = kernel/kernel.sum()
plt.figure()
plt.imshow(kernel)

temp = torch.zeros_like(A)
temp[shift:kernel.shape[-2]+shift, shift:shift+kernel.shape[-1]]=torch.tensor(kernel)
ker_cen = kernel.shape[-1]//2
temp = torch.roll(temp, shifts=(-ker_cen,-ker_cen), dims=(-2,-1))
E = torch.fft.fftshift(torch.fft.fft2(temp))
F = C*E
G = torch.fft.ifft2(torch.fft.ifftshift(F))
plt.figure()
plt.imshow(abs(G))