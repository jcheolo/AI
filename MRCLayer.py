# -*- coding: utf-8 -*-
"""
Created on Sat May 18 16:43:04 2024

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

input = create_circle(20, 200)
input = convolution_2d_with_padding(input, gaussian_image, 5)
input1 = create_circle(20, 200)
input1 = convolution_2d_with_padding(input1, gaussian_image, 5)
input1 = np.roll(input1, (22,0))
input_bridge = input+input1

input_pinch = (1-input_bridge).copy()
input_pinch = np.roll(input_pinch, (-70,0))
input_pinch[:,75:]=0

input = input_bridge+input_pinch
# plt.imshow(input)

# gaussian_image = create_gaussian_image(width, height, mu_x, mu_y, sigma_x, sigma_y)
# input2= create_circle(20, 200)
# input2 = convolution_2d_with_padding(input2, gaussian_image, 5)
# input2 = np.roll(input2, (-50,0))
# input3 = np.roll(input2, (-20,0))
# input_pinch = np.maximum(input2, input3)

# input = input_bridge+input_pinch
# plt.imshow(input)

upsample = 1
mrc_limit = 9
diameter = int(mrc_limit*upsample)
kernel_size = diameter if diameter%2==1 else diameter+1
kernel = torch.tensor(create_circle(diameter))
# plt.figure(), plt.imshow(kernel)
# upsample = 1
# mrc_limit = 11
# diameter = int(mrc_limit*upsample)
# kernel_size = diameter if diameter%2==1 else diameter+1
# kernel2 = torch.tensor(create_circle(diameter))
# X=86
# Y=109+1
# input[X:X+1,Y:Y+1] = 1

input= np.where(input<0.01, 0, input)
input = torch.tensor(input).unsqueeze(0).unsqueeze(0)
up_input = torch.nn.functional.interpolate(input, scale_factor=upsample, mode='bilinear')
up_input = up_input.clone().detach().requires_grad_(True)
ori = up_input.clone().detach().requires_grad_(False)

padding = int((kernel_size-1)/2)
# MRC_layer = custom_maxpool2d(in_channels= 1, kernel = kernel, stride = 1, padding = padding, dilation = 1)


# # # ### For space
# over = MRC_layer.get_maxpool(up_input)
# under = 1-MRC_layer.get_maxpool(1-over)

# # ### For patter_size
# # # under = 1-MRC_layer.get_maxpool(1-up_input)
# # # over = MRC_layer.get_maxpool(under)

# plt.figure(),
# plt.imshow(up_input.numpy()[0,0,:,:]-under.numpy()[0,0,:,:])

# # plt.figure(),
# # plt.imshow(up_input.numpy()[0,0,:,:]-under.numpy()[0,0,:,:])


# plt.figure(),
# plt.imshow(up_input.numpy()[0,0,:,:])
# plt.figure(),
# plt.imshow(over.numpy()[0,0,:,:])
# plt.figure(),
# plt.imshow(under.numpy()[0,0,:,:])


L2 = nn.MSELoss(reduction='none')
avg = torch.nn.MaxPool2d(kernel_size, stride=1, padding=padding)
optimizer = optim.Adam([up_input] , lr=0.0005)
epoch = 2000

for i in range(epoch):
    ##For bridge
    # over1 = kornia.morphology.dilation(up_input, kernel = kernel)
    # under1 = kornia.morphology.erosion(over1, kernel = kernel)
    
    ## Under_Over : pinch remove
    over1 = -avg(-up_input)
    under1 = avg(over1)
    
    ## Over_under : bridge remove
    over2 = avg(up_input)
    under2 = -avg(-over2)
    
    
    # plt.figure(),
    # plt.imshow(up_input.numpy()[0,0,:,:])
    
    
    
    output = under1
    if i==0:
        output_1 = output.detach()
    
    # diff = L2(up_input, output)
    diff = L2(under1.detach(), under2)
    with torch.no_grad():
        T =diff.sum()
        print(f'{i}: {T}')
    
    # up_input.grad = None
    optimizer.zero_grad()
    diff.sum().backward()
    optimizer.step()
    

    # torch.no_grad()로 업데이트 수행
    # with torch.no_grad():
    #     if i<epoch-1:
    #         up_input -=  0.01 * up_input.grad 
    # up_input = up_input.clone().detach().requires_grad_(True)


# plt.figure(),
# plt.imshow(under1.detach().numpy()[0,0,:,:])
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10, 5), sharex=True, sharey=True)

# 4. 원본 A 플롯
st = 0
ax1.imshow(ori.detach().numpy()[0,0,:,st:])
ax1.set_title("Original input")
# ax1.axis("off")

# 5. A.grad 플롯
ax2.imshow(up_input.detach().numpy()[0,0,:,st:])
ax2.set_title("After over_under")
# ax2.axis("off")

ax3.imshow(diff.detach().numpy()[0,0,:,st:])
ax3.set_title("Loss")

ax4.imshow(up_input.grad.numpy()[0,0,:,st:])
ax4.set_title("Gradient of A")

# 6. 플롯 표시
plt.tight_layout()
plt.show()
