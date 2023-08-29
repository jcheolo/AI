# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 22:21:00 2023

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
import seaborn as sns

from ssim import SSIM
from gradient.modules import GradientCorrelationLoss2d



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

def create_box_image(imagex, imagey, width, height, cenx, ceny):
    I = np.zeros([imagey, imagex])
    sx = int(cenx-width/2)
    ex = int(sx+width)
    sy = int(ceny-height/2)
    ey = int(sy+height)
    I[sy:ey,sx:ex]=1
    return I

def create_random_box_image(imagex, imagey, width, height, cenx, ceny):
    I = np.zeros([imagey, imagex])
    sx = int(cenx-width/2)
    ex = int(sx+width)
    sy = int(ceny-height/2)
    ey = int(sy+height)
    I[sy:ey,sx:ex]=1
    
    shift_size = 5
    shiftX = np.random.randint(-shift_size, shift_size+1)
    shiftY = np.random.randint(-shift_size, shift_size+1)
    I =np.roll(I, (shiftX, shiftY), axis=(1, 0))
    return I, float(shiftX), float(shiftY)


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

class FFTLoss(nn.Module):
    def __init__(self):
        super(FFTLoss, self).__init__()
        self.L1 = nn.L1Loss()
        self.L2 = nn.MSELoss()

    def forward(self, inputs, targets, smooth=1):
        
        #flatten label and prediction tensors
        inputs = torch.fft.rfft2(inputs)
        targets = torch.fft.rfft2(targets)
        
        amp = self.L1(torch.abs(inputs),torch.abs(targets))
        phase = self.L1(inputs/torch.abs(inputs),targets/torch.abs(targets))
        loss = amp+phase*1       
        return loss


width = 11  # 이미지 너비
height = 11  # 이미지 높이
mu_x = 5  # Gaussian 중심 x 좌표
mu_y = 5  # Gaussian 중심 y 좌표
sigma_x = 1  # Gaussian x 축 표준편차
sigma_y = 1  # Gaussian y 축 표준편차

gaussian_image = create_gaussian_image(width, height, mu_x, mu_y, sigma_x, sigma_y)

bdataList = []
n = 50
realB = torch.zeros(n, 1, 64, 64)
mean_shiftX = 0
mean_shiftY = 0

for i in range(n):
    box, shiftX, shiftY = create_random_box_image(64,64,40,20,32,32)
    box = convolution_2d_with_padding(box, gaussian_image, 5)
    realB[i,0,:,:]= torch.Tensor(box)
    mean_shiftX += shiftX
    mean_shiftY += shiftY
    
mean_shiftX = mean_shiftX/n
mean_shiftY = mean_shiftY/n
    
plt.figure()
plt.title(f'mean shift X,Y = {mean_shiftX},{mean_shiftX}')
plt.imshow((realB.sum(dim=0))[0])
## b : (n,1,64,64)
    

L2 = nn.MSELoss()  # 평균 제곱 오차 손실 함수
L1 = nn.L1Loss()
FFT = FFTLoss()
GL = GradientCorrelationLoss2d()
a = nn.L1Loss(reduction='none')
ssim = SSIM(return_msssim=False, L=1, padding=0, ensemble_kernel=False, window_size=11, sigma=1.5, return_log=False)

# plt.figure()
# plt.imshow(b1[0,0,:,:])


### For box shift

shift_range = 8
shift = list(np.arange(-shift_range,shift_range+1,1))
Loss_shift = np.zeros((len(shift), len(shift)))

for x in shift:
    for y in shift:
        box = create_box_image(64,64,40,20,32+x,32-y)
        box = convolution_2d_with_padding(box, gaussian_image, 5)
        sameB = torch.zeros(n, 1, 64, 64)
        for i in range(n):
            sameB[i,0,:,:]= torch.Tensor(box)
               
        ### Loss
        loss = L1(realB,sameB)*0 + L2(realB,sameB)*1 + (1-ssim(realB,sameB))*0
        
        Loss_shift[shift_range-y,shift_range+x] = loss

plt.figure()
plt.imshow(Loss_shift, extent = (-shift_range-0.5,shift_range+0.5,-shift_range-0.5,shift_range+0.5) )

plt.title('shift loss')
#plt.xlim(-shift_range-0.5, shift_range+0.5)  # x축 범위: 0에서 6까지
#plt.ylim(-shift_range-0.5, shift_range+0.5)
plt.xlabel('X-shift')
plt.ylabel('Y-shift')

    # eps = 1e-4
    # buffer = 0.01
    # B=b+eps
    # B1=b1+eps
    # B2=b2+eps
    
    # loss1 = L1(b,b1)*1 + L2(b,b1)*10 + (1-ssim(b,b1))*0 + abs((b-b1).mean())*5
    # loss2 = L1(b,b2)*1 + L2(b,b2)*10 + (1-ssim(b,b2))*0 + abs((b-b2).mean())*5
    
    
    # loss1 = (((B+B1)/(2*torch.sqrt(B*B1)) - 1)*((B1+B)/2)).mean()
    # loss2 = (((B+B2)/(2*torch.sqrt(B*B2)) - 1)*((B2+B)/2)).mean()
    
    #loss1 = ( ((b+b1+0.1)/(2*torch.sqrt(b*b1)+0.1) - 1) ).mean()
    #loss2 = ( ((b+b2+0.1)/(2*torch.sqrt(b*b2)+0.1) - 1) ).mean()
    
    #loss1 = ( torch.log((b+b1+0.0001)/(2*torch.sqrt(b*b1)+0.0001)) ).mean()
    #loss2 = ( torch.log((b+b2+0.0001)/(2*torch.sqrt(b*b2)+0.0001)) ).mean()
    
    # loss1 = ( ((B+B1+buffer)/(2*torch.sqrt(B*B1)+buffer) - 1) ).mean()
    # loss2 = ( ((B+B2+buffer)/(2*torch.sqrt(B*B2)+buffer) - 1) ).mean()
    
    #loss1 = ( ((B+B1+eps)/(2*torch.sqrt(B*B1)+0.0) - 1) ).mean()
    #loss2 = ( ((B+B2+eps)/(2*torch.sqrt(B*B2)+0.0) - 1) ).mean()







# ### For box size
# Loss_size = []
# temp = []
# size = list(np.arange(-14,14+1,2))

# for i in size:
#     box = create_box_image(64,64,40+i,20,32,32)
#     bdata = convolution_2d_with_padding(box, gaussian_image, 5)
#     b = torch.zeros(1, 1, 64, 64)
#     b[0,0,:,:]= torch.Tensor(bdata)

#     eps = 1e-4
#     buffer = 0.01 #or 0.1
#     B=b+eps
#     B1=b1+eps
    
#     #loss1 = L1(b,b1)*0 + L2(b,b1)*0+ (1-ssim(b,b1))*1 + abs((b-b1).mean())*0
#     # loss2 = L1(b,b2)*0 + L2(b,b2)*10 + (1-ssim(b,b2))*0 + abs((b-b2).mean())*0
    
#     #loss1 = ( ((b+b1+0.1)/(2*torch.sqrt(b*b1)+0.1) - 1) ).mean()
    
#     #loss1 = ( torch.log((b+b1+0.0001)/(2*torch.sqrt(b*b1)+0.0001)) ).mean()
    
#     #loss1 = ( ((B+B1+0)/(2*torch.sqrt(B*B1)+0) - 1) * (torch.min(B,B1)+0.1) ).mean()
#     loss1 = ( ((B+B1+buffer)/(2*torch.sqrt(B*B1)+buffer) - 1) ).mean()
#     #loss1 = ( torch.log(((B+B1+0.01)/(2*torch.sqrt(B*B1)+0.01))) ).mean()
    
#     # loss1 = ((2*(B*B+B1*B1))/((B+B1)*(B+B1)) -1 ).mean()
    
#     loss = loss1 #+loss2
#     #loss = FFT(b,b1)+FFT(b,b2)
#     Loss_size.append(loss)
    
#     c = a(b,b1).flatten()
#     temp.append(c)

# # B = np.arange(0.0001,1,0.0001)
# # B1 = np.arange(0.0001,1,0.0001)
# # B, B1 = np.meshgrid(B, B1)
# # losss = ( ((B+B1+0.1)/(2*np.sqrt(B*B1)+0.1) - 1) )

# # plt.figure()
# # sns.scatterplot(x=size, y=Loss_size)
# # plt.gca().invert_yaxis()
# # plt.grid()


    
# # plt.figure()
# # sns.scatterplot(x=size, y=Loss_shift)





# plt.figure(figsize=(8, 4))  # 전체 그림 크기 설정 (가로 12, 세로 4)
# plt.subplot(1, 2, 1)  # 1행 3열의 그리드에서 첫 번째 서브플롯
# sns.scatterplot(x=size, y=Loss_size)
# plt.gca().invert_yaxis()
# plt.grid()

# plt.subplot(1, 2, 2)  # 1행 3열의 그리드에서 두 번째 서브플롯
# sns.scatterplot(x=shift, y=Loss_shift)
# plt.gca().invert_yaxis()
# plt.grid()