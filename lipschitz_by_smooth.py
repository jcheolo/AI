# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 22:50:05 2025

@author: JCH
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# torch.manual_seed(3)

ratio = []
# 랜덤한 4D weight 생성 (예: N=2, C=1, H=5, W=5)
for i in range(100):
    N, C, H, W = 128, 128, 5, 5
    weight = torch.randn(N, C, H, W)  # 정규 분포로 초기화
    
    # L1-norm 계산 (변환 전)
    norm_before = torch.linalg.norm(weight.view(N, -1), ord=1)
    
    # Blur 필터 정의 (1,1,3,3) 크기
    c = 0.5
    blur_filter = torch.tensor([[[[c, c, c],
                                  [c, 1.0, c],
                                  [c, c, c]]]], dtype=torch.float32)
    
    # 필터 정규화 (전체 합이 1이 되도록)
    blur_filter = blur_filter / blur_filter.sum()
    blur_filter = blur_filter.repeat(C, 1, 1, 1)
    
    # Reflection padding (1,1) 적용
    weight_padded = F.pad(weight, (1, 1, 1, 1), mode='reflect')
    
    # F.conv2d 적용 (groups=C, stride=1)
    weight_blurred = F.conv2d(weight_padded, blur_filter, stride=1, padding=0, groups=C)
    
    # L1-norm 계산 (변환 후)
    norm_after = torch.linalg.norm(weight_blurred.view(N, -1), ord=1)
    
    # print(f"L1-norm before blur: {norm_before.item():.6f}")
    # print(f"L1-norm after  blur: {norm_after.item():.6f}")
    
    # plt.figure(),
    # plt.imshow(weight.numpy()[0,0,:,:])
    
    # plt.figure(),
    # plt.imshow(weight_blurred.numpy()[0,0,:,:])
    
    ratio.append(norm_before.item()/norm_after.item())
    
R = np.array(ratio).mean()
# plt.figure()
# sns.scatterplot(np.array(ratio))
print(R)


    