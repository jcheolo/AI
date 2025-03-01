# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 03:01:51 2025

@author: JCH
"""

from perlin_noise import PerlinNoise
import numpy as np
import cv2
import matplotlib.pyplot as plt


# def generate_worm_perlin(image, num_worms, mask):
    
#     noise = PerlinNoise(octaves=3)
#     y_indices, x_indices = np.where(mask == 1)
#     indices = np.random.choice(len(x_indices), size=num_worms, replace=False)
    
#     for i in range(num_worms):
        
#         random_x = x_indices[indices]
#         random_y = y_indices[indices]

#         # x, y = np.random.randint(100, 900), np.random.randint(100, 900)
#         worm = []
#         #length = np.random.randint(20, 50)
#         #scale = np.random.randint(10, 50)
#         length = 50
#         scale = 200
#         for _ in range(length):  # 지렁이 길이
#             dx = int(noise([x / scale, y / scale])*20)
#             dy = int(noise([y / scale, x / scale])*20)
#             x, y = x + dx, y + dy
#             worm.append((x, y))
        
#         cv2.polylines(image, [np.array(worm, dtype=np.int32)], isClosed=False, color=1, thickness=5)

#     return image

image_size = 1024
image = np.zeros( (image_size,image_size) , dtype=np.float32)


mask = np.random.choice( [0, 1], size=(image_size, image_size), p=[0.9, 0.1])
# y1, x1 = np.where(mask == 1)
y0, x0 = np.where(mask == 0)

num_worms = int( len(y0)/(image_size*image_size) * 100)  # 뽑을 개수

indices = np.random.choice(len(x0), size=num_worms, replace=False)
noise = PerlinNoise(octaves=4)

for ind in indices:
    x = x0[ind]
    y = y0[ind]
    
    # length = 50
    # scale = 200
    # thick = 5
    length = np.random.randint(20, 50)
    # scale = np.random.randint(10, 20)
    scale = 15
    normalize = int(image_size/2)
    thick = np.random.randint(2,10)
    
    worm = []
    for _ in range(length):
        dx = int(noise([x / normalize, y / normalize])*scale)
        dy = int(noise([y / normalize, x / normalize])*scale)
        x, y = x + dx, y + dy
        worm.append((x, y))
        
    cv2.polylines(image, [np.array(worm, dtype=np.int32)], isClosed=False, color=1, thickness=thick)
    

# image = generate_worm_perlin(image, num_worms, mask)

ksize = (5,5)
sigmaX = 2

image = cv2.GaussianBlur(image, ksize, sigmaX)



# image = generate_worm_perlin()
# cv2.imwrite("worm_perlin.png", (image * 255).astype(np.uint8))

plt.imshow(image)