# -*- coding: utf-8 -*-
"""
Created on Sat May 18 16:41:10 2024
https://github.com/loeweX/Custom-ConvLayers-Pytorch/blob/main/src/Conv2d.py
@author: JCH
"""

import math
import collections
from typing import Union, Tuple, Iterable

import torch
import torch.nn as nn
import numpy as np


def pair(x: Union[int, Iterable[int]]) -> Tuple[int, int]:
    """
    If input is iterable (e.g., list or tuple) of length 2, return it as tuple. If input is a single integer, duplicate
    it and return as a tuple.

    Arguments:
    x: Either an iterable of length 2 or a single integer.

    Returns:
    A tuple of length 2.
    """
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return tuple(np.repeat(x, 2))

class custom_maxpool2d(nn.Module):
    """
    Step-by-step implementation of a 2D convolutional layer.

    Arguments:
    in_channels (int): Number of input channels.
    kernel_size (int or tuple): Size of the convolutional kernel.
    stride (int or tuple, optional): Stride of the convolution. Default: 1
    padding (int or tuple, optional): Zero-padding added to the input. Default: 0
    dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
    bias (bool, optional): If True, adds a learnable bias to the output. Default: True
    """

    def __init__(
        self,
        in_channels: int,
        kernel: torch.Tensor,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        
    ):
        super(custom_maxpool2d, self).__init__()

        self.stride = pair(stride)
        self.padding = pair(padding)
        self.dilation = pair(dilation)
        self.kernel_size = (kernel.shape[-2], kernel.shape[-1])
        
        if len(kernel.shape)==3:
            assert kernel.shape[1] == in_channels, "kernel channels and in_channels are not same"
        elif len(kernel.shape)!=2:
            raise ValueError("support only 3D or 2D kernel")        
        kernel = kernel.view(1,-1)
        self.weight = nn.Parameter(kernel)
        self.weight.requires_grad_(False)
        
    def get_output_size(self, input_size: int, idx: int) -> int:
        return (input_size 
                + 2*self.padding[idx] 
                - self.dilation[idx]*(self.kernel_size[idx] - 1)
                - 1) // self.stride[idx] + 1
    
    def get_maxpool(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]

        # generate sliding block
        x = torch.nn.functional.unfold(
            x,
            self.kernel_size,
            padding=self.padding,
            stride=self.stride,
            dilation=self.dilation,
        )

        # Apply weight matrix (element-wise multiplication).
        output = torch.einsum("b i p, o i -> b o i p", x, self.weight)
        output, _ = output.max(dim=-2, keepdim=True)
        
        # Rearrange output to (b, c, h, w).
        output_height = self.get_output_size(height, 0)
        output_width = self.get_output_size(width, 1)
        output = output.view(output.shape[0], output.shape[1], output_height, output_width)

        return output
    
# torch.manual_seed(1)
# test = np.arange(1, 17)
# test = test.reshape((4, 4))
# test = torch.tensor(test, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
# test1 = np.arange(-16, 0)
# test1 = test1.reshape((4, 4))
# test1 = torch.tensor(test1, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# test = torch.cat((test, test1), 0)
# kernel = torch.tensor([[0., 1., 0.],[1., 1., 1.],[0., 1., 0.]])
# A = custom_maxpool2d(in_channels= 1, kernel = kernel, stride = 1, padding = 1, dilation = 1)


# X, output = A.get_maxpool(test)
# print(test)
# print(X)
# print(X.shape)
# print(A.weight)
# print(output)