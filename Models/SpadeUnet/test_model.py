import torch.nn as nn
import segmentation_models_pytorch as smp

from .normalisation import SPADE
"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from typing import Type, Any, Callable, Union, List, Optional


class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1,bias=False)
        self.norm1 = SPADE(out_c)
        # self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1,bias=False)
        self.norm2 = SPADE(out_c)

        # self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
            
    def forward(self, inputs,segmap):
        x = self.conv1(inputs)
        # x = self.bn1(x)
        x = self.norm1(x,segmap)
        x = self.relu(x)
        x = self.conv2(x)
        # x = self.bn2(x)
        x = self.norm2(x,segmap)

        x = self.relu(x)
        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))
    def forward(self, inputs,segmap):
        x = self.conv(inputs,segmap)
        p = self.pool(x)
        return p



# class SpadeBN(nn.Module):
#     def __init__(self, conv_cin, conv_cout):
#         super().__init__()
        
#         self.conv = nn.Conv2d(conv_cin, conv_cout, kernel_size=3, padding=1,bias=False)
#         self.conv_gamma = nn.Conv2d(conv_cout, conv_cout, kernel_size=3, padding=1,bias=False)        
#         self.conv_beta = nn.Conv2d(conv_cout, conv_cout, kernel_size=3, padding=1,bias=False)
#         self.pool = nn.MaxPool2d((2, 2))
        
#         self.conv1 = nn.Sequential(self.conv, nn.ReLU(), self.pool)
#         self.gamma = nn.Sequential(self.conv_gamma, nn.ReLU(),self.pool)
#         self.beta = nn.Sequential(self.conv_beta, nn.ReLU(),self.pool)
        
#     def forward(self, x):
        
#         y = self.conv1(x)
#         gamma = self.gamma(y)
#         beta = self.beta(y)
                
#         return y, gamma, beta
    
class SpadeUnetTest(nn.Module, smp.encoders._base.EncoderMixin):
    def __init__(self, **kwargs):
        super().__init__()
        
        # A number of channels for each encoder feature tensor, list of integers
        self._out_channels: List[int] = [3, 32, 64, 128, 256]

        # A number of stages in decoder (in other words number of downsampling operations), integer
        # use in in forward pass to reduce number of returning features
        self._depth: int = 4

        # Default number of input channels in first Conv2d layer for encoder (usually 3)
        self._in_channels: int = 3
        
        """ Encoder """
        self.e1 = encoder_block(3, 32)
        self.e2 = encoder_block(32, 64)
        self.e3 = encoder_block(64, 128)
        self.e4 = encoder_block(128, 256)
        
        # self.sp1 = SpadeBN(3, 32)
        # self.sp2 = SpadeBN(32, 64)
        # self.sp3 = SpadeBN(64, 128)
        # self.sp4 = SpadeBN(128, 256)
        #self.e5 = encoder_block(256, 512)
        # self.reset_parameters()

    def forward(self, *features):
        """Produce list of features of different spatial resolutions, each feature is a 4D torch.tensor of
        shape NCHW (features should be sorted in descending order according to spatial resolution, starting
        with resolution same as input `x` tensor).

        Input: `x` with shape (1, 3, 64, 64)
        Output: [f0, f1, f2, f3, f4, f5] - features with corresponding shapes
                [(1, 3, 64, 64), (1, 64, 32, 32), (1, 128, 16, 16), (1, 256, 8, 8),
                (1, 512, 4, 4), (1, 1024, 2, 2)] (C - dim may differ)
        also should support number of features according to specified depth, e.g. if depth = 5,
        number of feature tensors = 6 (one with same resolution as input and 5 downsampled),
        depth = 3 -> number of feature tensors = 4 (one with same resolution as input and 3 downsampled).
        """
        x = features[0]
        lgx = features[1]
        
        pe1 = self.e1(x,lgx)
        # y1, g1, b1 = self.sp1(lgx)
        
        # print('pe1 =', pe1.grad)
        # print('g1 =', g1.grad)
        # print('b1 =', b1.grad)
        
        # po1 = pe1*g1 + b1
        
        pe2 = self.e2(pe1,lgx)
        # y2, g2, b2 = self.sp2(y1)
        # po2 = pe2*g2 + b2
        
        pe3 = self.e3(pe2,lgx)
        # y3, g3, b3 = self.sp3(y2)
        # po3 = pe3*g3 + b3
        
        pe4 = self.e4(pe3,lgx)
        # y4, g4, b4 = self.sp4(y3)
        # po4 = pe4*g4 + b4
        
        return [x, pe1, pe2, pe3, pe4]

class SpadeSMPUnetTest(smp.Unet):
    
    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(*x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks