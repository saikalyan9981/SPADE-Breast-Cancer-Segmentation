from copy import deepcopy
from tkinter.tix import Tree

import torch.nn as nn

from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck
from pretrainedmodels.models.torchvision_models import pretrained_settings
import segmentation_models_pytorch as smp
from typing import Type, Any, Callable, Union, List, Optional
from .SpadeBlock import SPADEResnetBlock

class conv_Res_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
            
    def forward(self, inputs):
        sc = self.conv1(inputs)
        x = self.bn1(sc)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x+sc)
        x = self.relu(x)
        return x

class Encoder_block(nn.Module):
    def __init__(self, in_c, out_c,learned_shortcut=True):
        super().__init__()
        self.conv = conv_Res_block(in_c, out_c)
        self.context_block = SPADEResnetBlock(out_c,out_c,learned_shortcut=learned_shortcut)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d((2, 2))
        self.xspade_conv_out = nn.Sequential(self.relu,self.pool)
    def forward(self, inputs,segmap):
        x = self.conv(inputs)
        x = self.context_block(x,segmap)
        x = self.xspade_conv_out(x)
        return x


class SpadeResNetEncoder(nn.Module, smp.encoders._base.EncoderMixin):
    def __init__(self, learned_shortcut=True,**kwargs):
        super().__init__()
        self._out_channels: List[int] = [3, 32, 64, 128, 256]
        self._depth: int = 4
        self._in_channels: int = 3
        
        """ Encoder """        
        self.e1 = Encoder_block(3, 32,learned_shortcut=learned_shortcut)
        self.e2 = Encoder_block(32, 64,learned_shortcut=learned_shortcut)
        self.e3 = Encoder_block(64, 128,learned_shortcut=learned_shortcut)
        self.e4 = Encoder_block(128, 256,learned_shortcut=learned_shortcut)

    def forward(self, *features):
      
        x = features[0]
        lgx = features[1]

        p1 = self.e1(x, lgx)
        p2 = self.e2(p1, lgx)
        p3 = self.e3(p2, lgx)
        p4 = self.e4(p3, lgx)
    
        return [x, p1, p2, p3, p4]



class UEncoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_Res_block(in_c, out_c)
        self.context_block = BasicBlock(out_c,out_c)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d((2, 2))
        self.xspade_conv_out = nn.Sequential(self.relu,self.pool)
    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.context_block(x)
        x = self.xspade_conv_out(x)
        return x

class UResNetEncoder(nn.Module, smp.encoders._base.EncoderMixin):
    def __init__(self, **kwargs):
        super().__init__()
        self._out_channels: List[int] = [3, 32, 64, 128, 256]
        self._depth: int = 4
        self._in_channels: int = 3
        
        """ Encoder """        
        self.e1 = UEncoder_block(3, 32)
        self.e2 = UEncoder_block(32, 64)
        self.e3 = UEncoder_block(64, 128)
        self.e4 = UEncoder_block(128, 256)

    def forward(self, x):
      
        p1 = self.e1(x)
        p2 = self.e2(p1)
        p3 = self.e3(p2)
        p4 = self.e4(p3)
    
        return [x, p1, p2, p3, p4]
# class SpadeResNetEncoder(ResNet, smp.encoders._base.EncoderMixin):
#     def __init__(self, out_channels, depth=5, **kwargs):
#         super().__init__(**kwargs)
#         self._depth = depth
#         self._out_channels = out_channels
#         self._in_channels = 3

#         del self.fc
#         del self.avgpool

#     def get_stages(self):
#         return [
#             nn.Identity(),
#             nn.Sequential(self.conv1, self.bn1, self.relu),
#             # nn.Sequential(self.maxpool, self.layer1),
#             self.maxpool, 
#             self.layer1,
#             self.layer2,
#             self.layer3,
#             self.layer4,
#         ]

#     def forward(self, *in_features):
        
#         x = in_features[0]
#         lgx = in_features[1]
#         stages = self.get_stages()

#         features = []
#         for i in range(self._depth + 1):
#             if i<=1:
#                 x = stages[i](x)
#             elif i==2:
#                 x = stages[i](x)
#                 continue
#             else:
#                 x = stages[i](x,lgx)
#             features.append(x)

#         return features

#     def load_state_dict(self, state_dict, **kwargs):
#         state_dict.pop("fc.bias", None)
#         state_dict.pop("fc.weight", None)
#         super().load_state_dict(state_dict, **kwargs)

