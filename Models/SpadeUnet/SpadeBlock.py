import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor
from .normalisation import SPADE
import torch.nn.utils.spectral_norm as spectral_norm
import torch.nn.functional as F

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

class SPADEResnetBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        learned_shortcut: bool = True,
    ) -> None:
        super().__init__()
        # if norm_layer is None:
        #     norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.learned_shortcut = learned_shortcut
        # (planes!=inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)


        # self.bn2 = norm_layer(planes)

        ## SpadeNorms
        self.norm_0 = SPADE(inplanes)
        self.norm_1 = SPADE(planes)
        if self.learned_shortcut:
            self.norm_s = SPADE(inplanes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor,seg:Tensor) -> Tensor:
        x_s = self.shortcut(x, seg)

        dx = self.conv1(self.actvn(self.norm_0(x, seg)))
        dx = self.conv2(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out
        # identity = x

        # out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)

        # out = self.conv2(out)
        # out = self.bn2(out)

        # if self.downsample is not None:
        #     identity = self.downsample(x)

        # out += identity
        # out = self.relu(out)

        # return out
    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)