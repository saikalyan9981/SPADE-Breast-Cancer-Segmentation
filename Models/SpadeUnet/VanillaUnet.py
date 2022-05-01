
from .TigerBlocks import conv_block
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p

class VanillaUnet(nn.Module, smp.encoders._base.EncoderMixin):
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
        #self.e5 = encoder_block(256, 512)
        # self.reset_parameters()

    def forward(self, x: torch.Tensor):
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
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)        

        return [x, p1, p2, p3, p4]