import segmentation_models_pytorch as smp
import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
            
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class res_block(nn.Module):
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
        
#https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035

class spade_block(nn.Module):
    def __init__(self, conv_cout):
        super().__init__()
        
        self.conv = nn.Conv2d(conv_cout, conv_cout, kernel_size=3, padding=1,bias=False)
        self.conv_gamma = nn.Conv2d(conv_cout, conv_cout, kernel_size=3, padding=1,bias=False)        
        self.conv_beta = nn.Conv2d(conv_cout, conv_cout, kernel_size=3, padding=1,bias=False)
                
        self.conv1 = nn.Sequential(self.conv, nn.ReLU())
                
    def forward(self, *inputs):
        x = inputs[0]
        lgx = inputs[1]

        y = self.conv1(lgx)
        gamma = self.conv_gamma(y)
        beta = self.conv_beta(y)
                
        return x * gamma + beta

class spadeRes_block(nn.Module):
    def __init__(self, conv_cout):
        super().__init__()
        
        self.spade_block1 = spade_block(conv_cout)
        self.spade_block2 = spade_block(conv_cout)
        self.spade_blockres = spade_block(conv_cout)

        self.conv1 = nn.Conv2d(conv_cout, conv_cout, kernel_size=3, padding=1,bias=False)  
        self.conv2 = nn.Conv2d(conv_cout, conv_cout, kernel_size=3, padding=1,bias=False) 
        self.convres = nn.Conv2d(conv_cout, conv_cout, kernel_size=3, padding=1,bias=False) 
        
        self.spade_out1 = nn.Sequential(self.conv1, nn.ReLU())
        self.spade_out2 = nn.Sequential(self.conv2, nn.ReLU())
        self.spade_outres = nn.Sequential(self.convres, nn.ReLU())
                
    def forward(self, *inputs):
        
        x = inputs[0]
        lgx = inputs[1]

        x1 = self.spade_block1(*(x,lgx))
        x1 = self.spade_out1(x1)

        x2 = self.spade_block2(*(x1,lgx))
        x2 = self.spade_out2(x2)

        xres = self.spade_blockres(*(x,lgx)) 
        xres = self.spade_outres(xres)
                
        return x2 + xres

class spadeSCRes_block(nn.Module):
    def __init__(self, conv_cout):
        super().__init__()
        
        self.spade_block1 = spade_block(conv_cout)
        self.spade_block2 = spade_block(conv_cout)
        
        self.conv1 = nn.Conv2d(conv_cout, conv_cout, kernel_size=3, padding=1,bias=False)  
        self.conv2 = nn.Conv2d(conv_cout, conv_cout, kernel_size=3, padding=1,bias=False) 
                
        self.spade_out1 = nn.Sequential(self.conv1, nn.ReLU())
        self.spade_out2 = nn.Sequential(self.conv2, nn.ReLU())
        
                
    def forward(self, *inputs):
        
        x = inputs[0]
        lgx = inputs[1]

        x1 = self.spade_block1(*(x,lgx))
        x1 = self.spade_out1(x1)

        x2 = self.spade_block2(*(x1,lgx))
        x2 = self.spade_out2(x2)
                
        return x2 + x

class spadeRes_block(nn.Module):
    def __init__(self, conv_cout):
        super().__init__()
        
        self.spade_block1 = spade_block(conv_cout)
        self.spade_block2 = spade_block(conv_cout)
        self.spade_blockres = spade_block(conv_cout)
        
        self.conv1 = nn.Conv2d(conv_cout, conv_cout, kernel_size=3, padding=1,bias=False)  
        self.conv2 = nn.Conv2d(conv_cout, conv_cout, kernel_size=3, padding=1,bias=False) 
        self.convres = nn.Conv2d(conv_cout, conv_cout, kernel_size=3, padding=1,bias=False) 
                
        self.spade_out1 = nn.Sequential(self.conv1, nn.ReLU())
        self.spade_out2 = nn.Sequential(self.conv2, nn.ReLU())
        self.spade_outres = nn.Sequential(self.convres, nn.ReLU())
                
    def forward(self, *inputs):
        
        x = inputs[0]
        lgx = inputs[1]

        x1 = self.spade_block1(*(x,lgx))
        x1 = self.spade_out1(x1)

        x2 = self.spade_block2(*(x1,lgx))
        x2 = self.spade_out2(x2)

        xres = self.spade_blockres(*(x,lgx)) 
        xres = self.spade_outres(xres)
                
        return x2 + xres

class Encoder_block(nn.Module):
    def __init__(self, in_c, out_c, experiment_name='res_spadeSCRes'):
        super().__init__()

        if experiment_name == 'res_spadeSCRes':
          self.conv_block = res_block(in_c, out_c)
          self.context_block = spadeSCRes_block(out_c)
        elif experiment_name == 'res_spadeRes':
          self.conv_block = res_block(in_c, out_c)
          self.context_block = spadeRes_block(out_c)
        elif experiment_name == 'conv_spadeRes':
          self.conv_block = conv_block(in_c, out_c)
          self.context_block = spadeRes_block(out_c)          
        elif experiment_name == 'conv_spadeSCRes':
          self.conv_block = conv_block(in_c, out_c)
          self.context_block = spadeSCRes_block(out_c)

        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1,bias=False)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d((2, 2))

        self.lgx_conv = nn.Sequential(self.conv, self.relu, self.pool)
        self.xspade_conv_out = nn.Sequential(self.relu,self.pool)
                
    def forward(self, *inputs):

        x = inputs[0]
        lgx = inputs[1]

        x = self.conv_block(x)
        lgx = self.lgx_conv(lgx)

        xspade = self.context_block(*(x,lgx))

        xout = self.xspade_conv_out(xspade)
        
        return xout, lgx

class TigerUnet(nn.Module, smp.encoders._base.EncoderMixin):
    def __init__(self, **kwargs):
        super().__init__()
        self._out_channels: List[int] = [3, 32, 64, 128, 256]
        self._depth: int = 4
        self._in_channels: int = 3
        
        """ Encoder """        
        self.e1 = Encoder_block(3, 32)
        self.e2 = Encoder_block(32, 64)
        self.e3 = Encoder_block(64, 128)
        self.e4 = Encoder_block(128, 256)

    def forward(self, features):
        x = features[0]
        lgx = features[1]

        p1, sp1 = self.e1(*(x, lgx))
        p2, sp2 = self.e2(*(p1, sp1))
        p3, sp3 = self.e3(*(p2, sp2))
        p4, sp4 = self.e4(*(p3, sp3))
    
        return [x, p1, p2, p3, p4]


