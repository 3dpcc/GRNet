import torch.nn as nn
import MinkowskiEngine as ME


class ResNet(nn.Module):
    """
    Basic block: Residual
    """
    
    def __init__(self, channels):
        super(ResNet, self).__init__()
        #path_1
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=3)

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=3)
        
        self.relu = ME.MinkowskiReLU(inplace=True)
        
    def forward(self, x):
        out = self.relu(self.conv0(x))
        out = self.conv1(out)
        out += x

        return out


class Inception_ResNet(nn.Module):
    def __init__(self,
                 channels,
                 stride=1,
                 dilation=1,
                 bn_momentum=0.1,
                 dimension=3):
        super(Inception_ResNet, self).__init__()
        assert dimension > 0

        self.conv1 = ME.MinkowskiConvolution(
            channels, channels//4, kernel_size=1, stride=stride, dilation=dilation, bias=True, dimension=dimension)
        self.norm1 = ME.MinkowskiBatchNorm(channels//4, momentum=bn_momentum)
        self.conv2 = ME.MinkowskiConvolution(
            channels//4, channels//4, kernel_size=3, stride=stride, dilation=dilation, bias=True, dimension=dimension)
        self.norm2 = ME.MinkowskiBatchNorm(channels//4, momentum=bn_momentum)
        self.conv3 = ME.MinkowskiConvolution(
            channels//4, channels//2, kernel_size=1, stride=stride, dilation=dilation, bias=True, dimension=dimension)
        self.norm3 = ME.MinkowskiBatchNorm(channels//2, momentum=bn_momentum)
        
        self.conv4 = ME.MinkowskiConvolution(
            channels, channels//4, kernel_size=3, stride=stride, dilation=dilation, bias=True, dimension=dimension)
        self.norm4 = ME.MinkowskiBatchNorm(channels//4, momentum=bn_momentum)
        self.conv5 = ME.MinkowskiConvolution(
            channels//4, channels//2, kernel_size=3, stride=stride, dilation=dilation, bias=True, dimension=dimension)
        self.norm5 = ME.MinkowskiBatchNorm(channels//2, momentum=bn_momentum)
        
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        # 1
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.norm3(out)
        out = self.relu(out)
        
        # 2
        out1 = self.conv4(x)
        out1 = self.norm4(out1)
        out1 = self.relu(out1)
        
        out1 = self.conv5(out1)
        out1 = self.norm5(out1)
        out1 = self.relu(out1)

        # 3
        out2 = ME.cat(out,out1)
        out2 += x

        return out2


