import torch.nn as nn
import MinkowskiEngine as ME
from model.BasicBlock import Inception_ResNet
from model.transformer_ln import Pct

class knn_multiscale(ME.MinkowskiNetwork):
    CHANNELS = [32,64,32]
    BLOCK_1 = Inception_ResNet

    def __init__(self,
                 in_channels=1,
                 bn_momentum=0.1,
                 D=3):

        ME.MinkowskiNetwork.__init__(self, D)
        CHANNELS = self.CHANNELS
        BLOCK_1 = self.BLOCK_1

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=CHANNELS[0],
            kernel_size=7,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.block1 = self.make_layer(BLOCK_1, BLOCK_1, CHANNELS[0], bn_momentum=bn_momentum, D=D)
        self.Pct1 = Pct(input_channel=CHANNELS[0])

        self.down1 = ME.MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=CHANNELS[0],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.block1_down1 = self.make_layer(BLOCK_1, BLOCK_1, CHANNELS[0], bn_momentum=bn_momentum, D=D)
        self.Pct1_down1 = Pct(input_channel=CHANNELS[0])

        self.conv_down1 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[0],
            out_channels=CHANNELS[1],
            kernel_size=5,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.block2_down1 = self.make_layer(BLOCK_1, BLOCK_1, CHANNELS[1], bn_momentum=bn_momentum, D=D)
        self.Pct2_down1 = Pct(input_channel=CHANNELS[1])

        self.up1 = ME.MinkowskiConvolutionTranspose(
            in_channels=CHANNELS[1],
            out_channels=CHANNELS[2],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)

        self.down2_1 = ME.MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=CHANNELS[0],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.down2_2 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[0],
            out_channels=CHANNELS[1],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.block1_down2 = self.make_layer(BLOCK_1, BLOCK_1, CHANNELS[1], bn_momentum=bn_momentum, D=D)
        self.Pct1_down2 = Pct(input_channel=CHANNELS[1])

        self.conv_down2 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[1],
            out_channels=CHANNELS[2],
            kernel_size=5,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.block2_down2 = self.make_layer(BLOCK_1, BLOCK_1, CHANNELS[2], bn_momentum=bn_momentum, D=D)
        self.Pct2_down2 = Pct(input_channel=CHANNELS[2])

        self.up2_1 = ME.MinkowskiConvolutionTranspose(
            in_channels=CHANNELS[2],
            out_channels=CHANNELS[2],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)

        self.up2_2 = ME.MinkowskiConvolutionTranspose(
            in_channels=CHANNELS[2],
            out_channels=CHANNELS[2],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)

        self.conv3=ME.MinkowskiConvolution(
            in_channels=CHANNELS[2]*3,
            out_channels=CHANNELS[2]*2,
            kernel_size=5,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.block3=self.make_layer(BLOCK_1, BLOCK_1, CHANNELS[2]*2, bn_momentum=bn_momentum, D=D)
        self.Pct3 = Pct(input_channel=CHANNELS[2]*2)

        self.conv4 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[2] * 2,
            out_channels=CHANNELS[2] * 1,
            kernel_size=5,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.block4 = self.make_layer(BLOCK_1, BLOCK_1, CHANNELS[2] * 1, bn_momentum=bn_momentum, D=D)
        self.Pct4 = Pct(input_channel=CHANNELS[2] * 1)

        #
        self.fc1_offset = ME.MinkowskiConvolution(
            in_channels=CHANNELS[2],
            out_channels=128,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=3)
        self.fc2_offset = ME.MinkowskiConvolution(
            in_channels=128,
            out_channels=128,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=3)
        self.fc3_offset = ME.MinkowskiConvolution(
            in_channels=128,
            out_channels=3,
            kernel_size=1,
            stride=1,
            bias=True,dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)

    def make_layer(self, block_1, block_2, channels, bn_momentum, D):
        layers = []
        layers.append(block_1(channels=channels, bn_momentum=bn_momentum))
        layers.append(block_2(channels=channels, bn_momentum=bn_momentum))
        layers.append(block_1(channels=channels, bn_momentum=bn_momentum))

        return nn.Sequential(*layers)

    def forward(self, x):
        ###################### feature extraction #################################
        out =self.relu(self.conv1(x))
        out = self.block1(out)
        out_scale1 = self.Pct1(out)

        out = self.relu(self.down1(x))
        out = self.block1_down1(out)
        out = self.Pct1_down1(out)
        out = self.relu(self.conv_down1(out))
        out = self.block2_down1(out)
        out = self.Pct2_down1(out)
        out_scale2 = self.up1(out)

        out = self.relu(self.down2_1(x))
        out = self.relu(self.down2_2(out))
        out = self.block1_down2(out)
        out = self.Pct1_down2(out)

        out = self.relu(self.conv_down2(out))
        out = self.block2_down2(out)
        out = self.Pct2_down2(out)
        out_scale3 = self.up2_1(self.relu(self.up2_2(out)))

        out = ME.cat(out_scale1,out_scale2,out_scale3)

        out = self.relu(self.conv3(out))
        out = self.block3(out)
        out = self.Pct3(out)

        out = self.relu(self.conv4(out))
        out = self.block4(out)
        out = self.Pct4(out)
        out_gt = out
        ###################### offset ################################
        out_offset = self.relu(self.fc1_offset(out))
        out_offset = self.relu(self.fc2_offset(out_offset))
        out_offset = self.fc3_offset(out_offset)
        # print(out_offset.F)
        offset = out_offset.F.float()
        # print(torch.max(offset))
        out_C_offset = out_gt.C[:, 1:] + offset
        out_C_offset = out_C_offset.unsqueeze(0)

        return  out_C_offset,out_offset