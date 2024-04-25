import torch
import torch.nn as nn
import MinkowskiEngine as ME
from model.BasicBlock import Inception_ResNet
import numpy as np
from model.transformer_ln import Pct
from util import sort_sparse_tensor
from util import create_new_sparse_tensor,scale_sparse_tensor,array2vector,topk_1


class GPCC_upsampling(torch.nn.Module):
    def __init__(self,  last_kernel_size=5):
        super().__init__()
        self.up = knn_up(last_kernel_size=last_kernel_size)
        self.offset = knn_offset()

    def forward(self, x, coords_T, device, prune=False,train_offset=False):
        scale_factor = 1/2
        x= scale_sparse_tensor(x,scale_factor,'round')
        x=sort_sparse_tensor(x)
        coords_T = coords_T*scale_factor
        if (train_offset==False):
            out, out_cls, target, keep = self.up(x, coords_T, device, prune=prune)
            out = scale_sparse_tensor(out, 1 / scale_factor, 'round')
            out_offset = 0
            return out, out_cls, target, keep, out_offset
        else:
            with torch.no_grad():
                out, out_cls, target, keep = self.up(x, coords_T, device, prune=prune)#######冻upsample网络参数#########
            out=scale_sparse_tensor(out,1/scale_factor,'round')
            out=sort_sparse_tensor(out)
            out_offset=0
            out_offset = ME.SparseTensor(features=torch.ones([out.C.shape[0], 1]).float(),
                                      coordinates=out.C,
                                      device=device)
            out_offset = self.offset(out_offset)
            return out, out_cls, target, keep, out_offset



class knn_offset(ME.MinkowskiNetwork):
    CHANNELS = [32, 64, 32]
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
            kernel_size=5,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.block1 = self.make_layer(BLOCK_1, BLOCK_1, CHANNELS[0], bn_momentum=bn_momentum, D=D)

        self.conv2 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[0],
            out_channels=CHANNELS[1],
            kernel_size=5,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.block2 = self.make_layer(BLOCK_1, BLOCK_1, CHANNELS[1], bn_momentum=bn_momentum, D=D)

        self.fc1_offset = ME.MinkowskiConvolution(
            in_channels=CHANNELS[1],
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
            bias=True, dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)

    def make_layer(self, block_1, block_2, channels, bn_momentum, D):
        layers = []
        layers.append(block_1(channels=channels, bn_momentum=bn_momentum))
        layers.append(block_2(channels=channels, bn_momentum=bn_momentum))
        layers.append(block_1(channels=channels, bn_momentum=bn_momentum))

        return nn.Sequential(*layers)

    def forward(self, x):
        ###################### feature extraction #################################
        out_gt=x
        out = self.relu(self.conv1(x))
        out = self.block1(out)
        out = self.relu(self.conv2(out))
        out = self.block2(out)
        ###################### offset ################################
        out_offset = self.relu(self.fc1_offset(out))
        out_offset = self.relu(self.fc2_offset(out_offset))
        out_offset = self.fc3_offset(out_offset)
        offset = out_offset.F.float()
        out_C_offset = out_gt.C[:, 1:] + offset
        out_C_offset = out_C_offset.unsqueeze(0)

        return out_C_offset

class knn_up(ME.MinkowskiNetwork):
    CHANNELS = [32, 64, 32]
    BLOCK_1 = Inception_ResNet

    def __init__(self,
                 in_channels=1,
                 bn_momentum=0.1,
                 last_kernel_size=5,
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
            stride=4,
            dilation=1,
            bias=False,
            dimension=D)
        self.block1_down2 = self.make_layer(BLOCK_1, BLOCK_1, CHANNELS[0], bn_momentum=bn_momentum, D=D)
        self.Pct1_down2 = Pct(input_channel=CHANNELS[0])

        self.conv_down2 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[0],
            out_channels=CHANNELS[1],
            kernel_size=5,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.block2_down2 = self.make_layer(BLOCK_1, BLOCK_1, CHANNELS[1], bn_momentum=bn_momentum, D=D)
        self.Pct2_down2 = Pct(input_channel=CHANNELS[1])

        self.up2_1 = ME.MinkowskiConvolutionTranspose(
            in_channels=CHANNELS[1],
            out_channels=CHANNELS[2],
            kernel_size=3,
            stride=4,
            dilation=1,
            bias=False,
            dimension=D)

        self.conv3 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[2] * 3,
            out_channels=CHANNELS[2] * 2,
            kernel_size=5,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.block3 = self.make_layer(BLOCK_1, BLOCK_1, CHANNELS[2] * 2, bn_momentum=bn_momentum, D=D)
        self.Pct3 = Pct(input_channel=CHANNELS[2] * 2)

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

        self.up = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=CHANNELS[2],
            out_channels=CHANNELS[2],
            kernel_size=last_kernel_size,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.block_up = self.make_layer(BLOCK_1, BLOCK_1, CHANNELS[2], bn_momentum=bn_momentum, D=D)
        self.conv_up = ME.MinkowskiConvolution(
            in_channels=CHANNELS[2],
            out_channels=CHANNELS[2],
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)

        self.classifier = ME.MinkowskiConvolution(
            in_channels=CHANNELS[2],
            out_channels=1,
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)

        self.pruning = ME.MinkowskiPruning()

        self.relu = ME.MinkowskiReLU(inplace=True)

    def make_layer(self, block_1, block_2, channels, bn_momentum, D):
        layers = []
        layers.append(block_1(channels=channels, bn_momentum=bn_momentum))
        layers.append(block_2(channels=channels, bn_momentum=bn_momentum))
        layers.append(block_1(channels=channels, bn_momentum=bn_momentum))

        return nn.Sequential(*layers)

    def get_target_by_sp_tensor(self, out, coords_T):
        with torch.no_grad():
            def ravel_multi_index(coords, step):
                coords = coords.long()
                step = step.long()
                coords_sum = coords[:, 0] \
                             + coords[:, 1] * step \
                             + coords[:, 2] * step * step \
                             + coords[:, 3] * step * step * step
                return coords_sum

            step = max(out.C.cpu().max(), coords_T.max()) + 1

            out_sp_tensor_coords_1d = ravel_multi_index(out.C.cpu(), step)
            target_coords_1d = ravel_multi_index(coords_T, step)
            # test whether each element of a 1-D array is also present in a second array.
            target = np.in1d(out_sp_tensor_coords_1d, target_coords_1d)

            return torch.Tensor(target).bool()

    def choose_keep(self, out, coords_T, device):
        with torch.no_grad():
            feats = torch.from_numpy(np.expand_dims(np.ones(coords_T.shape[0]), 1))
            x = ME.SparseTensor(features=feats, coordinates=coords_T, device=device)
            coords_nums = [len(coords) for coords in x.decomposed_coordinates]
            row_indices_per_batch = out._batchwise_row_indices
            keep = torch.zeros(len(out), dtype=torch.bool)
            for row_indices, ori_coords_num in zip(row_indices_per_batch, coords_nums):
                coords_num = min(len(row_indices), ori_coords_num)  # select top k points.
                values, indices = torch.topk(out.F[row_indices].squeeze(), int(coords_num))
                keep[row_indices[indices]] = True

        return keep

    def forward(self, x, coords_T,device, prune=True):
        ###################### feature extraction #################################
        out = self.relu(self.conv1(x))
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
        out = self.block1_down2(out)
        out = self.Pct1_down2(out)
        out = self.relu(self.conv_down2(out))
        out = self.block2_down2(out)
        out = self.Pct2_down2(out)
        out_scale3 = self.up2_1(out)

        out = ME.cat(out_scale1, out_scale2, out_scale3)

        out = self.relu(self.conv3(out))
        out = self.block3(out)
        out = self.Pct3(out)

        out = self.relu(self.conv4(out))
        out = self.block4(out)
        out = self.Pct4(out)

        ###################### upsample +classify+purning #########################
        out_gpcc_C=out.C
        out = self.relu(self.up(out))
        out = self.relu(self.conv_up(out))
        out = self.block_up(out)
        out_cls = self.classifier(out)
        target = self.get_target_by_sp_tensor(out, coords_T)
        keep = self.choose_keep(out_cls, coords_T,out_gpcc_C)
        if prune:
            out = self.pruning(out_cls, keep.cuda())

        return out, out_cls, target, keep

