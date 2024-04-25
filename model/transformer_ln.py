import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
import numpy as np
import math

from pytorch3d.ops import knn_points,knn_gather


class Pct(nn.Module):
    def __init__(self, input_channel):
        super(Pct, self).__init__()
        self.SA=self_attention(input_channel)
        self.SA_1=self_attention(input_channel)
        self.Linear = ME.MinkowskiConvolution(
            in_channels=input_channel,
            out_channels=input_channel,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=3)
        self.pos = nn.Sequential(
            nn.Linear(3, input_channel)
        )
        self.pos_1 = nn.Sequential(
            nn.Linear(3, input_channel)
        )
        self.relu = ME.MinkowskiReLU(inplace=True)
        # self.bn = ME.MinkowskiBatchNorm(input_channel)
        self.norm = torch.nn.LayerNorm(input_channel,input_channel)
        # self.bn_1 = ME.MinkowskiBatchNorm(input_channel)
        self.norm_1 = torch.nn.LayerNorm(input_channel,input_channel)

    def forward(self, x):
        #knn
        x_C=x.C.unsqueeze(0).float()
        x_F=x.F.unsqueeze(0).float()
        dist,idx,_=knn_points(x_C,x_C,K=16,return_nn=False,return_sorted=True)

        xyz = x_C.squeeze(0)[:, 1:]
        new_xyz = knn_gather(x_C[:, :, 1:], idx).squeeze(0)
        new_xyz = xyz[:, None, :] - new_xyz[:, :, :]
        xyz_enc=self.pos(new_xyz)

        new_feature = knn_gather(x_F,idx).squeeze(0)
        #knn
        out = self.SA(x,xyz_enc,new_feature)#self-attention

        # out=self.SA(x,new_xyz,new_feature,head,d_k)#multi-head self-attention

        # knn
        out_F = out.F.unsqueeze(0).float()
        new_feature = knn_gather(out_F, idx).squeeze(0)
        xyz_enc_1=self.pos_1(new_xyz)
        # knn
        out = self.SA_1(out,xyz_enc_1,new_feature)#self-attention

        #out = self.SA_1(out, new_xyz,new_feature,head,d_k)#multi-head self-attention
        #skip-connettion
        out_F = self.norm(x.F + out.F)
        out = ME.SparseTensor(out_F,coordinate_map_key=out.coordinate_map_key,
                              coordinate_manager=out.coordinate_manager,
                              device=out.device)
        out_1 = self.Linear(out)
        out_F = self.norm_1(out.F+out_1.F)
        out = ME.SparseTensor(out_F, coordinate_map_key=out.coordinate_map_key,
                              coordinate_manager=out.coordinate_manager,
                              device=out.device)
        # out = self.bn_1(out_1 + out)
        # print('enc_0')

        return out
class self_attention(nn.Module):
    def __init__(self,channels):
        super(self_attention, self).__init__()
        self.q_conv = torch.nn.Linear(channels, channels)
        self.k_conv = torch.nn.Linear(channels, channels)
        self.v_conv = torch.nn.Linear(channels, channels)
        self.d = math.sqrt(channels)


    def forward(self, x, xyz_enc, new_feature):
        x_q = x.F
        Q = self.q_conv(x_q)
        new_feature=new_feature+xyz_enc
        K = self.k_conv(new_feature)
        K = K.permute(0, 2, 1)
        attention_map = torch.einsum('ndk,nd->nk', K, Q)
        attention_map = F.softmax(attention_map / self.d, dim=-1)
        # print(attention_map)
        V = self.v_conv(new_feature)
        attention_feature = torch.einsum('nk,nkd->nd', attention_map, V)
        x = ME.SparseTensor(features=attention_feature, coordinate_map_key=x.coordinate_map_key,
                                    coordinate_manager=x.coordinate_manager)
        return x

class mh_attention(nn.Module):
    def __init__(self,channels):
        super(mh_attention, self).__init__()
        self.q_conv = torch.nn.Linear(channels, channels)
        self.k_conv = torch.nn.Linear(channels, channels)
        self.v_conv = torch.nn.Linear(channels, channels)
        self.d = math.sqrt(channels)

    def forward(self, x,xyz_enc,new_feature,head,d_k):
        x_q = x.F
        k=new_feature.shape[1]
        x_q=self.q_conv(x_q).view(-1,head,d_k)
        new_feature=new_feature+xyz_enc
        x_k=self.k_conv(new_feature).view(-1,head,k,d_k)
        x_v=self.v_conv(new_feature).view(-1,head,k,d_k)
        att = torch.einsum('nhd,nhkd->nhk',x_q,x_k)
        att = F.softmax(att/self.d,dim=-1)
        # print(att)
        attention_new = torch.einsum('nhk,nhkd->nhd', att, x_v).view(-1,d_k*head)
        x_att = ME.SparseTensor(features=attention_new, coordinate_map_key=x.coordinate_map_key,
                                coordinate_manager=x.coordinate_manager, device=x.device)

        return x_att

