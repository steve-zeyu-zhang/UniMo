from cmath import nan
import math
import sys
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vq.encdec import Encoder, Decoder
from models.vq.quantize import QuantizeEMAReset


class PointEncoder(nn.Module):
    def __init__(self, code_dim, nb_code, in_dim):
        super(PointEncoder, self).__init__()
        self.latent_dim = code_dim
        self.conv1 = torch.nn.Conv1d(in_dim, 256, 1)
        self.conv2 = torch.nn.Conv1d(256, 512, 1)
        self.conv3 = torch.nn.Conv1d(512, code_dim, 1)
        self.vq_encoder = Encoder(256, 256, down_t=2, stride_t=2, width=256, depth=3, dilation_growth_rate=3, activation='relu', norm=None)


    def forward(self, x):  # [B,L,N,3]->[B,L,latent_dim]

        B, L, N, channel = x.shape

        xyz = x.contiguous().view(B * L, N, channel)  # [B*L,N,3]

        xyz = xyz.permute(0, 2, 1)  # [B*L,3,N]

        xyz = F.relu(self.conv1(xyz))
        xyz = F.relu(self.conv2(xyz))
        xyz = self.conv3(xyz)
        # 最大池化 [B,latent_dim,N]->[B,latent_dim]
        z = torch.max(xyz, dim=2, keepdim=True)[0]
        z = z.view(-1, self.latent_dim)  # [B*L,latent_dim]
        z = z.view(B, L, self.latent_dim)  # [B,L,latent_dim]


        z = z.permute(0,2,1).float()
        z = self.vq_encoder(z)
        return z

class PointDecoder(nn.Module):
    def __init__(self, code_dim, nb_code, in_dim):
        super(PointDecoder, self).__init__()
        self.latent_dim = code_dim
        self.output_dim = in_dim
        self.num_points = 256


        self.vq_decoder = Decoder(256, 256, down_t=2, stride_t=2, width=256, depth=3, dilation_growth_rate=3, activation='relu', norm=None)


        self.fc_expand = nn.Linear(self.latent_dim, self.latent_dim * self.num_points)  
        self.fc_reshape = nn.Linear(self.latent_dim, 256) 


        self.deconv1 = nn.Conv1d(256, 512, 1)
        self.deconv2 = nn.Conv1d(512, 256, 1)
        self.deconv3 = nn.Conv1d(256, self.output_dim, 1)  

    def forward(self, quantized):  # [B, L, latent_dim] -> [B, L, N, 3]
        y = self.vq_decoder(quantized)  # [B, L, latent_dim]
        # y = quantized

        B, L, latent_dim = y.shape

        #[B, L, latent_dim * N]
        y = self.fc_expand(y)

        #  (B, L, N, latent_dim)
        y = y.view(B, L, self.num_points, latent_dim)  # [B, L, N, latent_dim]


        y = y.permute(0, 1, 3, 2)  # [B, L, latent_dim, N]
        y = self.fc_reshape(y)  # [B, L, 256, N]

        y = y.view(B * L, self.num_points, self.latent_dim)


        y = F.relu(self.deconv1(y))
        y = F.relu(self.deconv2(y))
        y = self.deconv3(y)  # [B, L, 3, N]


        y = y.view(B, L, self.output_dim, self.latent_dim)
        y = y.permute(0, 1, 3, 2)  # [B, L, N, 3]
        return y


class PVQVAE(nn.Module):
    def __init__(self, code_dim, nb_code, in_dim, mu):
        super().__init__()
        self.latent_dim = code_dim

        self.encoder = PointEncoder(code_dim, nb_code, in_dim)
        self.decoder = PointDecoder(code_dim, nb_code, in_dim)

        self.quantizer = QuantizeEMAReset(nb_code, code_dim, mu)
        # self._chamfer_density_loss=Chamfer_Density_Loss()

    def forward(self, x):
        
        z = self.encoder(x)

        # Quantization
        x_quantized, commit_loss, perplexity =  self.quantizer(z)
        # print(x_quantized.shape)

        # Decoder
        y = self.decoder(x_quantized)

        return y, commit_loss, perplexity

    def encode(self, x):
        z = self.encoder(x)
        z = z.permute(0, 2, 1)
        # Quantization
        x_quantized =  self.quantizer.quantize(z)
        # print(x_quantized.shape)
        return x_quantized
