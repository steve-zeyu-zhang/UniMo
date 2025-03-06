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
# from model.upsample_utils import bilateral_block_l1,bilateral_block_l2,bilateral_block_l3
from utils.chamfer_loss import ChamferLoss, Chamfer_Density_Loss
from utils.furthestpointsample import farthest_point_sample, index_points
from models.vq.encdec import Encoder, Decoder
from models.vq.quantize import QuantizeEMAReset
from models_pcmrl.pc_mrl import PointCloudDecoder


class PointEncoder(nn.Module):
    def __init__(self, args):
        super(PointEncoder, self).__init__()
        self.latent_dim = args.latent_dim
        self.conv1 = torch.nn.Conv1d(args.input_dim, 256, 1)
        self.conv2 = torch.nn.Conv1d(256, 512, 1)
        self.conv3 = torch.nn.Conv1d(512, args.latent_dim, 1)
        self.vq_encoder = Encoder(256, 256, down_t=2, stride_t=2, width=256, depth=3, dilation_growth_rate=3, activation='relu', norm=None)


    def forward(self, batch):  # [B,L,N,3]->[B,L,latent_dim]
        xyz = batch["xyz"].cuda()  # [B,L,N,3]

        B, L, N, channel = batch["xyz"].shape

        xyz = xyz.contiguous().view(B * L, N, channel)  # [B*L,N,3]

        xyz = xyz.permute(0, 2, 1)  # [B*L,3,N]

        xyz = F.relu(self.conv1(xyz))
        xyz = F.relu(self.conv2(xyz))
        xyz = self.conv3(xyz)
        # 最大池化 [B,latent_dim,N]->[B,latent_dim]
        point_encoder_z = torch.max(xyz, dim=2, keepdim=True)[0]
        point_encoder_z = point_encoder_z.view(-1, self.latent_dim)  # [B*L,latent_dim]
        point_encoder_z = point_encoder_z.view(B, L, self.latent_dim)  # [B,L,latent_dim]


        point_encoder_z = point_encoder_z.permute(0,2,1).float()
        point_encoder_z = self.vq_encoder(point_encoder_z)
        return point_encoder_z

class PointDecoder(nn.Module):
    def __init__(self, args):
        super(PointDecoder, self).__init__()
        self.latent_dim = args.latent_dim
        self.output_dim = args.input_dim  # 这里应该是3，因为点云是 xyz
        self.num_points = 256

        # VQ 解码部分
        self.vq_decoder = Decoder(256, 256, down_t=2, stride_t=2, width=256, depth=3, dilation_growth_rate=3, activation='relu', norm=None)

        # MLP 先扩展点数
        self.fc_expand = nn.Linear(self.latent_dim, self.latent_dim * self.num_points)  # 先拉成 N 个点
        self.fc_reshape = nn.Linear(self.latent_dim, 256)  # 变换维度

        # 1D 反卷积
        self.deconv1 = nn.Conv1d(256, 512, 1)
        self.deconv2 = nn.Conv1d(512, 256, 1)
        self.deconv3 = nn.Conv1d(256, self.output_dim, 1)  # 输出 xyz 坐标

    def forward(self, quantized):  # [B, L, latent_dim] -> [B, L, N, 3]
        point_decoder_z = self.vq_decoder(quantized)  # [B, L, latent_dim]

        B, L, latent_dim = point_decoder_z.shape

        # **扩展成 N 维** -> [B, L, latent_dim * N]
        point_decoder_z = self.fc_expand(point_decoder_z)

        # **Reshape 成 (B, L, N, latent_dim)**
        point_decoder_z = point_decoder_z.view(B, L, self.num_points, latent_dim)  # [B, L, N, latent_dim]

        # **调整通道顺序** 以进行 1D 卷积
        point_decoder_z = point_decoder_z.permute(0, 1, 3, 2)  # [B, L, latent_dim, N]
        point_decoder_z = self.fc_reshape(point_decoder_z)  # [B, L, 256, N]

        point_decoder_z = point_decoder_z.view(B * L, self.num_points, self.latent_dim)

        # **1D 卷积恢复点云特征**
        point_decoder_z = F.relu(self.deconv1(point_decoder_z))
        point_decoder_z = F.relu(self.deconv2(point_decoder_z))
        point_decoder_z = self.deconv3(point_decoder_z)  # [B, L, 3, N]

        # **调整回 (B, L, N, 3)**
        point_decoder_z = point_decoder_z.view(B, L, self.output_dim, self.latent_dim)
        point_decoder_z = point_decoder_z.permute(0, 1, 3, 2)  # [B, L, N, 3]
        return point_decoder_z


class PCMG(nn.Module):
    def __init__(self, args):
        super(PCMG, self).__init__()
        self.density_k = args.density_k

        self.latent_dim = args.latent_dim

        self.encoder = PointEncoder(args)
        self.decoder = PointDecoder(args)
        args.mu = 0.99
        self.quantizer = QuantizeEMAReset(512, 256, args)
        self.chamfer_loss = ChamferLoss()
        # self._chamfer_density_loss=Chamfer_Density_Loss()

    def forward(self, batch):
        
        point_encoder_z = self.encoder(batch)

        # Quantization
        x_quantized, commit_loss, perplexity =  self.quantizer(point_encoder_z)
        print(x_quantized.shape)

        # Decoder
        x_decoder = self.decoder(x_quantized)

        # x_out_list.append(x_out)
        # loss_list.append(loss)
        # perplexity_list.append(perplexity)

        # batch["latent_z"] = self.reparameterize(batch)
        # batch.update(self.decoder(batch))

        return x_decoder, commit_loss, perplexity

    def reparameterize(self, batch, seed=None):
        mu, logvar = batch["mu"], batch["logvar"]
        std = torch.exp(logvar / 2)

        if seed is None:
            eps = std.data.new(std.size()).normal_()
        else:
            generator = torch.Generator(device=self.device)
            # 随机数种子
            generator.manual_seed(seed)
            eps = std.data.new(std.size()).normal_(generator=generator)

        z = eps.mul(std).add_(mu)
        return z

    def generate(self, classes, animal_cls, n_frame, n_vertices):
        B = classes.shape[0]  # [B,1]
        xyz = torch.zeros((B, n_frame[0].item(), n_vertices, 3))

        y = classes  # [B,1]
        cls = animal_cls  # [B,1]
        latent_z = torch.randn(B, self.latent_dim, device=classes.device)
        mask = torch.full((B, n_frame[0].item()), True)
        length = n_frame

        batch = {"latent_z": latent_z, "y": y, "xyz": xyz, "cls": cls, "mask": mask, "lengths": length}
        batch.update(self.decoder(batch))
        # batch["output_xyz"]=batch["output"]
        return batch

    def interpolate_latent_generate(self, seq_1, seq_2):

        L, N, channel = seq_1["xyz"][0].shape
        gt_1 = seq_1["xyz"]
        gt_2 = seq_2["xyz"]
        xyz = torch.cat((gt_1, gt_2), dim=0)

        # print(seq_1[1])
        gt_1 = torch.Tensor([seq_1["y"]]).type(torch.long)
        gt_2 = torch.Tensor([seq_2["y"]]).type(torch.long)
        y = torch.cat((gt_1, gt_2), dim=0)

        gt_1 = torch.Tensor([seq_1["cls"]]).type(torch.long)
        gt_2 = torch.Tensor([seq_2["cls"]]).type(torch.long)
        cls = torch.cat((gt_1, gt_2), dim=0)

        gt_1 = seq_1["mask"]
        gt_2 = seq_2["mask"]
        mask = torch.cat((gt_1, gt_2), dim=0)

        # print(cls)

        batch = {"y": y, "xyz": xyz, "cls": cls, "mask": mask}

        batch.update(self.encoder(batch))
        batch["latent_z"] = self.reparameterize(batch)

        a = self.decoder.interpolate_latent_z(batch)

        return batch["output"]

    def compute_kl_loss(self, batch):
        mu, logvar = batch["mu"], batch["logvar"]
        # loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return loss

    def batch_pairwise_dist(self, x, y):
        '''

        :param x:[B,N,C]
        :param y:[B,M,C]
            :return:[B,N,M] square distance between x,y
        '''
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))


        dtype = torch.LongTensor
        diag_ind_x = torch.arange(0, num_points_x).type(dtype)
        diag_ind_y = torch.arange(0, num_points_y).type(dtype)
        # brk()
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P

    def chamfer_density_loss(self, gt_seq, out_seq):
        """
        Calculate Chamfer Distance between two point sets
        :param gt_seq: size[B,L, N, C]
        :param out_seq: size[B,L, M, C]
        :return: sum of Chamfer Distance of two point sets
        """
        B, L, N, C = gt_seq.shape
        _, _, M, _ = gt_seq.shape
        gt_seq = gt_seq.permute(1, 0, 2, 3)  # [L，B，N，C]
        out_seq = out_seq.permute(1, 0, 2, 3)

        gt_seq = gt_seq.reshape(L * B, N, C)
        out_seq = out_seq.reshape(L * B, M, C)
        charmfer_loss, density_loss = self._chamfer_density_loss(gt_seq, out_seq, self.density_k)
        charmfer_loss = charmfer_loss / (B * L * N)
        return charmfer_loss, density_loss

    def chamfer_distance_loss(self, seq1, seq2):
        """
        Calculate Chamfer Distance between two point sets
        :param seq1: size[B,L, N, C]
        :param seq1: size[B,L, M, C]
        :return: sum of Chamfer Distance of two point sets
        """
        B, L, N, C = seq1.shape
        seq1 = seq1.permute(1, 0, 2, 3)  # [L，B，N，C]
        seq2 = seq2.permute(1, 0, 2, 3)

        seq1 = seq1.reshape(L * B, N, C)
        seq2 = seq2.reshape(L * B, N, C)
        loss = self.chamfer_loss(seq1, seq2)
        # loss=loss/(B*L*N)
        return loss

    def compute_density_loss(self, gt_seq, output_seq):
        """
        Calculate density between two point sets seq
        :param gt_seq: size[B,L,N,C]
        :param output_seq: size[B,L,M,C]
        :return: sum of density loss of two point sets seq
        """
        k = self.density_k
        B, L, N, C = gt_seq.shape
        _, _, M, _ = output_seq.shape
        gt_seq = gt_seq.reshape(B * L, N, C)  # [L*B,N,C]
        output_seq = output_seq.reshape(B * L, M, C)  # [L*B,M,C]
        dist1 = self.batch_pairwise_dist(gt_seq, output_seq)  # [L*B,N,M]
        # dist1 = torch.sqrt(dist1)    #[L*B,N,M]

        dist2 = self.batch_pairwise_dist(gt_seq, gt_seq)  # [L*B,N,N]

        val_1, _ = torch.sort(dist1, dim=2, descending=False)
        val_2, _ = torch.sort(dist2, dim=2, descending=False)

        loss = F.mse_loss(val_1, val_2, reduction='mean')

        return loss

    def compute_vertices_loss(self, gt_vertices, output):


        loss = F.mse_loss(gt_vertices.cuda(), output.cuda(), reduction='mean')

        return loss
