from cmath import nan
import math
import sys
import numpy as np
import os

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(BASE_DIR)
# sys.path.append(ROOT_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
# from model.upsample_utils import bilateral_block_l1,bilateral_block_l2,bilateral_block_l3
from utils.chamfer_loss import ChamferLoss, Chamfer_Density_Loss
from utils.furthestpointsample import farthest_point_sample, index_points
from models.vq.quantize import QuantizeEMAReset
from models.vq.encdec import Encoder, Decoder


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

class PointEncoder(nn.Module):
    def __init__(self, args):
        super(PointEncoder, self).__init__()
        self.latent_dim = args.latent_dim
        self.conv1 = torch.nn.Conv1d(args.input_dim, 256, 1)
        self.conv2 = torch.nn.Conv1d(256, 512, 1)
        self.conv3 = torch.nn.Conv1d(512, args.latent_dim, 1)
        self.vq_encoder = Encoder(256, 256, down_t=2, stride_t=2, width=256, depth=3, dilation_growth_rate=3, activation='relu', norm=None)

    def forward(self, x):  # [B,L,N,3]->[B,L,latent_dim]
        xyz = x

        B, L, N, channel = xyz.shape

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

class PEncoder(nn.Module):
    def __init__(self, args, pointencoder):
        super(PEncoder, self).__init__()
        self.latent_dim = args.latent_dim
        self.pointencoder = pointencoder
        # print("encoder::")
        # print(id(self.pointencoder) )

        self.args = args

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, args.dropout)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=args.num_heads,
                                                          dim_feedforward=args.ff_size,
                                                          dropout=args.dropout,
                                                          activation=args.activation)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=args.num_layers)

    def forward(self, x):  # [B,L,N,3]->[B,L,latent_dim]

        B, L, N, channel = x.shape

        z = self.pointencoder(x)  # [B,L,N,3]->[B,L//4,latent_dim]
        xseq = z.permute(2, 0, 1)  # [B,L,latent_dim]->[L,B,latent_dim]

        xseq = self.sequence_pos_encoder(xseq)  # [L,B,latent_dim]

        mask = torch.ones((B, xseq.shape[0]), dtype=torch.bool, device=x.device)

        final = self.seqTransEncoder(xseq, src_key_padding_mask=~mask)

        final = final.permute(1, 2, 0)

        return final


class MappingNet(nn.Module):
    def __init__(self, args):
        super(MappingNet, self).__init__()
        self.K1 = args.latent_dim

        self.fc1 = nn.Linear(self.K1, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, self.K1)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(self.K1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        return x


class Transformer_pointdecoder_base_v2(nn.Module):
    def __init__(self, args):
        super(Transformer_pointdecoder_base_v2, self).__init__()
        self.latent_dim = args.latent_dim
        self.k = 32
        self.num_branch = 1
        self.n = 256 // self.num_branch

        self.fc_Q = nn.Linear(self.latent_dim, self.n * self.k)
        self.fc_K = nn.Linear(self.latent_dim, self.n * self.k)
        self.fc_V = nn.Linear(self.latent_dim, self.n * self.k)
        self.fc_Ori = nn.Linear(self.latent_dim, self.n * self.k)
        self.softmax = nn.Softmax(dim=2)
        self.conv_output = nn.Conv1d(self.k, 3, 1)
        self.conv_ori = nn.Conv1d(self.k, 3, 1)

    def forward(self, x):
        # B,L,N,channel=batch["xyz"].shape
        # x=batch["point_encoder_z"]
        ori = self.fc_Q(x)  # [B,latent_dim]->[B,n*k]  论文的结果
        # ori=self.fc_Ori(x) #[B,latent_dim]->[B,n*k]
        Q = self.fc_Q(x)  # [B,latent_dim]->[B,n*k]
        K = self.fc_K(x)  # [B,latent_dim]->[B,n*k]
        V = self.fc_V(x)  # [B,latent_dim]->[B,n*k]
        ori = ori.view(-1, self.n, self.k)  # [B,n,k]
        Q = Q.view(-1, self.n, self.k)  # [B,n,k]
        K = K.view(-1, self.n, self.k)  # [B,n,k]
        V = V.view(-1, self.n, self.k)  # [B,n,k]

        ori = ori.transpose(1, 2).contiguous()  # [B,K,n]
        ori = self.conv_ori(ori)  # [B,3,n]

        K = K.transpose(1, 2).contiguous()  # [B,K,n]

        # x=self.softmax(torch.div(torch.bmm(Q, K), math.sqrt(self.k)))   #[B,num_points//num_branch,K]
        x = self.softmax(torch.bmm(Q, K) / math.sqrt(self.k))  # [B,n,n]
        x = torch.bmm(x, V)  # [B,n,k]
        x = x.transpose(1, 2).contiguous()  # [B,K,n]
        x = self.conv_output(x)  # [B,3,n]
        x = x + ori
        x = x.transpose(1, 2).contiguous()  # [B,n,3]

        # print("x.shape")
        # print(x.shape)

        return x



class Transformer_pointdecoder(nn.Module):
    def __init__(self, args):
        super(Transformer_pointdecoder, self).__init__()
        self.num_branch = 1
        self.featmap = nn.ModuleList([MappingNet(args) for i in range(self.num_branch)])
        self.mapping_net = True
        self.pointgen = nn.ModuleList([Transformer_pointdecoder_base_v2(args) for i in range(self.num_branch)])
        self.B = args.batch_size

    def forward(self, latent_z):

        output = torch.empty(size=(latent_z.shape[0], 0, 3)).to(latent_z.device)
        for i in range(self.num_branch):
            _x_1 = latent_z
            if hasattr(self, 'mapping_net'):
                if self.mapping_net:
                    _x_1 = self.featmap[i](_x_1)
            # print(_x_1.shape)
            _x_1 = self.pointgen[i](_x_1)
            # print(_x_1.shape)
            output = torch.cat((output, _x_1), dim=1)

        #print("batch['xyz'] shape:", batch["xyz"].shape)

        #print(output.shape)
        #print(f"num_branch: {self.num_branch}")

        #print(f"Expected shape: ({B}, {L}, {N}, {channel}), Actual shape: {output.shape}")
        #print("Before view:", output.shape)  # 确保是 (1200, 1024, 3)
        output = output.view(self.B, -1, output.shape[1], 3)
        #print("After view:", output.shape)  # 确保是 (20, 60, 1024, 3)

        return output

    
class PDecoder(nn.Module):
    def __init__(self, args):
        super(PDecoder, self).__init__()
        self.latent_dim = args.latent_dim

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, args.dropout)

        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                          nhead=args.num_heads,
                                                          dim_feedforward=args.ff_size,
                                                          dropout=args.dropout,
                                                          activation=args.activation)
        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                     num_layers=args.num_layers)

        self.points_decoder = Transformer_pointdecoder(args)

        self.decoder = Decoder(256, 256, down_t=2, stride_t=2, width=256, depth=3, dilation_growth_rate=3, activation='relu', norm=None)
        
        self.point_out_num = args.points_num

    def forward(self, x): 
        B, latent_dim, L = x.shape
        mask = torch.ones((B, L), dtype=torch.bool, device=x.device)
        # print("latent_z.shape:")
        # print(latent_z.shape)
        # print(self.actionBiases[y].shape)
        # print(self.clsBiases[cls].shape)

        # latent_z = latent_z.unsqueeze(0)

        # timequeries = torch.zeros(L, B, latent_dim, device=latent_z.device)

        timequeries = torch.zeros(L, B, latent_dim, device=x.device)
        timequeries = self.sequence_pos_encoder(timequeries)

        x = x.mean(dim=2, keepdim=True)
        x = x.contiguous().view(1, B, latent_dim)

        trans_output = self.seqTransDecoder(tgt=timequeries, memory=x, tgt_key_padding_mask=~mask)

        # timequeries表示positional encoding的输出

        trans_output = trans_output.permute(1, 2, 0)  # [B,L,latent_dim]

        trans_output = self.decoder(trans_output) # [B,latent_dim, L] -> [B,4*L,latent_dim]

        trans_output = trans_output.reshape(-1, self.latent_dim)  # [B,L,latent_dim]->#[B*L,latent_dim]

        output = self.points_decoder(trans_output)

        N = self.point_out_num

        mask = torch.ones((B, L*4), dtype=torch.bool, device=x.device)
        # batch["output"]=batch["output"].reshape(B,L,N,channel)
        # batch["output"]=batch["output"].reshape(L,B,N,channel)
        output = output.permute(1, 0, 2, 3)
        output[~mask.T] = 0
        # print(lengths)
        # print(batch["output"])
        output = output.permute(1, 0, 2, 3)

        return output

    def interpolate_latent_z(self, batch):

        latent_dim = batch["latent_z"].shape[1]
        latent_z, y, cls = batch["latent_z"], batch["y"], batch["cls"]

        alphs = torch.linspace(0.0, 1.0, 3)

        print("latent_z.shape:")
        print(alphs)

        latent_z = latent_z + self.actionBiases[y] + self.clsBiases[cls]

        temp_latent_zs = torch.zeros((len(alphs), latent_dim)).cuda()
        for i in range(0, len(alphs)):
            alph = alphs[i]
            temp_latent_zs[i] = (latent_z[0] * (1 - alph) + latent_z[1] * (alph))

        latent_z = temp_latent_zs
        print(temp_latent_zs.shape)
        B, _ = latent_z.shape
        _, L, n, channel = batch["xyz"].shape
        batch["xyz"] = torch.zeros((B, L, n, 3))

        latent_z = temp_latent_zs[None]  # sequence of size 1

        print(latent_z.shape)

        timequeries = torch.zeros(L, B, latent_dim, device=latent_z.device)
        timequeries = self.sequence_pos_encoder(timequeries)
        trans_output = self.seqTransDecoder(tgt=timequeries, memory=latent_z)

        trans_output = trans_output.permute(1, 0, 2)  # [B,L,latent_dim]

        trans_output = trans_output.reshape(B * L, self.latent_dim)  # [B,L,latent_dim]->#[B*L,latent_dim]

        # print(trans_output.shape)
        batch["trans_output"] = trans_output

        batch.update(self.points_decoder(batch))
        N = self.point_out_num
        batch["output"] = batch["output"].reshape(B, L, N, channel)

        return batch["output"]


class PCMGVQ(nn.Module):
    def __init__(self, args):
        super(PCMGVQ, self).__init__()
        self.density_k = args.density_k
        self.pointencoder = PointEncoder(args)

        self.latent_dim = args.latent_dim

        self.encoder = PEncoder(args, self.pointencoder)
        self.decoder = PDecoder(args)

        self.quantizer = QuantizeEMAReset(512, 256, args)

        self.chamfer_loss = ChamferLoss()

    def forward(self, x):
        
        x_encoder = self.encoder(x) # [B, L, n, 3] -> [B, L, dim]

        x_quantized, loss, perplexity = self.quantizer(x_encoder)

        x_decoder = self.decoder(x_quantized)

        return x_decoder, loss, perplexity
    
    def reparameterize(self, batch, seed=None):
        mu, logvar = batch["mu"], batch["logvar"]
        std = torch.exp(logvar / 2)

        if seed is None:
            eps = std.data.new(std.size()).normal_()
        else:
            generator = torch.Generator(device=self.device)

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
