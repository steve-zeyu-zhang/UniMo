import torch
import torch.nn.functional as F
from utils.chamfer_loss import ChamferLoss, Chamfer_Density_Loss
from utils.emd.emd import earth_mover_distance

class PcmgLoss:
        def __init__(self, density_k):

            self.density_k = density_k
            self.chamfer_loss = ChamferLoss()
            self._chamfer_density_loss=Chamfer_Density_Loss()

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
        
        def compute_emd_loss(self, gt_seq, pred_seq):
            """
            Calculate Density-aware Chamfer Distance between two point sets
            :param gt_seq: size[B,L, N, C]
            :param pred_seq: size[B,L, M, C]
            :return: sum of Density-aware Chamfer Distance of two point sets
            """
            B,L,N,C=gt_seq.shape
            _,_,M,_=pred_seq.shape
            gt_seq=gt_seq.permute(1,0,2,3)      #[L，B，N，C]
            pred_seq=pred_seq.permute(1,0,2,3)  #[L，B，M，C]
            
            gt_seq=gt_seq.reshape(L*B,N,C)
            pred_seq=pred_seq.reshape(L*B,M,C)
            emd_loss= earth_mover_distance(pred_seq,gt_seq, transpose=False)
            #emd_loss=emd_loss/(L*B*N)
            emd_loss=torch.mean(emd_loss)
            #print(emd_loss)
            
            return emd_loss