import math
import torch
import torch.nn as nn

class KNNLoss(nn.Module):
    def __init__(self, n_bgroups, k):
        super(KNNLoss, self).__init__()
        self.n_bgroups = n_bgroups
        self.k = k

    def forward(self, expected, actual, m_lens, output_indices=False):
        """
        参数：
          expected: [batch, frames, points, 3 + bgroups]，motion数据（其中前三个通道为坐标）
          actual: 同 expected
          m_lens: 长度为 batch 的列表，每个元素为该 sample 的真实帧数（即有效帧数）
          output_indices: 是否输出索引（这里暂不支持输出索引，返回 None）
        """
        batches = expected.shape[0]
        
        # 分别存放各个 sample 的 loss
        loss_glob_p_list = []
        loss_norm_p_list = []
        loss_v_list = []
        
        # 针对每个 sample 分别处理（只使用有效帧）
        for i in range(batches):
            valid_length = m_lens[i]  # 当前 sample 的真实帧数
            # 只取有效帧（假定 m_lens 中数值不超过 expected.shape[1]）
            exp_i = expected[i, :valid_length, :, :]  # [valid_length, points, 3 + bgroups]
            act_i = actual[i, :valid_length, :, :]

            # 提取坐标信息，得到形状 [valid_length, points, 3]
            p_e = exp_i[..., :3].clone()
            p_a = act_i[..., :3].clone()
            
            # 计算速度（时序差分），第一帧速度设为0
            v_e = p_e.clone()
            v_a = p_a.clone()
            if valid_length > 1:
                v_e[1:] = v_e[1:] - v_e[:-1]
                v_a[1:] = v_a[1:] - v_a[:-1]
            v_e[0] = 0
            v_a[0] = 0
            
            # 根据第一帧的 bgroups 得到每个点所属组，形状为 [points]
            bgroups_e = torch.argmax(exp_i[0, :, 3:].contiguous(), dim=-1)
            bgroups_a = torch.argmax(act_i[0, :, 3:].contiguous(), dim=-1)
            
            # 存放当前 sample 各个组的距离（每组可能有不同数量的点）
            dists_glob_list = []
            dists_norm_list = []
            dists_v_list = []
            
            # 针对每个背景组分别处理
            for g in range(self.n_bgroups):
                # 选择当前组的点，结果形状 [valid_length, num_points_g, 3]
                p_e_g = p_e[:, bgroups_e == g]
                p_a_g = p_a[:, bgroups_a == g]
                v_e_g = v_e[:, bgroups_e == g]
                v_a_g = v_a[:, bgroups_a == g]
                
                # 如果该组没有点，则跳过
                if p_e_g.shape[1] == 0:
                    continue

                # 对位置做归一化（在所有帧和点上计算均值和标准差）
                norm_p_e = p_e_g - torch.mean(p_e_g, dim=(0,1), keepdim=True)
                norm_std_e = torch.std(p_e_g, dim=(0,1), keepdim=True) + 1e-6
                norm_p_e = norm_p_e / norm_std_e
                norm_p_a = p_a_g - torch.mean(p_a_g, dim=(0,1), keepdim=True)
                norm_std_a = torch.std(p_a_g, dim=(0,1), keepdim=True) + 1e-6
                norm_p_a = norm_p_a / norm_std_a

                # 将时序信息与点维度合并，即将 [valid_length, num_points_g, 3] 转为 [num_points_g, valid_length * 3]
                p_e_flat = p_e_g.transpose(0, 1).reshape(p_e_g.shape[1], valid_length * 3)
                p_a_flat = p_a_g.transpose(0, 1).reshape(p_a_g.shape[1], valid_length * 3)
                norm_p_e_flat = norm_p_e.transpose(0, 1).reshape(norm_p_e.shape[1], valid_length * 3)
                norm_p_a_flat = norm_p_a.transpose(0, 1).reshape(norm_p_a.shape[1], valid_length * 3)
                v_e_flat = v_e_g.transpose(0, 1).reshape(v_e_g.shape[1], valid_length * 3)
                v_a_flat = v_a_g.transpose(0, 1).reshape(v_a_g.shape[1], valid_length * 3)

                # 计算欧氏距离
                dists_glob = torch.cdist(p_e_flat, p_a_flat, p=2)
                dists_norm = torch.cdist(norm_p_e_flat, norm_p_a_flat, p=2)
                dists_v = torch.cdist(v_e_flat, v_a_flat, p=2)

                # 对距离进行排序，并取前 k 个距离
                dists_glob, indices = torch.sort(dists_glob, dim=-1)
                dists_norm, _ = torch.sort(dists_norm, dim=-1)
                dists_v = torch.gather(dists_v, -1, indices)
                dists_glob_list.append(dists_glob[..., :self.k])
                dists_norm_list.append(dists_norm[..., :self.k])
                dists_v_list.append(dists_v[..., :self.k])
            
            # 若有至少一个组有数据，则对当前 sample 的 loss 作平均（同时归一化因子使用有效帧数）
            if len(dists_glob_list) > 0:
                loss_glob = torch.mean(torch.cat(dists_glob_list, dim=0)) / math.sqrt(valid_length)
                loss_norm = torch.mean(torch.cat(dists_norm_list, dim=0)) / math.sqrt(valid_length)
                loss_v_sample = torch.mean(torch.cat(dists_v_list, dim=0)) / math.sqrt(valid_length)
            else:
                loss_glob = torch.tensor(0.0, device=expected.device)
                loss_norm = torch.tensor(0.0, device=expected.device)
                loss_v_sample = torch.tensor(0.0, device=expected.device)
            
            loss_glob_p_list.append(loss_glob)
            loss_norm_p_list.append(loss_norm)
            loss_v_list.append(loss_v_sample)
        
        # 最终对所有 sample 的 loss 取平均
        loss_glob_p = torch.stack(loss_glob_p_list).mean()
        loss_norm_p = torch.stack(loss_norm_p_list).mean()
        loss_v = torch.stack(loss_v_list).mean()
        
        if output_indices:
            return loss_glob_p, loss_norm_p, loss_v, None
        else:
            return loss_glob_p, loss_norm_p, loss_v