import json
import os
import torch
from torch import nn
from models.pc_mrl import PointCloudDecoder
from models.utils.knn_loss import KNNLoss
from data.lafan import LaFANDataLoader
import sys
import numpy as np
from viewport.visualiser import Visualiser


class PointCloudEval:
    def __init__(self, dataloaders, tgt_skeletons, ckpt_path, cfg, device='cpu'):
        """
        初始化评估器
        :param dataloaders: 数据加载器列表
        :param tgt_skeletons: 目标骨架列表
        :param ckpt_path: 模型检查点路径
        :param cfg: 配置文件
        :param device: 计算设备（CPU 或 GPU）
        """
        self.cfg = cfg
        self.device = device
        self.dataloaders = dataloaders
        self.tgt_skeletons = tgt_skeletons
        self.ckpt_path = ckpt_path
        self.ep = int(''.join(filter(str.isdigit, os.path.basename(ckpt_path))))

        # 创建保存目录
        self.pcd_save_dir = "/root/autodl-tmp/PCMRL/pcd"
        self.bvh_save_dir = "/root/autodl-tmp/PCMRL/bvh"
        os.makedirs(self.pcd_save_dir, exist_ok=True)
        os.makedirs(self.bvh_save_dir, exist_ok=True)

        # 为每个骨骼计算参数
        self.valid_joints_list = []
        self.d_pose_outputs = []
        for skeleton in tgt_skeletons:
            d_pose_output = 3
            valid_joints = []
            for i in range(len(skeleton.joint_lengths)):
                if skeleton.joint_lengths[i] > 1e-6:
                    d_pose_output += 4
                    valid_joints.append(i)
            self.valid_joints_list.append(torch.tensor(valid_joints, dtype=torch.int64, device=device))
            self.d_pose_outputs.append(d_pose_output)

        # 最大输出维度
        max_d_pose = max(self.d_pose_outputs)
        self.model = PointCloudDecoder(
            self.cfg["PC-MRL"]["n_bgroups"],
            self.cfg["PC-MRL"]["d_model"],
            max_d_pose,
            self.cfg["PC-MRL"]["temporal_n_heads"],
            self.cfg["PC-MRL"]["temporal_n_layers"],
            self.cfg["PC-MRL"]["k_farthest_first"],
            self.cfg["GENERAL"]["max_length"]
        ).to(self.device)

        self.model.load_state_dict(torch.load(ckpt_path))
        self.model.eval()

        # KNN损失
        self.knn_loss = KNNLoss(self.cfg["PC-MRL"]["n_bgroups"], self.cfg["PC-MRL"]["knn_loss_k"])

    def evaluate(self, n_tests):
        """
        评估模型并保存结果
        :param n_tests: 测试样本数量
        """
        total_loss = 0.0
        knn_loss_g_total = 0.0
        knn_loss_l_total = 0.0
        knn_loss_v_total = 0.0
        end_loss_total = 0.0
        unit_loss_total = 0.0

        for loader_idx, dataloader in enumerate(self.dataloaders):
            with torch.no_grad():
                # 加载数据
                root_p, q = dataloader.load_samples(n_tests, self.cfg["GENERAL"]["max_length"])
                global_p, global_q = dataloader.skeleton.fk(root_p.to(self.device), q.to(self.device), local_q=True)
                samples = dataloader.skeleton.generate_pointcloud(global_p, global_q,
                                                                  n_points=self.cfg["PC-MRL"]["src_n_points"],
                                                                  std=self.cfg["PC-MRL"]["point_sampling_std"])
                end_joints = global_p.transpose(0, 2).clone()[dataloader.skeleton.end_joints].transpose(0, 2)

                # 数据归一化
                means = torch.mean(samples[..., :3], dim=(1, 2), keepdim=True)
                samples[..., :3] -= means
                end_joints -= means
                real_global_p = global_p - means
                real_global_q = global_q

                # 前向传播
                pred = self.model(samples)

                skeleton_idx = loader_idx
                valid_joints = self.valid_joints_list[skeleton_idx]
                d_pose_output = self.d_pose_outputs[skeleton_idx]
                tgt_skeleton = self.tgt_skeletons[skeleton_idx]

                pred_p = pred[..., :3]
                pred_q_part = pred[..., 3:d_pose_output]
                pred_q = torch.reshape(pred_q_part, (pred.shape[0], pred.shape[1], -1, 4))

                pred_q_full = torch.zeros((pred.shape[0], pred.shape[1], tgt_skeleton.n_joints, 4),
                                          dtype=torch.float32, device=self.device)
                pred_q_full[..., 0] = 1
                indices = torch.reshape(valid_joints, (1, 1, -1, 1)).repeat(pred_q.shape[0], pred_q.shape[1], 1, 4)
                pred_q_full = torch.scatter(pred_q_full, 2, indices, pred_q)

                pred_global_p, pred_global_q = tgt_skeleton.fk(pred_p, pred_q_full, local_q=False)

                batch_size, num_frames = pred_global_p.shape[:2]
                for batch_idx in range(batch_size):
                    batch_dir = os.path.join(self.bvh_save_dir, f"batch_{batch_idx:03d}")
                    os.makedirs(batch_dir, exist_ok=True)

                    # 获取根关节位置和旋转数据
                    root_pos = pred_global_p[batch_idx, :, 0, :]  # [frames, 3]
                    batch_global_q = pred_global_q[batch_idx]  # [frames, joints, 4]

                    # 保存 BVH
                    bvh_filename = f"motion_ep{self.ep}_batch{batch_idx:03d}.bvh"
                    bvh_filepath = os.path.join(batch_dir, bvh_filename)

                    tgt_skeleton.save_bvh(
                        bvh_filepath,
                        root_pos.detach().cpu(),
                        batch_global_q.detach().cpu(),
                        frame_time=1 / 60.0
                    )
                    print(f"Saved BVH file: {bvh_filepath}")


                # 生成目标点云并恢复全局坐标
                pred_pointcloud = tgt_skeleton.generate_pointcloud(pred_global_p, pred_global_q,
                                                                   n_points=self.cfg["PC-MRL"]["tgt_n_points"],
                                                                   std=self.cfg["PC-MRL"]["point_sampling_std"])
                pred_pointcloud[..., :3] += means  # 恢复全局坐标

                # 保存点云文件
                for batch_idx in range(batch_size):
                    batch_dir = os.path.join(self.pcd_save_dir, f"batch_{batch_idx:03d}")
                    os.makedirs(batch_dir, exist_ok=True)

                    for frame_idx in range(num_frames):
                        frame_pc = pred_pointcloud[batch_idx, frame_idx, ..., :3].cpu().numpy()
                        frame_pc = frame_pc[:, [0, 2, 1]]  # 交换 Y 和 Z 轴
                        frame_pc[:, 2] *= -1  # 反转 Z 轴方向

                        filename = f"frame_{frame_idx:04d}.pcd"
                        filepath = os.path.join(batch_dir, filename)
                        tgt_skeleton.save_as_pcd(frame_pc, filepath)

                # 计算损失
                knn_loss_g, knn_loss_l, knn_loss_v = self.knn_loss(samples, pred_pointcloud)
                pred_end_joints = pred_global_p.transpose(0, 2)[tgt_skeleton.end_joints].transpose(0, 2)
                end_loss = torch.mean(torch.abs(end_joints - pred_end_joints))
                unit_loss = torch.mean(torch.abs(1 - torch.linalg.vector_norm(pred_q, dim=-1)))

                # 累积损失
                loss = knn_loss_g + knn_loss_l + knn_loss_v + end_loss + unit_loss
                total_loss += loss.item()
                knn_loss_g_total += knn_loss_g.item()
                knn_loss_l_total += knn_loss_l.item()
                knn_loss_v_total += knn_loss_v.item()
                end_loss_total += end_loss.item()
                unit_loss_total += unit_loss.item()

        # 打印评估结果
        avg_loss = total_loss / len(self.dataloaders)
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"  knn_g: {knn_loss_g_total / len(self.dataloaders):.4f} "
              f"knn_l: {knn_loss_l_total / len(self.dataloaders):.4f} "
              f"knn_v: {knn_loss_v_total / len(self.dataloaders):.4f} "
              f"end: {end_loss_total / len(self.dataloaders):.4f} "
              f"unit: {unit_loss_total / len(self.dataloaders):.4f}")

        with open("./ckpt_pc-mrl/evaluation_log.txt", 'a') as log:
            log.write(f"Average Loss: {avg_loss:.4f}\n")
            log.write(f"  knn_g: {knn_loss_g_total / len(self.dataloaders):.4f} "
                      f"knn_l: {knn_loss_l_total / len(self.dataloaders):.4f} "
                      f"knn_v: {knn_loss_v_total / len(self.dataloaders):.4f} "
                      f"end: {end_loss_total / len(self.dataloaders):.4f} "
                      f"unit: {unit_loss_total / len(self.dataloaders):.4f}\n")


if __name__ == "__main__":
    torch.set_printoptions(sci_mode=False)

    cfg_file = sys.argv[2] if len(sys.argv) > 1 else './config.json'
    with open(cfg_file, 'r') as f:
        cfg = json.load(f)

    device = torch.device("cuda")

    # 初始化数据加载器和目标骨骼
    lafan_loader = LaFANDataLoader(
        cfg["GENERAL"]["lafan1_path_val"],
        cfg["GENERAL"]["lafan1_skeleton_val"],
        min_length=128,
        data_device=device,
        skel_device=device
    )

    lafan_skeleton = LaFANDataLoader.setup_skeleton(cfg["GENERAL"]["lafan1_skeleton_val"], device)

    evaluator = PointCloudEval(
        dataloaders=[lafan_loader],
        tgt_skeletons=[lafan_skeleton],
        ckpt_path="./ckpt_pc-mrl/PCMRL50000.pt",
        cfg=cfg,
        device=device
    )
    evaluator.evaluate(n_tests=64)