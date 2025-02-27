import json
import os
import torch
from torch import nn
from models.pc_mrl import PointCloudDecoder
from models.utils.knn_loss import KNNLoss
from data.lafan import LaFANDataLoader
import sys
from data.cmu import CMUDataLoader
import numpy as np
from viewport.visualiser import Visualiser


class PointCloudEval:
    def __init__(self, dataloaders, tgt_skeletons, skeleton_names, ckpt_path, cfg, device='cpu'):
        """
        初始化评估器（支持多骨骼）
        :param dataloaders: 数据加载器列表（每个对应一个骨骼）
        :param tgt_skeletons: 目标骨架列表（与dataloaders顺序一致）
        :param skeleton_names: 骨骼名称列表（如['cmu', 'lafan']）
        :param ckpt_path: 模型检查点路径
        :param cfg: 配置文件
        :param device: 计算设备（CPU 或 GPU）
        """
        self.cfg = cfg
        self.device = device
        self.dataloaders = dataloaders
        self.tgt_skeletons = tgt_skeletons
        self.skeleton_names = skeleton_names  # 新增骨骼名称列表
        self.ckpt_path = ckpt_path
        self.ep = int(''.join(filter(str.isdigit, os.path.basename(ckpt_path))))

        # 修改基础目录结构
        self.base_dir = "/root/autodl-tmp/PCMRL/output"
        self.epoch_dir = os.path.join(self.base_dir, f"epoch_{self.ep}")
        os.makedirs(self.epoch_dir, exist_ok=True)  # 只创建epoch目录，子目录按需创建

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

        # 模型初始化
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
        多骨骼评估（自动创建对应目录结构）
        :param n_tests: 每个骨骼的测试样本数量
        """
        total_loss = 0.0
        knn_loss_g_total = 0.0
        knn_loss_l_total = 0.0
        knn_loss_v_total = 0.0
        end_loss_total = 0.0
        unit_loss_total = 0.0

        for loader_idx, dataloader in enumerate(self.dataloaders):
            skeleton_name = self.skeleton_names[loader_idx]  # 获取当前骨骼名称
            tgt_skeleton = self.tgt_skeletons[loader_idx]  # 当前目标骨骼

            # 创建当前骨骼的目录结构
            skeleton_dir = os.path.join(self.epoch_dir, skeleton_name)
            bvh_predict_dir = os.path.join(skeleton_dir, "predict")
            bvh_gt_dir = os.path.join(skeleton_dir, "GT")
            pcd_dir = os.path.join(skeleton_dir, "pcd")
            os.makedirs(bvh_predict_dir, exist_ok=True)
            os.makedirs(bvh_gt_dir, exist_ok=True)
            os.makedirs(pcd_dir, exist_ok=True)

            with torch.no_grad():
                # 数据加载与预处理
                root_p, q = dataloader.load_samples(n_tests, self.cfg["GENERAL"]["max_length"])
                global_p, global_q = dataloader.skeleton.fk(root_p.to(self.device), q.to(self.device), local_q=True)
                samples = dataloader.skeleton.generate_pointcloud(
                    global_p, global_q,
                    n_points=self.cfg["PC-MRL"]["src_n_points"],
                    std=self.cfg["PC-MRL"]["point_sampling_std"]
                )
                end_joints = global_p.transpose(0, 2).clone()[dataloader.skeleton.end_joints].transpose(0, 2)

                # 数据归一化
                means = torch.mean(samples[..., :3], dim=(1, 2), keepdim=True)
                samples[..., :3] -= means
                end_joints -= means
                real_global_p = global_p - means
                real_global_q = global_q

                # 前向推理
                pred = self.model(samples)

                # 获取当前骨骼参数
                valid_joints = self.valid_joints_list[loader_idx]
                d_pose_output = self.d_pose_outputs[loader_idx]

                # 解析预测结果
                pred_p = pred[..., :3]
                pred_q_part = pred[..., 3:d_pose_output]
                pred_q = torch.reshape(pred_q_part, (pred.shape[0], pred.shape[1], -1, 4))

                # 构建完整四元数
                pred_q_full = torch.zeros((pred.shape[0], pred.shape[1], tgt_skeleton.n_joints, 4),
                                          dtype=torch.float32, device=self.device)
                pred_q_full[..., 0] = 1
                indices = torch.reshape(valid_joints, (1, 1, -1, 1)).repeat(pred_q.shape[0], pred_q.shape[1], 1, 4)
                pred_q_full = torch.scatter(pred_q_full, 2, indices, pred_q)

                # 前向运动学
                pred_global_p, pred_global_q = tgt_skeleton.fk(pred_p, pred_q_full, local_q=False)
                pred_global_p_adjusted = pred_global_p + means.view(pred_global_p.shape[0], 1, 1, 3)  # 恢复全局坐标

                # 保存BVH文件
                batch_size, num_frames = pred_global_p.shape[:2]
                for batch_idx in range(batch_size):
                    # 保存预测结果
                    pred_root_pos = pred_global_p_adjusted[batch_idx, :, 0, :]
                    pred_bvh_path = os.path.join(bvh_predict_dir, f"bvh_predict_{batch_idx:03d}.bvh")
                    tgt_skeleton.save_bvh(pred_bvh_path,
                                          pred_root_pos.cpu().detach(),
                                          pred_global_q[batch_idx].cpu().detach(),
                                          frame_time=1 / 30.0)

                    # 保存Ground Truth
                    gt_root_pos = global_p[batch_idx, :, 0, :].cpu()
                    gt_bvh_path = os.path.join(bvh_gt_dir, f"bvh_GT_{batch_idx:03d}.bvh")
                    dataloader.skeleton.save_bvh(gt_bvh_path,
                                                 gt_root_pos,
                                                 global_q[batch_idx].cpu(),
                                                 frame_time=1 / 30.0)

                # 保存点云数据
                pred_pointcloud = tgt_skeleton.generate_pointcloud(
                    pred_global_p_adjusted, pred_global_q,
                    n_points=self.cfg["PC-MRL"]["tgt_n_points"],
                    std=self.cfg["PC-MRL"]["point_sampling_std"]
                )
                for batch_idx in range(batch_size):
                    batch_pcd_dir = os.path.join(pcd_dir, f"pcd_{batch_idx:03d}")
                    os.makedirs(batch_pcd_dir, exist_ok=True)

                    for frame_idx in range(num_frames):
                        frame_pc = pred_pointcloud[batch_idx, frame_idx, ..., :3].cpu().numpy()
                        frame_pc = frame_pc[:, [0, 2, 1]]  # 坐标系调整
                        frame_pc[:, 2] *= -1

                        filename = f"frame_{frame_idx:04d}.pcd"
                        filepath = os.path.join(batch_pcd_dir, filename)
                        tgt_skeleton.save_as_pcd(frame_pc, filepath)

                # 损失计算
                knn_loss_g, knn_loss_l, knn_loss_v = self.knn_loss(samples, pred_pointcloud)
                pred_end_joints = pred_global_p.transpose(0, 2)[tgt_skeleton.end_joints].transpose(0, 2)
                end_loss = torch.mean(torch.abs(end_joints - pred_end_joints))
                unit_loss = torch.mean(torch.abs(1 - torch.linalg.vector_norm(pred_q, dim=-1)))

                # 累计损失
                loss = knn_loss_g + knn_loss_l + knn_loss_v + end_loss + unit_loss
                total_loss += loss.item()
                knn_loss_g_total += knn_loss_g.item()
                knn_loss_l_total += knn_loss_l.item()
                knn_loss_v_total += knn_loss_v.item()
                end_loss_total += end_loss.item()
                unit_loss_total += unit_loss.item()

        # 输出评估结果
        avg_loss = total_loss / len(self.dataloaders)
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"  knn_g: {knn_loss_g_total / len(self.dataloaders):.4f} "
              f"knn_l: {knn_loss_l_total / len(self.dataloaders):.4f} "
              f"knn_v: {knn_loss_v_total / len(self.dataloaders):.4f} "
              f"end: {end_loss_total / len(self.dataloaders):.4f} "
              f"unit: {unit_loss_total / len(self.dataloaders):.4f}")

        # 保存日志文件
        log_path = os.path.join(self.epoch_dir, "evaluation_log.txt")
        with open(log_path, 'a') as f:
            f.write(f"Average Loss: {avg_loss:.4f}\n")
            f.write(f"  knn_g: {knn_loss_g_total / len(self.dataloaders):.4f} "
                    f"knn_l: {knn_loss_l_total / len(self.dataloaders):.4f} "
                    f"knn_v: {knn_loss_v_total / len(self.dataloaders):.4f} "
                    f"end: {end_loss_total / len(self.dataloaders):.4f} "
                    f"unit: {unit_loss_total / len(self.dataloaders):.4f}\n")


if __name__ == "__main__":
    torch.set_printoptions(sci_mode=False)

    # 配置加载
    cfg_file = sys.argv[2] if len(sys.argv) > 1 else './config.json'
    with open(cfg_file, 'r') as f:
        cfg = json.load(f)

    device = torch.device("cuda")

    # 初始化两个骨骼的数据加载器（示例）
    # LaFAN数据
    lafan_loader = LaFANDataLoader(
        cfg["GENERAL"]["lafan1_path_val"],
        cfg["GENERAL"]["lafan1_skeleton_val"],
        min_length=128,
        data_device=device,
        skel_device=device
    )
    lafan_skeleton = LaFANDataLoader.setup_skeleton(cfg["GENERAL"]["lafan1_skeleton_val"], device)

    cmu_loader = CMUDataLoader(
        cfg["GENERAL"]["cmu_path_train"],
        cfg["GENERAL"]["cmu_skeleton_train"],
        min_length=128,
        data_device=device,
        skel_device=device
    )
    cmu_skeleton = CMUDataLoader.setup_skeleton(cfg["GENERAL"]["cmu_skeleton_train"], device)

    # 创建评估器（传入两个骨骼）
    evaluator = PointCloudEval(
        dataloaders=[lafan_loader],  # 添加cmu_loader到列表
        tgt_skeletons=[lafan_skeleton],  # 添加cmu_skeleton到列表
        skeleton_names=['lafan','cmu'],  # 改为['cmu', 'lafan']
        ckpt_path="./ckpt_pc-mrl/PCMRL50000.pt",
        cfg=cfg,
        device=device
    )
    evaluator.evaluate(n_tests=5)