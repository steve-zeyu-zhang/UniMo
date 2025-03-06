import json
from turtle import pd


from torch import nn
from data.cmu import CMUDataLoader
from models_pcmrl.pc_mrl import PointCloudDecoder
from models_pcmrl.utils.knn_loss import KNNLoss
import datetime
import math
from operator import mod
import os
from statistics import mode
import sys
from _dataset.src.dataset import get_dataset_collate
from tqdm import *
import torch
from evaluate.action2motion.models import load_classifier as load_gru_classifier
from models_pcmg.PCMG import PCMG

from utils.furthestpointsample import farthest_point_sample, index_points
from evaluate.action2motion.accuracy import calculate_accuracy
from utils.emd_loss import compute_emd_loss
from visuals.log import log_out_message
from visuals.pcd_video  import visualize_point_cloud_offscreen, load_pcd_files
from visuals.ploy_script import plot_3d_motion
from Motion import BVH
from Motion.AnimationStructure import get_kinematic_chain
from Motion.Animation import positions_global as anim_pos
from _parser.parser import get_parser
from argparse import ArgumentParser


class PointCloudTrainer:
    def __init__(self, dataloaders, tgt_skeletons, cfg, device='cpu'):
        self.cfg = cfg
        self.device = device
        self.dataloaders = dataloaders
        self.tgt_skeletons = tgt_skeletons  # Changed to support multiple target skeletons

        # Calculate parameters for each skeleton
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

        # Add mapping from skeleton type to model index
        self.skeleton_mapping = {
            type(dl.skeleton).__name__: idx
            for idx, dl in enumerate(dataloaders)
        }

        self.pcmg_args = self._init_pcmg_args(cfg)

        self.vq_pc = PCMG(self.pcmg_args).to(self.device)

        self.current_means = None

    def _init_pcmg_args(self, cfg):
        """Reuse the argument object from the main function and override configuration"""
        # Get the args object from the main function
        _, args = get_parser()

        # Override key parameters from the configuration file
        args.points_num = cfg["PC-MRL"]["src_n_points"]
        args.batch_size = cfg["PC-MRL"]["batch_size"]
        args.num_frames = cfg["GENERAL"]["max_length"]

        return args

    def save(self, ep):
        torch.save(self.model.state_dict(), "./ckpt_pc-mrl/PCMRL%d.pt" % ep)

    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr

        return current_lr
    
    def save_visuals(self):
        return
        # Determine the skeleton type (e.g.,: cmu or lafan)
        skeleton_name = "cmu" if loader_idx == 0 else "lafan"

        # Create a pcd directory under the current epoch
        epoch_dir = os.path.join(self.output_dir, f"epoch_{ep:03d}")

        encode_pcd_root_dir = os.path.join(epoch_dir, "pcd", "encode", skeleton_name)
        predict_pcd_root_dir = os.path.join(epoch_dir, "pcd", "predict", skeleton_name)
        os.makedirs(encode_pcd_root_dir, exist_ok=True)
        os.makedirs(predict_pcd_root_dir, exist_ok=True)

        # Save encode point cloud (PCD): save at most 5 samples
        encode_global_pc = samples.clone()
        encode_global_pc[..., :3] += self.current_means
        encode_pc_numpy = encode_global_pc.cpu().numpy()
        # Swap Y/Z axes and invert the Z axis
        # encode_pc_numpy = encode_pc_numpy[..., [0, 2, 1]]
        # encode_pc_numpy[..., 2] *= -1

        for batch_idx in range(min(5, batch_size)):
            encoder_pcd_dir = os.path.join(encode_pcd_root_dir, f"encode_{batch_idx:03d}")
            os.makedirs(encoder_pcd_dir, exist_ok=True)
            for frame_idx in range(max_length):
                frame_pc = encode_pc_numpy[batch_idx, frame_idx]
                filename = f"frame_{frame_idx:04d}.pcd"
                filepath = os.path.join(encoder_pcd_dir, filename)
                tgt_skeleton.save_as_pcd(frame_pc, filepath)

        # Save predict point cloud (PCD): save at most 5 samples
        pred_global_pc = pred_pointcloud.clone()
        pred_global_pc[..., :3] += self.current_means
        pred_pc_numpy = pred_global_pc.cpu().numpy()
        # Swap Y/Z axes and invert the Z axis
        # pred_pc_numpy = pred_pc_numpy[..., [0, 2, 1]]
        # pred_pc_numpy[..., 2] *= -1

        for batch_idx in range(min(5, batch_size)):
            predict_pcd_dir = os.path.join(predict_pcd_root_dir, f"predict_{batch_idx:03d}")
            os.makedirs(predict_pcd_dir, exist_ok=True)
            for frame_idx in range(max_length):
                frame_pc = pred_pc_numpy[batch_idx, frame_idx]
                filename = f"frame_{frame_idx:04d}.pcd"
                filepath = os.path.join(predict_pcd_dir, filename)
                tgt_skeleton.save_as_pcd(frame_pc, filepath)

        # Create directories for saving BVH files: save both GT and predict
        gt_dir = os.path.join(epoch_dir, "bvh", "GT", skeleton_name)
        predict_dir = os.path.join(epoch_dir, "bvh", "predict", skeleton_name)
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(predict_dir, exist_ok=True)

        # Save BVH files for at most 5 samples
        for batch_idx in range(min(5, batch_size)):

            # Save GT BVH file using the true global joint data
            root_pos = global_p[batch_idx, :, 0, :]  # [frames, 3]
            batch_global_q = global_q[batch_idx]  # [frames, joints, 4]
            gt_filepath = os.path.join(gt_dir, f"gt_{batch_idx:03d}.bvh")
            tgt_skeleton.save_bvh(
                gt_filepath,
                root_pos.detach().cpu(),
                batch_global_q.detach().cpu(),
                frame_time=1 / 30.0
            )

            # Save the predicted BVH file
            root_pos = pred_global_p[batch_idx, :, 0, :]  # [frames, 3]
            batch_global_q = pred_global_q[batch_idx]  # [frames, joints, 4]
            predict_filepath = os.path.join(predict_dir, f"predict_{batch_idx:03d}.bvh")
            tgt_skeleton.save_bvh(
                predict_filepath,
                root_pos.detach().cpu(),
                batch_global_q.detach().cpu(),
                frame_time=1 / 30.0
            )

        # -------------------------------
        encode_pcd_vis_dir = os.path.join(epoch_dir, "pcd_vis", "encode", skeleton_name)
        os.makedirs(encode_pcd_vis_dir, exist_ok=True)
        sample_dirs = sorted([d for d in os.listdir(encode_pcd_root_dir) if d.startswith("encode_")])

        for sample_dir in sample_dirs:
            sample_path = os.path.join(encode_pcd_root_dir, sample_dir)
            sample_base = sample_dir.rsplit('.bvh', 1)[0]
            output_video_path = os.path.join(encode_pcd_vis_dir, f"{sample_base}.mp4")
            try:
                pcd_files = load_pcd_files(sample_path)
                if not pcd_files:
                    print(f"Skipping empty folder: {sample_path}")
                    continue
                print(f"Generating video: {skeleton_name}/{sample_dir} -> {output_video_path}")

                visualize_point_cloud_offscreen(
                    pcd_files,
                    output_video_path,
                    fps=30,
                    frame_size=(1280, 720)
                )
            except Exception as e:
                print(f"Processing failed: {sample_path}, error: {str(e)}")
        # -------------------------------

        # -------------------------------
        predict_pcd_vis_dir = os.path.join(epoch_dir, "pcd_vis", "predict", skeleton_name)
        os.makedirs(predict_pcd_vis_dir, exist_ok=True)
        sample_dirs = sorted([d for d in os.listdir(predict_pcd_root_dir) if d.startswith("predict_")])

        for sample_dir in sample_dirs:
            sample_path = os.path.join(predict_pcd_root_dir, sample_dir)
            sample_base = sample_dir.rsplit('.bvh', 1)[0]
            output_video_path = os.path.join(predict_pcd_vis_dir, f"{sample_base}.mp4")
            try:
                pcd_files = load_pcd_files(sample_path)
                if not pcd_files:
                    print(f"Skipping empty folder: {sample_path}")
                    continue
                print(f"Generating video: {skeleton_name}/{sample_dir} -> {output_video_path}")

                visualize_point_cloud_offscreen(
                    pcd_files,
                    output_video_path,
                    fps=30,
                    frame_size=(1280, 720)
                )
            except Exception as e:
                print(f"Processing failed: {sample_path}, error: {str(e)}")
        # -------------------------------

        # -------------------------------
        predict_bvh_vis_dir = os.path.join(epoch_dir, "bvh_vis", "predict", skeleton_name)
        os.makedirs(predict_bvh_vis_dir, exist_ok=True)
        sample_dirs = sorted([d for d in os.listdir(predict_dir) if d.startswith("predict_")])

        for sample_dir in sample_dirs:
            sample_path = os.path.join(predict_dir, sample_dir)
            output_video_path = os.path.join(predict_bvh_vis_dir, f"{sample_dir}.mp4")
            anim, joint_names, frametime = BVH.load(sample_path)
            kinematic_chain = get_kinematic_chain(anim.parents)
            # print(f'This pose shape is',anim.positions.shape)
            joint = anim_pos(anim) 
            plot_3d_motion(output_video_path, kinematic_chain, joints=joint, dataset='bvh_general', title=f"{sample_dir}", fps=20)
        # -------------------------------

        # -------------------------------
        gt_bvh_vis_dir = os.path.join(epoch_dir, "bvh_vis", "gt", skeleton_name)
        os.makedirs(gt_bvh_vis_dir, exist_ok=True)
        sample_dirs = sorted([d for d in os.listdir(gt_dir) if d.startswith("gt_")])

        for sample_dir in sample_dirs:
            sample_path = os.path.join(gt_dir, sample_dir)
            output_video_path = os.path.join(gt_bvh_vis_dir, f"{sample_dir}.mp4")
            anim, joint_names, frametime = BVH.load(sample_path)
            kinematic_chain = get_kinematic_chain(anim.parents)
            # print(f'This pose shape is',anim.positions.shape)
            joint = anim_pos(anim) 
            plot_3d_motion(output_video_path, kinematic_chain, joints=joint, dataset='bvh_general', title=f"{sample_dir}", fps=20)
        # -------------------------------


    def update(self, batch_data):
        loss, acc = self.forward(batch_data)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.item(), acc

    def forward(self, batch_data):

        conds, motion, m_lens = batch_data
        motion = motion.detach().float().to(self.device)
        m_lens = m_lens.detach().long().to(self.device)

        # (b, n, q)
        code_idx, _ = self.vq_model.encode(motion)
        m_lens = m_lens // 4

        conds = conds.to(self.device).float() if torch.is_tensor(conds) else conds

        # loss_dict = {}
        # self.pred_ids = []
        # self.acc = []

        _loss, _pred_ids, _acc = self.t2m_transformer(code_idx[..., 0], conds, m_lens)

        return _loss, _acc
    
    def train(self, epochs, args):

        self.optimizer = torch.optim.AdamW(self.vq_pc.parameters(), lr=0.0001, betas=(0.9, 0.99), weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

        self.motion_save_dir = "/root/autodl-tmp/pcmrl-vis/ckpt_pc-mrl/motion"
        os.makedirs(self.motion_save_dir, exist_ok=True)
 
        self.output_dir = "/root/autodl-tmp/pcmrl-vis/output"
        os.makedirs(self.output_dir, exist_ok=True)

        batch_size = self.cfg["PC-MRL"]["batch_size"]
        max_length = self.cfg["GENERAL"]["max_length"]
        warmup_steps = self.cfg["PC-MRL"]["warmup_steps"]
        src_n_points = self.cfg["PC-MRL"]["src_n_points"]
        tgt_n_points = self.cfg["PC-MRL"]["tgt_n_points"]
        std_cloud = self.cfg["PC-MRL"]["point_sampling_std"]

        epoch = 0
        it = 0

        total_loss = 0.0
        commit_loss_total = 0.0
        
        while epoch < epochs:

            self.vq_pc.train()

            mean_loss = 0.0
            mean_commit_loss = 0.0
            # Process each dataloader corresponding to a skeleton separately
            for loader_idx, dataloader in enumerate(self.dataloaders):

                it += 1

                # Load data for the current skeleton
                root_p, q = dataloader.load_samples(batch_size, max_length)
                global_p, global_q = dataloader.skeleton.fk(root_p.to(self.device), q.to(self.device), local_q=True)
                samples = dataloader.skeleton.generate_pointcloud(global_p, global_q, n_points=src_n_points,
                                                                    std=std_cloud)

                # ==== Key Modification 1: Save the current batch's mean ====
                self.current_means = torch.mean(samples[..., :3], dim=(1, 2), keepdim=True)
                samples[..., :3] -= self.current_means

                # ==== Key Modification 2: Pass the point cloud to PCMG ====

                # Adjust dimensions to fit PCMG: [B,T,N,3]
                pcmg_input = samples[..., :3].requires_grad_(True)
                batch_size, time_steps = samples.shape[0], samples.shape[1]
                mask = torch.ones((batch_size, time_steps), dtype=torch.bool, device=self.device)

                # Construct batch format required by PCMG
                pcmg_batch = {
                    "xyz": pcmg_input,
                    "lengths": torch.full((pcmg_input.shape[0],), pcmg_input.shape[-1], device=self.device),
                    "mask": mask,
                    "y": torch.zeros(batch_size, dtype=torch.long, device=self.device),  # Eight types
                    "cls": torch.zeros(batch_size, dtype=torch.long, device=self.device),
                }

                # PCMG forward pass
                pcmg_output, commit_loss, perplexity = self.pcmg_model(pcmg_batch)

                pcmg_loss = self.pcmg_model.compute_vertices_loss(
                    pcmg_input,  # Original input
                    pcmg_output  # Generated output
                )

                loss = pcmg_loss + 0.02 * commit_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if it >= args.warm_up_iter:
                    self.scheduler.step()

                # Accumulate loss values
                mean_loss += loss.item()
                mean_commit_loss += commit_loss.item()

            
            epoch += 1

            total_loss += mean_loss.item()
            total_commit_loss += mean_commit_loss.item()

            print(f"Epoch {epoch} - Avg Loss: {total_loss / it :.4f}")
            print(f"commit: {total_commit_loss / it :.4f}")

            with open("./ckpt_pc-mrl/log.txt", 'a') as log:
                log.write(f"Epoch {epoch} - Avg Loss: {total_loss / it :.4f}\n")
                log.write(f"commit: {total_commit_loss / it :.4f}\n")

            if epoch % 5000 == 0:
                self.save(epoch)


if __name__ == "__main__":
    torch.set_printoptions(sci_mode=False)

    cfg_file = sys.argv[2] if len(sys.argv) > 1 else './config.json'
    with open(cfg_file, 'r') as f:
        cfg = json.load(f)

    device = torch.device("cuda")

    # Initialize a different skeleton

    cmu_loader = CMUDataLoader(
        cfg["GENERAL"]["cmu_path_train"],
        cfg["GENERAL"]["cmu_skeleton_train"],
        min_length=128,
        data_device=device,
        skel_device=device
    )

    # Get a target skeleton respectively

    cmu_skeleton = CMUDataLoader.setup_skeleton(cfg["GENERAL"]["cmu_skeleton_train"], device)

    trainer = PointCloudTrainer(
        dataloaders=[cmu_loader],
        tgt_skeletons=[cmu_skeleton],  # Pass in two target skeletons
        cfg=cfg,
        device=device
    )
    trainer.train(50000)
