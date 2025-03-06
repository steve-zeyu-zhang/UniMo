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

        # The model needs to support the maximum output dimension
        max_d_pose = max(self.d_pose_outputs)
        self.model = PointCloudDecoder(
            self.cfg["PC-MRL"]["n_bgroups"],
            self.cfg["PC-MRL"]["d_model"],
            max_d_pose,  # Use the maximum output dimension
            self.cfg["PC-MRL"]["temporal_n_heads"],
            self.cfg["PC-MRL"]["temporal_n_layers"],
            self.cfg["PC-MRL"]["k_farthest_first"],
            self.cfg["GENERAL"]["max_length"]
        ).to(self.device)

        # Add mapping from skeleton type to model index
        self.skeleton_mapping = {
            type(dl.skeleton).__name__: idx
            for idx, dl in enumerate(dataloaders)
        }

        self.optimiser = torch.optim.AdamW(self.model.parameters(), lr=self.cfg["PC-MRL"]["learning_rate"],
                                           amsgrad=True)
        self.knn_loss = KNNLoss(self.cfg["PC-MRL"]["n_bgroups"], self.cfg["PC-MRL"]["knn_loss_k"])

        ##### Initialize PCMG #####

        # Get PCMG parameters from the configuration file
        self.pcmg_args = self._init_pcmg_args(cfg)

        # Initialize the PCMG model
        self.pcmg_model = PCMG(self.pcmg_args).to(self.device)

        # PCMG optimizer
        self.pcmg_optimizer = torch.optim.AdamW(
            self.pcmg_model.parameters(),
            lr=self.pcmg_args.lr
        )

        # Storage for PCMG normalization means
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

    def train(self, epochs):
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

        for ep in range(1, epochs + 1):
            lr = self.cfg["PC-MRL"]["learning_rate"] * min(ep / warmup_steps, (epochs - ep) / (epochs - warmup_steps))
            alpha_l = min(ep / warmup_steps, (epochs - ep) / (epochs - warmup_steps))
            for g in self.optimiser.param_groups:
                g["lr"] = lr

            total_loss = 0.0
            knn_loss_g_total = 0.0
            knn_loss_l_total = 0.0
            knn_loss_v_total = 0.0
            end_loss_total = 0.0
            unit_loss_total = 0.0
            commit_loss_total = 0.0

            # Process each dataloader corresponding to a skeleton separately
            for loader_idx, dataloader in enumerate(self.dataloaders):
                

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

                # Get the processed point cloud [B,T,N,3]
                processed_cloud = pcmg_output

                end_joints = global_p.transpose(0, 2).clone()[dataloader.skeleton.end_joints].transpose(0, 2)

                # Denormalization: add back the original mean
                processed_cloud += self.current_means

                # Re-normalize for subsequent processing
                processed_means = torch.mean(processed_cloud, dim=(1, 2), keepdim=True)
                processed_cloud -= processed_means

                    # ==== Key Modification 3: Use the point cloud processed by PCMG ====
                    # Pass the processed point cloud into the PC-MRL model
                pred = self.model(processed_cloud)  # Replace original samples input

                pcmg_loss = self.pcmg_model.compute_vertices_loss(
                    pcmg_input,  # Original input
                    pcmg_output  # Generated output
                )
                # Backpropagation for PCMG
                self.optimiser.zero_grad()
                pcmg_loss.backward()
                # Update PCMG parameters
                self.pcmg_optimizer.step()

                # Select parameters according to the current skeleton
                skeleton_idx = self.skeleton_mapping[type(dataloader.skeleton).__name__]
                valid_joints = self.valid_joints_list[skeleton_idx]
                d_pose_output = self.d_pose_outputs[skeleton_idx]
                tgt_skeleton = self.tgt_skeletons[skeleton_idx]

                # Parse the prediction results
                pred_p = pred[..., :3]
                pred_q_part = pred[..., 3:d_pose_output]
                pred_q = torch.reshape(pred_q_part, (pred.shape[0], pred.shape[1], -1, 4))

                # Fill in the complete quaternion
                pred_q_full = torch.zeros((pred.shape[0], pred.shape[1], tgt_skeleton.n_joints, 4),
                                        dtype=torch.float32, device=self.device)
                pred_q_full[..., 0] = 1
                indices = torch.reshape(valid_joints, (1, 1, -1, 1)).repeat(pred_q.shape[0], pred_q.shape[1], 1, 4)
                pred_q_full = torch.scatter(pred_q_full, 2, indices, pred_q)
                # Forward kinematics
                pred_global_p, pred_global_q = tgt_skeleton.fk(pred_p, pred_q_full, local_q=False)

                # Generate the target point cloud
                pred_pointcloud = tgt_skeleton.generate_pointcloud(pred_global_p, pred_global_q,
                                                                n_points=tgt_n_points, std=std_cloud)

                # Compute the KNN loss
                knn_loss_g, knn_loss_l, knn_loss_v = self.knn_loss(samples, pred_pointcloud)

                # Compute the end-effector joint loss
                pred_end_joints = pred_global_p.transpose(0, 2)[tgt_skeleton.end_joints].transpose(0, 2)
                end_loss = torch.mean(torch.abs(end_joints - pred_end_joints))

                # Compute the unit quaternion loss
                unit_loss = torch.mean(torch.abs(1 - torch.linalg.vector_norm(pred_q, dim=-1)))

                # Total loss
                loss = 1.0 * knn_loss_g + alpha_l * knn_loss_l + knn_loss_v + 0.0 * end_loss + 0.01 * unit_loss + 0.02 * commit_loss
                loss.backward()

                # Accumulate loss values
                total_loss += loss.item()
                knn_loss_g_total += knn_loss_g.item()
                knn_loss_l_total += knn_loss_l.item()
                knn_loss_v_total += knn_loss_v.item()
                end_loss_total += end_loss.item()
                unit_loss_total += unit_loss.item()
                commit_loss_total += commit_loss.item()

                if ep % 200 == 0:
                    print("save vis!")
                    with torch.no_grad():
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

                # Update parameters
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimiser.step()

                # Print statistics
                avg_loss = total_loss / len(self.dataloaders)
                print(f"Epoch {ep} - Avg Loss: {avg_loss:.4f}")
                print(f"  knn_g: {knn_loss_g_total / len(self.dataloaders):.4f} "
                    f"knn_l: {knn_loss_l_total / len(self.dataloaders):.4f} "
                    f"knn_v: {knn_loss_v_total / len(self.dataloaders):.4f} "
                    f"end: {end_loss_total / len(self.dataloaders):.4f} "
                    f"unit: {unit_loss_total / len(self.dataloaders):.4f} "
                    f"commit: {commit_loss_total / len(self.dataloaders):.4f}")

                with open("./ckpt_pc-mrl/log.txt", 'a') as log:
                    log.write(f"Epoch {ep} - Avg Loss: {avg_loss:.4f}\n")
                    log.write(f"  knn_g: {knn_loss_g_total / len(self.dataloaders):.4f} "
                            f"knn_l: {knn_loss_l_total / len(self.dataloaders):.4f} "
                            f"knn_v: {knn_loss_v_total / len(self.dataloaders):.4f} "
                            f"end: {end_loss_total / len(self.dataloaders):.4f} "
                            f"unit: {unit_loss_total / len(self.dataloaders):.4f}"
                            f"commit: {commit_loss_total / len(self.dataloaders):.4f}\n")

                if ep % 5000 == 0:
                    self.save(ep)


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
