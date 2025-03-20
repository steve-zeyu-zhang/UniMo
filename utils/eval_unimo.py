import torch
from os.path import join as pjoin
import os

from visuals.pcd_video import load_pcd_files, visualize_point_cloud_offscreen
from visuals.ploy_script import plot_3d_motion

from Motion import BVH
from Motion.AnimationStructure import get_kinematic_chain
from Motion.Animation import positions_global as anim_pos

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

            # 恢复预测坐标到全局坐标系
            pred_global_p_adjusted = pred_global_p + means.view(pred_global_p.shape[0], 1, 1, 3)

            batch_size, num_frames = pred_global_p.shape[:2]
            for batch_idx in range(batch_size):
                # 保存预测BVH
                pred_root_pos = pred_global_p_adjusted[batch_idx, :, 0, :]
                pred_bvh_path = pjoin(self.bvh_predict_dir, f"bvh_predict_{batch_idx:03d}.bvh")
                tgt_skeleton.save_bvh(pred_bvh_path,
                                        pred_root_pos.cpu().detach(),
                                        pred_global_q[batch_idx].cpu().detach(),
                                        frame_time=1 / 30.0)

                # 保存Ground Truth BVH
                gt_root_pos = global_p[batch_idx, :, 0, :].cpu()
                gt_bvh_path = pjoin(self.bvh_gt_dir, f"bvh_GT_{batch_idx:03d}.bvh")
                dataloader.skeleton.save_bvh(gt_bvh_path,
                                                gt_root_pos,
                                                global_q[batch_idx].cpu(),
                                                frame_time=1 / 30.0)

            # 保存PCD文件
            pred_pointcloud = tgt_skeleton.generate_pointcloud(pred_global_p_adjusted, pred_global_q,
                                                                n_points=self.cfg["PC-MRL"]["tgt_n_points"],
                                                                std=self.cfg["PC-MRL"]["point_sampling_std"])
            for batch_idx in range(batch_size):
                batch_dir = pjoin(self.pcd_dir, f"pcd_{batch_idx:03d}")
                os.makedirs(batch_dir, exist_ok=True)

                for frame_idx in range(num_frames):
                    frame_pc = pred_pointcloud[batch_idx, frame_idx, ..., :3].cpu().numpy()
                    frame_pc = frame_pc[:, [0, 2, 1]]  # 交换 Y 和 Z 轴
                    frame_pc[:, 2] *= -1  # 反转 Z 轴方向

                    filename = f"frame_{frame_idx:04d}.pcd"
                    filepath = pjoin(batch_dir, filename)
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

    with open(pjoin(self.epoch_dir, "evaluation_log.txt"), 'a') as log:
        log.write(f"Average Loss: {avg_loss:.4f}\n")
        log.write(f"  knn_g: {knn_loss_g_total / len(self.dataloaders):.4f} "
                    f"knn_l: {knn_loss_l_total / len(self.dataloaders):.4f} "
                    f"knn_v: {knn_loss_v_total / len(self.dataloaders):.4f} "
                    f"end: {end_loss_total / len(self.dataloaders):.4f} "
                    f"unit: {unit_loss_total / len(self.dataloaders):.4f}\n")
        
            
@torch.no_grad()
def evaluation_pcde(out_dir, val_loaders_iters, pcde_model, tgt_skeletons, ep, writer, device, save=True, draw=True):
    pcde_model.eval()

    valid_joints_list = []
    d_pose_outputs = []
    for skeleton in tgt_skeletons:
        d_pose_output = 3
        valid_joints = []
        for i in range(len(skeleton.joint_lengths)):
            if skeleton.joint_lengths[i] > 1e-6:
                d_pose_output += 4
                valid_joints.append(i)
        valid_joints_list.append(torch.tensor(valid_joints, dtype=torch.int64, device=device))
        d_pose_outputs.append(d_pose_output)
    tgt_n_points = 1024

    it = 0

    for dataset_id, val_loaders_iter in enumerate(val_loaders_iters):

        batch = next(val_loaders_iter)
        it += 1

        cond, root_p, q, m_lens = batch
        root_p = root_p.to(device).float()
        q = q.to(device).float()

        skeleton = tgt_skeletons[dataset_id]

        global_p, global_q = skeleton.fk(root_p, q, local_q=True)
        samples = skeleton.generate_pointcloud(global_p, global_q, m_lens)

        end_joints = global_p.transpose(0, 2).clone()[skeleton.end_joints].transpose(0, 2)
        
        means = torch.mean(samples[..., :3], dim=(1, 2), keepdim=True)
        samples[..., :3] -= means
        end_joints -= means
        real_global_p = global_p - means
        real_global_q = global_q

        pred = pcde_model(samples)


        skeleton_idx = dataset_id
        valid_joints = valid_joints_list[skeleton_idx]
        d_pose_output = d_pose_outputs[skeleton_idx]
        tgt_skeleton = tgt_skeletons[skeleton_idx]


        pred_p = pred[..., :3]
        pred_q_part = pred[..., 3:d_pose_output]
        pred_q = torch.reshape(pred_q_part, (pred.shape[0], pred.shape[1], -1, 4))


        pred_q_full = torch.zeros((pred.shape[0], pred.shape[1], skeleton.n_joints, 4),
                                    dtype=torch.float32, device=device)
        pred_q_full[..., 0] = 1
        indices = torch.reshape(valid_joints, (1, 1, -1, 1)).repeat(pred_q.shape[0], pred_q.shape[1], 1, 4)
        pred_q_full = torch.scatter(pred_q_full, 2, indices, pred_q)

        pred_global_p, pred_global_q = skeleton.fk(pred_p, pred_q_full, local_q=False)


        pred_pointcloud = skeleton.generate_pointcloud(pred_global_p, pred_global_q, m_lens,
                                                            n_points=tgt_n_points)

        if save:
            eval_pointcloud(samples[..., :3].cpu().numpy(), skeleton, 'encode', out_dir, ep, m_lens=m_lens, draw=draw)
            eval_pointcloud(pred_pointcloud[..., :3].cpu().numpy(), skeleton, 'pred', out_dir, ep, m_lens=m_lens, draw=draw)

            eval_skeleton(global_p, global_q, skeleton, 'gt', out_dir, ep, m_lens=m_lens, draw=draw)
            eval_skeleton(pred_global_p, pred_global_q, skeleton, 'pred', out_dir, ep, m_lens=m_lens, draw=draw)

@torch.no_grad()
def evaluation_pvqvae(out_dir, val_loaders, vq_model, tgt_skeletons, ep, writer, device, save=True, draw=True):
    vq_model.eval()

    assert len(val_loaders) == len(tgt_skeletons)

    valid_joints_list = []
    d_pose_outputs = []
    for skeleton in tgt_skeletons:
        d_pose_output = 3
        valid_joints = []
        for i in range(len(skeleton.joint_lengths)):
            if skeleton.joint_lengths[i] > 1e-6:
                d_pose_output += 4
                valid_joints.append(i)
        valid_joints_list.append(torch.tensor(valid_joints, dtype=torch.int64, device=device))
        d_pose_outputs.append(d_pose_output)

    it = 0

    for dataset_id, val_loader in enumerate(val_loaders):
        for batch in val_loader:
            it += 1

            cond, root_p, q, m_lens = batch
            root_p = root_p.to(device).float()
            q = q.to(device).float()

            skeleton = tgt_skeletons[dataset_id]

            global_p, global_q = skeleton.fk(root_p, q, local_q=True)
            samples = skeleton.generate_pointcloud(global_p, global_q, m_lens)

            # current_means = torch.mean(samples[..., :3], dim=(1, 2), keepdim=True)
            # samples[..., :3] -= current_means
            x = samples[..., :3]

            pred_pointcloud, commit_loss, perplexity = vq_model(x) #[B, L, Num_point, 3]

            # global_pc = motions.clone().detach()
            # global_pc[..., :3] += current_means
            # pc_numpy = global_pc.cpu().numpy()

            if save:
                eval_pointcloud(samples[..., :3].cpu().numpy(), skeleton, 'encode', out_dir, it, m_lens=m_lens, draw=draw)
                eval_pointcloud(pred_pointcloud[..., :3].cpu().numpy(), skeleton, 'pred', out_dir, it, m_lens=m_lens, draw=draw)

                eval_skeleton(global_p, global_q, skeleton, 'gt', out_dir, it, m_lens=m_lens, draw=draw)
                # eval_skeleton(pred_global_p_adjusted, pred_global_q, skeleton, 'pred', out_dir, it, m_lens=m_lens, draw=draw)

 
@torch.no_grad()
def evaluation_unimo(out_dir, val_loaders, vq_model, pc_decoder_model, tgt_skeletons, ep, writer, device, save=True, draw=True):
    vq_model.eval()
    pc_decoder_model.eval()

    assert len(val_loaders) == len(tgt_skeletons)

    valid_joints_list = []
    d_pose_outputs = []
    for skeleton in tgt_skeletons:
        d_pose_output = 3
        valid_joints = []
        for i in range(len(skeleton.joint_lengths)):
            if skeleton.joint_lengths[i] > 1e-6:
                d_pose_output += 4
                valid_joints.append(i)
        valid_joints_list.append(torch.tensor(valid_joints, dtype=torch.int64, device=device))
        d_pose_outputs.append(d_pose_output)

    it = 0
    for dataset_id, val_loader in enumerate(val_loaders):
        
        for batch in val_loader:
            it += 1

            root_p, q = batch
            root_p = root_p.to(device).float()
            q = q.to(device).float()

            skeleton = tgt_skeletons[dataset_id]

            global_p, global_q = skeleton.fk(root_p, q, local_q=True)
            samples = skeleton.generate_pointcloud(global_p, global_q)

            current_means = torch.mean(samples[..., :3], dim=(1, 2), keepdim=True)
            samples[..., :3] -= current_means
            x = samples[..., :3]

            pred_pointcloud, commit_loss, perplexity = vq_model(x) #[B, L, Num_point, 3]


            pred = pc_decoder_model(pred_pointcloud)

            valid_joints = valid_joints_list[dataset_id]
            d_pose_output = d_pose_outputs[dataset_id]

            pred_p = pred[..., :3]
            pred_q_part = pred[..., 3:d_pose_output]
            pred_q = torch.reshape(pred_q_part, (pred.shape[0], pred.shape[1], -1, 4))

            pred_q_full = torch.zeros((pred.shape[0], pred.shape[1], skeleton.n_joints, 4),
                                        dtype=torch.float32, device=device)
            pred_q_full[..., 0] = 1
            indices = torch.reshape(valid_joints, (1, 1, -1, 1)).repeat(pred_q.shape[0], pred_q.shape[1], 1, 4)
            pred_q_full = torch.scatter(pred_q_full, 2, indices, pred_q)

            pred_global_p, pred_global_q = skeleton.fk(pred_p, pred_q_full, local_q=False)

            pred_global_p_adjusted = pred_global_p + current_means.view(pred_global_p.shape[0], 1, 1, 3)

            batch_size, num_frames = pred_global_p.shape[:2]

            pred_pointcloud = skeleton.generate_pointcloud(pred_global_p_adjusted, pred_global_q)
            
            eval_pointcloud(samples, skeleton, 'encode', out_dir, ep, current_means)
            eval_pointcloud(pred_pointcloud, skeleton, 'pred', out_dir, ep, current_means)

            eval_skeleton(global_p, global_q, skeleton, 'gt', out_dir, ep, max_num_saved=5)
            eval_skeleton(pred_global_p_adjusted, pred_global_q, skeleton, 'pred', out_dir, ep, max_num_saved=5)


def save_skeleton(global_p, global_q, skeleton, save_dir, m_lens, max_num_saved = 5):

    batch_size = global_p.shape[0]
    for batch_idx in range(min(max_num_saved, batch_size)):
        root_pos = global_p[batch_idx, :m_lens[batch_idx], 0, :]  # [frames, 3]
        batch_global_q = global_q[batch_idx, :m_lens[batch_idx]]  # [frames, joints, 4]

        predict_filepath = pjoin(save_dir, f"{batch_idx:03d}.bvh")
        skeleton.save_bvh(
            predict_filepath,
            root_pos.detach().cpu(),
            batch_global_q.detach().cpu(),
            frame_time=1 / 20.0
        )

def vis_skeleton(source_bvh_dir, video_save_dir):

    sample_dirs = sorted([d for d in os.listdir(source_bvh_dir)])

    for sample_dir in sample_dirs:
        sample_path = os.path.join(source_bvh_dir, sample_dir)
        sample_dir = sample_dir.rsplit('.bvh', 1)[0]
        output_video_path = os.path.join(video_save_dir, f"{sample_dir}.mp4")

        anim, joint_names, frametime = BVH.load(sample_path)
        kinematic_chain = get_kinematic_chain(anim.parents)

        # print(f'This pose shape is',anim.positions.shape)
        joint = anim_pos(anim) 
        plot_3d_motion(output_video_path, kinematic_chain, joints=joint, dataset='bvh_general', title=f"{sample_dir}", fps=20)


def eval_skeleton(global_p, global_q, skeleton, sub_type, eval_dir, it, m_lens, max_num_saved=5, draw=True):
    bvh_save_dir = pjoin(eval_dir, 'skeleton', skeleton.name, f"{it:04d}", sub_type)
    os.makedirs(bvh_save_dir, exist_ok=True)
    save_skeleton(global_p, global_q, skeleton, bvh_save_dir, m_lens=m_lens, max_num_saved=max_num_saved)

    if draw:
        video_save_dir = pjoin(eval_dir, 'visual_skeleton', skeleton.name, f"{it:04d}", sub_type)
        os.makedirs(video_save_dir, exist_ok=True)
        vis_skeleton(bvh_save_dir, video_save_dir)


def vis_pointcloud(source_pc_dir, video_save_dir):

    sample_dirs = sorted([d for d in os.listdir(source_pc_dir)])

    for sample_dir in sample_dirs:
        sample_path = pjoin(source_pc_dir, sample_dir)

        output_video_path = pjoin(video_save_dir, f"{sample_dir}.mp4")
        pcd_files = load_pcd_files(sample_path)
        if not pcd_files:
            print(f"Skipping empty folder: {sample_path}")
            continue
        # self.logger.info(f"Generating video: {skeleton_name}/{sample_dir} -> {output_video_path}")

        visualize_point_cloud_offscreen(
            pcd_files,
            output_video_path,
            fps=20,
            frame_size=(640, 480)
        )

def save_pointcloud(motions, skeleton, save_dir, m_lens, max_num_saved=5):

    total_num = motions.shape[0]

    for idx in range(min(max_num_saved, total_num)):

        pcdunit_save_dir = pjoin(save_dir, f"{idx:03d}")
        os.makedirs(pcdunit_save_dir, exist_ok=True)

        for frame_idx in range(m_lens[idx]):
            frame_pc = motions[idx, frame_idx]
            filename = f"frame_{frame_idx:04d}.pcd"
            filepath = pjoin(pcdunit_save_dir, filename)
            skeleton.save_as_pcd(frame_pc, filepath)



def eval_pointcloud(motions, skeleton, sub_type, eval_dir, it, m_lens, max_num_saved=5, draw=True):

    pc_save_dir = pjoin(eval_dir, 'pointcloud', skeleton.name, f"{it:04d}", sub_type)
    save_pointcloud(motions, skeleton, pc_save_dir, m_lens=m_lens, max_num_saved=max_num_saved)

    if draw:
        video_save_dir = pjoin(eval_dir, 'visual_pointcloud', skeleton.name, f"{it:04d}", sub_type)
        os.makedirs(video_save_dir, exist_ok=True)
        vis_pointcloud(pc_save_dir, video_save_dir)