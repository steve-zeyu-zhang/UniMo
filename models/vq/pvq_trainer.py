import torch
from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
import torch.nn.functional as F

import torch.optim as optim

import time
from collections import OrderedDict, defaultdict
from utils.utils import print_current_loss
from utils.pcmg_loss import PcmgLoss

from visuals.pcd_video import load_pcd_files, visualize_point_cloud_offscreen

import os
import sys

def def_value():
    return 0.0


class PVQTrainer:
    def __init__(self, args, vq_model, skeleton, logger):
        self.args = args
        self.vq_model = vq_model
        self.device = args.device
        self.skeleton = skeleton
        self.logger = logger
        losses = PcmgLoss(density_k=1024)

        if args.is_train:
            self.writter = SummaryWriter(args.log_dir)
            if args.recons_loss == 'l1':
                self.l1_criterion = torch.nn.L1Loss()
            elif args.recons_loss == 'l1_smooth':
                self.l1_criterion = torch.nn.SmoothL1Loss()
            elif args.recons_loss == 'emb':
                self.emd_criterion = losses.compute_emd_loss
            elif args.recons_loss == 'cd_density':
                self.distance_criterion = losses.chamfer_distance_loss
                self.density_criterion = losses.compute_density_loss
            elif args.recons_loss == 'emb_density':
                self.emd_criterion = losses.compute_emd_loss
                self.density_criterion = losses.compute_density_loss

        # self.critic = CriticWrapper(self.args.dataset_name, self.args.device)

    def forward(self, batch_data):

        root_p, q = batch_data
        root_p = root_p.to(self.device).float()
        q = q.to(self.device).float()

        global_p, global_q = self.skeleton.fk(root_p, q, local_q=True)
        samples = self.skeleton.generate_pointcloud(global_p, global_q, n_points=self.args.src_n_points,
                                                            std=self.args.std_cloud)
        

        self.current_means = torch.mean(samples[..., :3], dim=(1, 2), keepdim=True)
        samples[..., :3] -= self.current_means
        x = samples[..., :3]

        pred_pointcloud, commit_loss, perplexity = self.vq_model(x) #[B, L, Num_point, 3]
        
        self.motions = x
        self.pred_motion = pred_pointcloud
        
        if self.args.recons_loss == 'emb':
            vertice_loss = self.emd_criterion(x, pred_pointcloud)
            dentity_loss = torch.Tensor([0.0]).to(self.device)
        elif self.args.recons_loss == 'cd_density':
            vertice_loss = self.distance_criterion(x, pred_pointcloud)
            dentity_loss = self.density_criterion(x, pred_pointcloud)
        elif self.args.recons_loss == 'emb_density':
            vertice_loss = self.emd_criterion(x, pred_pointcloud)
            dentity_loss = self.density_criterion(x, pred_pointcloud)
        
        loss = vertice_loss + self.args.commit * commit_loss + dentity_loss

        return loss, vertice_loss, commit_loss, dentity_loss, perplexity


    # @staticmethod
    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_vq_model.param_groups:
            param_group["lr"] = current_lr

        return current_lr

    def save(self, file_name, ep, total_it):
        state = {
            "vq_model": self.vq_model.state_dict(),
            "opt_vq_model": self.opt_vq_model.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def vis_pointcloud(self, pcd_type="predict", it=0):
        root_dir = self.args.eval_dir
        skeleton_name = self.args.dataset_name

        pcd_root_dir = pjoin(root_dir, 'pcd', skeleton_name, f"{it:04d}", pcd_type)
        video_save_dir = os.path.join(root_dir, "pcd_vis", skeleton_name, f"{it:04d}", pcd_type)
        os.makedirs(video_save_dir, exist_ok=True)

        sample_dirs = sorted([d for d in os.listdir(pcd_root_dir) if d.startswith(f"{pcd_type}_")])

        for sample_dir in sample_dirs:
            sample_path = os.path.join(pcd_root_dir, sample_dir)
            sample_base = sample_dir.rsplit('.bvh', 1)[0]
            output_video_path = os.path.join(video_save_dir, f"{sample_base}.mp4")
            try:
                pcd_files = load_pcd_files(sample_path)
                if not pcd_files:
                    # self.logger.info(f"Skipping empty folder: {sample_path}")
                    continue
                # self.logger.info(f"Generating video: {skeleton_name}/{sample_dir} -> {output_video_path}")

                visualize_point_cloud_offscreen(
                    pcd_files,
                    output_video_path,
                    fps=30,
                    frame_size=(640, 480)
                )
            except Exception as e:
                self.logger.error(f"Processing failed: {sample_path}, error: {str(e)}")

    def eval_pointcloud(self, it):
        samples = self.motions
        pcd_type = 'encode'
        encode_pcd_root_dir = os.path.join(self.args.eval_dir, 'pcd', self.args.dataset_name, f"{it:04d}", pcd_type)
        os.makedirs(encode_pcd_root_dir, exist_ok=True)

        encode_global_pc = samples.clone().detach()
        encode_global_pc[..., :3] += self.current_means
        encode_pc_numpy = encode_global_pc.cpu().numpy()

        for batch_idx in range(min(5, self.args.batch_size)):
            encoder_pcd_dir = os.path.join(encode_pcd_root_dir, f"{pcd_type}_{batch_idx:03d}")
            os.makedirs(encoder_pcd_dir, exist_ok=True)
            for frame_idx in range(self.args.max_length):
                frame_pc = encode_pc_numpy[batch_idx, frame_idx]
                filename = f"frame_{frame_idx:04d}.pcd"
                filepath = os.path.join(encoder_pcd_dir, filename)
                self.skeleton.save_as_pcd(frame_pc, filepath)
        self.vis_pointcloud(pcd_type, it)


        processed_cloud = self.pred_motion
        pcd_type = 'predict'
        predict_pcd_root_dir = os.path.join(self.args.eval_dir, 'pcd', self.args.dataset_name, f"{it:04d}", pcd_type)
        os.makedirs(predict_pcd_root_dir, exist_ok=True)

        pred_global_pc = processed_cloud.clone().detach()
        pred_global_pc[..., :3] += self.current_means
        pred_pc_numpy = pred_global_pc.cpu().numpy()

        for batch_idx in range(min(5, self.args.batch_size)):
            predict_pcd_dir = os.path.join(predict_pcd_root_dir, f"{pcd_type}_{batch_idx:03d}")
            os.makedirs(predict_pcd_dir, exist_ok=True)
            for frame_idx in range(self.args.max_length):
                frame_pc = pred_pc_numpy[batch_idx, frame_idx]
                filename = f"frame_{frame_idx:04d}.pcd"
                filepath = os.path.join(predict_pcd_dir, filename)
                self.skeleton.save_as_pcd(frame_pc, filepath)
        self.vis_pointcloud(pcd_type, it)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.vq_model.load_state_dict(checkpoint['vq_model'])
        self.opt_vq_model.load_state_dict(checkpoint['opt_vq_model'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader, plot_eval=None):

        self.vq_model.to(self.device)

        self.opt_vq_model = optim.AdamW(self.vq_model.parameters(), lr=self.args.lr, betas=(0.9, 0.99), weight_decay=self.args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt_vq_model, milestones=self.args.milestones, gamma=self.args.gamma)

        epoch = 0
        it = 0

        best_loss = 99999.0

        if self.args.is_continue:
            model_dir = pjoin(self.args.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)
            self.logger.info("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        total_iters = self.args.max_epoch * len(train_loader)
        self.logger.info(f'Total Epochs: {self.args.max_epoch}, Total Iters: {total_iters}')

        current_lr = self.args.lr
        logs = defaultdict(def_value, OrderedDict())


        while epoch < self.args.max_epoch:
            self.vq_model.train()
            for i, batch_data in enumerate(train_loader):
                it += 1
                if it < self.args.warm_up_iter:
                    current_lr = self.update_lr_warm_up(it, self.args.warm_up_iter, self.args.lr)
                loss, vertice_loss, commit_loss, density_loss, perplexity = self.forward(batch_data)
                self.opt_vq_model.zero_grad()
                loss.backward()
                self.opt_vq_model.step()

                if it >= self.args.warm_up_iter:
                    self.scheduler.step()
                
                logs['loss'] += loss.item()
                logs['vertice_loss'] += vertice_loss.item()
                logs['density_loss'] += density_loss.item()
                # Note it not necessarily velocity, too lazy to change the name now
                logs['commit_loss'] += commit_loss.item()
                logs['perplexity'] += perplexity.item()
                logs['lr'] += self.opt_vq_model.param_groups[0]['lr']

                if it % self.args.log_every_it == 0:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        self.writter.add_scalar('Train/%s'%tag, value / self.args.log_every_it, it)
                        mean_loss[tag] = value / self.args.log_every_it
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i, logger=self.logger)

                if it % self.args.save_every_it == 0:
                    self.save(pjoin(self.args.model_dir, 'latest.tar'), epoch, it)

                if it % self.args.eval_every_it == 0:
                    self.logger.info("visualization...")
                    self.eval_pointcloud(it)

                if best_loss > logs['loss']:
                    self.save(pjoin(self.args.model_dir, 'best_loss.tar'), epoch, it)
                    best_loss = logs['loss']

            epoch += 1

            # self.logger.info('Validation')
