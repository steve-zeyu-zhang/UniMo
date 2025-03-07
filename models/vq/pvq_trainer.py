import torch
from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
import torch.nn.functional as F

import torch.optim as optim

import time
from collections import OrderedDict, defaultdict
from utils.utils import print_current_loss
from utils.pcmg_loss import PcmgLoss
from utils.eval_unimo import evaluation_pvqvae

from visuals.pcd_video import load_pcd_files, visualize_point_cloud_offscreen


import os
import sys

def def_value():
    return 0.0


class PVQTrainer:
    def __init__(self, args, vq_model, skeletons, logger):
        self.args = args
        self.vq_model = vq_model
        self.device = args.device
        self.skeletons = skeletons
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

    def forward(self, data_loader_iter):
        skeleton = self.skeletons[self.inner_idx]

        root_p, q = next(data_loader_iter)
        root_p = root_p.to(self.device).float()
        q = q.to(self.device).float()

        global_p, global_q = skeleton.fk(root_p, q, local_q=True)
        samples = skeleton.generate_pointcloud(global_p, global_q)
        

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

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.vq_model.load_state_dict(checkpoint['vq_model'])
        self.opt_vq_model.load_state_dict(checkpoint['opt_vq_model'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader_iters, val_loaders):

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
        total_iters = self.args.max_epoch * len(train_loader_iters)
        self.logger.info(f'Total Epochs: {self.args.max_epoch}, Total Iters: {total_iters}')

        current_lr = self.args.lr
        logs = defaultdict(def_value, OrderedDict())

        assert len(train_loader_iters) == len(self.skeletons)

        while epoch < self.args.max_epoch:
            for i, train_loader_iter in enumerate(train_loader_iters):
                self.vq_model.train()
                it += 1
                self.inner_idx = i
                if it < self.args.warm_up_iter:
                    current_lr = self.update_lr_warm_up(it, self.args.warm_up_iter, self.args.lr)
                loss, vertice_loss, commit_loss, density_loss, perplexity = self.forward(train_loader_iter)
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
                    evaluation_pvqvae(self.args.eval_dir, val_loaders, self.vq_model, self.skeletons, it, self.writter, device=self.device, save=True, draw=True)

                if best_loss > logs['loss']:
                    self.save(pjoin(self.args.model_dir, 'best_loss.tar'), epoch, it)
                    best_loss = logs['loss']

            epoch += 1

            # self.logger.info('Validation')
