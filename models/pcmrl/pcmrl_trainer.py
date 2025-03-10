import torch
from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin

import torch.optim as optim

import time
from collections import OrderedDict, defaultdict
from utils.utils import print_current_loss
from models.pcmrl.utils.knn_loss import KNNLoss
from utils.eval_unimo import evaluation_pcde

def def_value():
    return 0.0

class PCDETrainer:
    def __init__(self, args, pcde_model, skeletons, logger):
        self.args = args
        self.pcde_model = pcde_model
        self.device = args.device
        self.skeletons = skeletons
        self.logger = logger

        if args.is_train:
            self.writter = SummaryWriter(args.log_dir)

        self.valid_joints_list = []
        self.d_pose_outputs = []
        for skeleton in skeletons:
            d_pose_output = 3
            valid_joints = []
            for i in range(len(skeleton.joint_lengths)):
                if skeleton.joint_lengths[i] > 1e-6:
                    d_pose_output += 4
                    valid_joints.append(i)
            self.valid_joints_list.append(torch.tensor(valid_joints, dtype=torch.int64, device=self.device))
            self.d_pose_outputs.append(d_pose_output)
        
        self.knn_loss = KNNLoss(self.args.n_bgroups, self.args.knn_loss_k)

        # self.critic = CriticWrapper(self.args.dataset_name, self.args.device)

    def forward(self, data_loader_iter):
        with torch.no_grad():
            skeleton = self.skeletons[self.inner_idx]

            _, root_p, q, m_lens = next(data_loader_iter)
            root_p = root_p.to(self.device).float()
            q = q.to(self.device).float()

            global_p, global_q = skeleton.fk(root_p, q, local_q=True)
            samples = skeleton.generate_pointcloud(global_p, global_q, m_lens)

            end_joints = global_p.transpose(0, 2).clone()[skeleton.end_joints].transpose(0, 2)
            
            means = torch.mean(samples[..., :3], dim=(1, 2), keepdim=True)
            samples[..., :3] -= means
            end_joints -= means
            real_global_p = global_p - means
            real_global_q = global_q

        pred = self.pcde_model(samples)


        skeleton_idx = self.inner_idx
        valid_joints = self.valid_joints_list[skeleton_idx]
        d_pose_output = self.d_pose_outputs[skeleton_idx]
        tgt_skeleton = self.skeletons[skeleton_idx]


        pred_p = pred[..., :3]
        pred_q_part = pred[..., 3:d_pose_output]
        pred_q = torch.reshape(pred_q_part, (pred.shape[0], pred.shape[1], -1, 4))


        pred_q_full = torch.zeros((pred.shape[0], pred.shape[1], skeleton.n_joints, 4),
                                    dtype=torch.float32, device=self.device)
        pred_q_full[..., 0] = 1
        indices = torch.reshape(valid_joints, (1, 1, -1, 1)).repeat(pred_q.shape[0], pred_q.shape[1], 1, 4)
        pred_q_full = torch.scatter(pred_q_full, 2, indices, pred_q)

        pred_global_p, pred_global_q = skeleton.fk(pred_p, pred_q_full, local_q=False)




        pred_pointcloud = skeleton.generate_pointcloud(pred_global_p, pred_global_q, m_lens,
                                                            n_points=self.args.tgt_n_points)



        knn_loss_g, knn_loss_l, knn_loss_v = self.knn_loss(samples, pred_pointcloud)


        pred_end_joints = pred_global_p.transpose(0, 2)[tgt_skeleton.end_joints].transpose(0, 2)
        end_loss = torch.mean(torch.abs(end_joints - pred_end_joints))


        unit_loss = torch.mean(torch.abs(1 - torch.linalg.vector_norm(pred_q, dim=-1)))



        return knn_loss_g, knn_loss_l, knn_loss_v, end_loss, unit_loss



    # @staticmethod
    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_pcde_model.param_groups:
            param_group["lr"] = current_lr

        return current_lr

    def save(self, file_name, ep, total_it):
        state = {
            "pcde_model": self.pcde_model.state_dict(),
            "opt_pcde_model": self.opt_pcde_model.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.pcde_model.load_state_dict(checkpoint['pcde_model'])
        self.opt_pcde_model.load_state_dict(checkpoint['opt_pcde_model'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader_iters, val_loaders):

        self.pcde_model.to(self.device)
        
        self.opt_pcde_model = optim.AdamW(self.pcde_model.parameters(), lr=self.args.lr, betas=(0.9, 0.99), weight_decay=self.args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt_pcde_model, milestones=self.args.milestones, gamma=self.args.gamma)

        epoch = 0
        it = 0

        best_loss = 99999.0

        # total_loss += loss.item()
        # knn_loss_g_total += knn_loss_g.item()
        # knn_loss_l_total += knn_loss_l.item()
        # knn_loss_v_total += knn_loss_v.item()
        # end_loss_total += end_loss.item()
        # unit_loss_total += unit_loss.item()

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
                self.pcde_model.train()
                it += 1
                self.inner_idx = i

                alpha_l = min(epoch / self.args.warm_up_iter, (self.args.max_epoch - epoch) / (self.args.max_epoch - self.args.warm_up_iter))

                if it < self.args.warm_up_iter:
                    current_lr = self.update_lr_warm_up(it, self.args.warm_up_iter, self.args.lr)

                knn_loss_g, knn_loss_l, knn_loss_v, end_loss, unit_loss = self.forward(train_loader_iter)
                loss = 1.0 * knn_loss_g + alpha_l * knn_loss_l + knn_loss_v + 0.0 * end_loss + 0.01 * unit_loss

                self.opt_pcde_model.zero_grad()
                loss.backward()
                self.opt_pcde_model.step()

                if it >= self.args.warm_up_iter:
                    self.scheduler.step()
                
                logs['loss'] += loss.item()
                logs['knn_loss_g'] += knn_loss_g.item()
                logs['knn_loss_l'] += knn_loss_l.item()
                logs['knn_loss_v'] += knn_loss_v.item()
                logs['end_loss'] += end_loss.item()
                logs['unit_loss'] += unit_loss.item()

                logs['lr'] += self.opt_pcde_model.param_groups[0]['lr']

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
                    evaluation_pcde(self.args.eval_dir, val_loaders, self.pcde_model, self.skeletons, it, self.writter, device=self.device, save=True, draw=True)

                if best_loss > logs['loss']:
                    self.save(pjoin(self.args.model_dir, 'best_loss.tar'), epoch, it)
                    best_loss = logs['loss']

            epoch += 1

            # self.logger.info('Validation')
