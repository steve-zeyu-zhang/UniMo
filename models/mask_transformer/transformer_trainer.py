import torch
from collections import defaultdict
import torch.optim as optim
# import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from utils.utils import *
from os.path import join as pjoin
from utils.eval_unimo import evaluation_pvqvae
from models.mask_transformer.tools import *

from einops import rearrange, repeat

def def_value():
    return 0.0

class MaskTransformerTrainer:
    def __init__(self, args, t2m_transformer, vq_model, skeletons, logger):
        self.opt = args
        self.t2m_transformer = t2m_transformer
        self.vq_model = vq_model
        self.device = args.device
        self.vq_model.eval()
        self.skeletons = skeletons
        self.logger = logger
        self.inner_idx = 0

        if args.is_train:
            self.writter = SummaryWriter(args.log_dir)


    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_t2m_transformer.param_groups:
            param_group["lr"] = current_lr

        return current_lr


    def forward(self, batch_data):
        skeleton = self.skeletons[self.inner_idx]

        conds, root_p, q, m_lens = batch_data
        root_p = root_p.detach().to(self.device).float()
        q = q.detach().to(self.device).float()
        m_lens = m_lens.detach().long().to(self.device)

        global_p, global_q = skeleton.fk(root_p, q, local_q=True)
        samples = skeleton.generate_pointcloud(global_p, global_q)

        # self.current_means = torch.mean(samples[..., :3], dim=(1, 2), keepdim=True)
        # samples[..., :3] -= self.current_means
        x = samples[..., :3]

        # (b, n, q)
        code_idx = self.vq_model.encode(x)
        m_lens = m_lens // 4

        conds = conds.to(self.device).float() if torch.is_tensor(conds) else conds


        _loss, _pred_ids, _acc = self.t2m_transformer(code_idx, conds, m_lens)

        return _loss, _acc

    def update(self, batch_data):
        loss, acc = self.forward(batch_data)

        self.opt_t2m_transformer.zero_grad()
        loss.backward()
        self.opt_t2m_transformer.step()
        self.scheduler.step()

        return loss.item(), acc

    def save(self, file_name, ep, total_it):
        t2m_trans_state_dict = self.t2m_transformer.state_dict()
        clip_weights = [e for e in t2m_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del t2m_trans_state_dict[e]
        state = {
            't2m_transformer': t2m_trans_state_dict,
            'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            'scheduler':self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        missing_keys, unexpected_keys = self.t2m_transformer.load_state_dict(checkpoint['t2m_transformer'], strict=False)
        assert len(unexpected_keys) == 0
        assert all([k.startswith('clip_model.') for k in missing_keys])

        try:
            self.opt_t2m_transformer.load_state_dict(checkpoint['opt_t2m_transformer']) # Optimizer

            self.scheduler.load_state_dict(checkpoint['scheduler']) # Scheduler
        except:
            print('Resume wo optimizer')
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader_iters, val_loaders, eval_val_loader):
        self.t2m_transformer.to(self.device)
        self.vq_model.to(self.device)

        self.opt_t2m_transformer = optim.AdamW(self.t2m_transformer.parameters(), betas=(0.9, 0.99), lr=self.opt.lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.opt_t2m_transformer,
                                                        milestones=self.opt.milestones,
                                                        gamma=self.opt.gamma)

        epoch = 0
        it = 0

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')  # TODO
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()

        total_iters = self.opt.max_epoch * len(train_loader_iters)
        self.logger.info(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')

        # total_iters = self.opt.max_epoch * len(train_loader)
        # print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        # print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))

        logs = defaultdict(def_value, OrderedDict())

        # best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_mask_transformer(
        #     self.opt.save_root, eval_val_loader, self.t2m_transformer, self.vq_model, self.logger, epoch,
        #     best_fid=100, best_div=100,
        #     best_top1=0, best_top2=0, best_top3=0,
        #     best_matching=100, eval_wrapper=eval_wrapper,
        #     plot_func=plot_eval, save_ckpt=False, save_anim=False
        # )
        best_acc = 0.

        evaluation_pvqvae(self.opt.eval_dir, val_loaders, self.vq_model, self.skeletons, it, self.writter, device=self.device, save=True, draw=True)

        while epoch < self.opt.max_epoch:
            for i, train_loader_iter in enumerate(train_loader_iters):
                self.t2m_transformer.train()
                self.vq_model.eval()
                batch = next(train_loader_iter)
                self.inner_idx = i
                it += 1
                if it < self.opt.warm_up_iter:
                    self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)

                loss, acc = self.update(batch_data=batch)
                logs['loss'] += loss
                logs['acc'] += acc
                logs['lr'] += self.opt_t2m_transformer.param_groups[0]['lr']

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    # self.logger.add_scalar('val_loss', val_loss, it)
                    # self.l
                    for tag, value in logs.items():
                        self.writter.add_scalar('Train/%s'%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)
            epoch += 1

            print('Validation time:')
            self.vq_model.eval()
            self.t2m_transformer.eval()

            val_loss = []
            val_acc = []
            with torch.no_grad():
                for dataset_id, val_loader in enumerate(val_loaders):
                    for batch_data in val_loader:
                        loss, acc = self.forward(batch_data)
                        val_loss.append(loss.item())
                        val_acc.append(acc)

            print(f"Validation loss:{np.mean(val_loss):.3f}, accuracy:{np.mean(val_acc):.3f}")

            self.writter.add_scalar('Val/loss', np.mean(val_loss), epoch)
            self.writter.add_scalar('Val/acc', np.mean(val_acc), epoch)

            if np.mean(val_acc) > best_acc:
                print(f"Improved accuracy from {best_acc:.02f} to {np.mean(val_acc)}!!!")
                self.save(pjoin(self.opt.model_dir, 'net_best_acc.tar'), epoch, it)
                best_acc = np.mean(val_acc)

            # best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_mask_transformer(
            #     self.opt.save_root, eval_val_loader, self.t2m_transformer, self.vq_model, self.logger, epoch, best_fid=best_fid,
            #     best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
            #     best_matching=best_matching, eval_wrapper=eval_wrapper,
            #     plot_func=plot_eval, save_ckpt=True, save_anim=(epoch%self.opt.eval_every_e==0)
            # )
