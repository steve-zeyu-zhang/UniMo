import os
from os.path import join as pjoin
import json
import re

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from data import uni_dataset

from models.vq.pvqvae import PVQVAE
from models.vq.pcmgvq import PCMGVQ
from models.vq.model import PointVQVAE
from models.vq.pointcloud_AE import PCMGAE

from models.vq.vq_trainer import PVQTrainer

from options.vq_option import arg_parse

from utils.fixseed import fixseed
import utils.utils as utils_model

os.environ["OMP_NUM_THREADS"] = "1"

if __name__ == "__main__":

    args = arg_parse(True)
    fixseed(args.seed)

    # [Prepare description]
    desc = args.dataset_name  # dataset
    desc += f'-{args.vqvae_cfg}'

    args.device = torch.device("cpu" if args.gpu_id == -1 else "cuda:" + str(args.gpu_id))
    print(f"Using Device: {args.device}")

    # Pick output directory.
    prev_run_dirs = []
    outdir = args.out_dir
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    args.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{args.dataset_name}-{args.name}', f'VQVAE-{args.name}-{desc}')
    assert not os.path.exists(args.run_dir)

    args.model_dir = pjoin(args.run_dir, 'model')
    args.meta_dir = pjoin(args.run_dir, 'meta')
    args.eval_dir = pjoin(args.run_dir, 'eval')
    args.log_dir = pjoin(args.run_dir, 'logs')

    print('Creating directory...')
    os.makedirs(args.run_dir)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.meta_dir, exist_ok=True)
    os.makedirs(args.eval_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    ##### ---- Logger ---- #####
    logger = utils_model.get_logger(args.run_dir)
    writer = SummaryWriter(args.run_dir)
    # logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    if args.is_train:
    # save to the disk
        if not os.path.exists(args.run_dir):
            os.makedirs(args.run_dir)
        file_name = os.path.join(args.run_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(vars(args).items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

    if args.dataset_name == "t2m":
        args.data_root = './dataset/HumanML3D/'
        args.motion_dir = pjoin(args.data_root, 'new_joint_vecs')
        args.text_dir = pjoin(args.data_root, 'texts')
        args.joints_num = 22
        dim_pose = 263
        fps = 20
        radius = 4
        # kinematic_chain = paramUtil.t2m_kinematic_chain
        dataset_opt_path = './checkpoints/t2m/Comp_v6_KLD005/args.txt'
        train_split_file = pjoin(args.data_root, 'train.txt')
        val_split_file = pjoin(args.data_root, 'val.txt')
        # train_dataset = MotionDataset(args, mean, std, train_split_file)
        # val_dataset = MotionDataset(args, mean, std, val_split_file)
    elif args.dataset_name == "cmu":
        args.data_root = ''
        args.input_dim = 8
        dataset_path = 'dataset/Jaguar/'
        args.skeleton_train_path = '/root/autodl-tmp/pcmrl-vis/dataset/cmu/train/001/01_01.bvh'

        args.min_length = 128
        args.src_n_points = 256
        args.std_cloud = 0.05
        args.max_length = 128
        args.dim_pose = args.src_n_points * 3

        args.points_num = args.src_n_points
        args.Transformer_pointdecoder_k = 32
        args.Transformer_pointdecoder_num_branch = 1
    elif args.dataset_name == 'Jaguar':
        args.input_dim = 8
        args.min_length = 64
        args.src_n_points = 256
        args.std_cloud = 0.05

        skeleton_train_path = '/root/autodl-tmp/pcmrl-vis/dataset/cmu/train/001/01_01.bvh'
        train_split_file = 'dataset/Jaguar/train.txt'
        val_split_file = 'dataset/Jaguar/val.txt'
        dataset_path = 'dataset/Jaguar/'
        
    else:
        raise KeyError('Dataset Does not Exists')
    
    if args.vq_mode == 'vq':
        net = PointVQVAE(args,
                    args.dim_pose,
                    args.nb_code,
                    args.code_dim,
                    args.code_dim,
                    args.down_t,
                    args.stride_t,
                    args.width,
                    args.depth,
                    args.dilation_growth_rate,
                    args.vq_act,
                    args.vq_norm)
    elif args.vq_mode == 'pvq':
        net = PVQVAE(args.code_dim,
                     args.nb_code,
                     args.input_dim,
                     args.mu)
    elif args.vq_mode == 'pcmgvq':
        net = PCMGVQ(args)
    elif args.vq_mode == 'pcmgae':
        net = PCMGAE(args)

    pc_vq = sum(param.numel() for param in net.parameters())
    print(net)
    print("Total parameters of discriminator net: {}".format(pc_vq))
    print('Total parameters of all models: {}M'.format(pc_vq/1000_000))


    train_loader = uni_dataset.DATALoader(dataset_path, train_split_file, min_length=args.min_length, max_length=196, step=4, batch_size=args.batch_size, drop_last=True, num_workers=4,
                              shuffle=True, pin_memory=True)
    train_loader_iter = uni_dataset.cycle(train_loader)

    val_loader = uni_dataset.DATALoader(dataset_path, val_split_file, min_length=args.min_length, max_length=196, step=4, batch_size=8, drop_last=True, num_workers=4,
                              shuffle=True, pin_memory=True)
    
    skeleton = uni_dataset.setup_skeleton(skeleton_train_path, args.device, args.dataset_name, args.src_n_points, args.std_cloud)

    train_loader_iters = [train_loader_iter]
    val_loaders = [val_loader]
    skeletons = [skeleton]

    trainer = PVQTrainer(args, vq_model=net, skeletons=skeletons, logger=logger)
    trainer.train(train_loader_iters, val_loaders)

## xvfb-run -a python train_vq.py --dataset_name cmu --batch_size 12 --name point --gpu_id 0 --vqvae_cfg default --max_epoch 60000 --eval_every_it 1000 --recons_loss emd
## xvfb-run -a python train_vq.py --dataset_name cmu --batch_size 12 --name emd --gpu_id 0 --vqvae_cfg point --max_epoch 8000 --eval_every_it 1000 --recons_loss emd
## xvfb-run -a python train_vq.py --dataset_name cmu --batch_size 12 --name cd --gpu_id 0 --vqvae_cfg point --max_epoch 8000 --eval_every_it 1000 --recons_loss cd_density
## xvfb-run -a python train_vq.py --dataset_name cmu --batch_size 12 --name emb_density --gpu_id 0 --vqvae_cfg pvq --max_epoch 8000 --eval_every_it 1000 --recons_loss emb_density --vq_mode pvq
## xvfb-run -a python train_vq.py --dataset_name cmu --batch_size 12 --name emb_density --gpu_id 0 --vqvae_cfg pcmgae --max_epoch 40000 --eval_every_it 1000 --recons_loss emb_density --vq_mode pcmgae