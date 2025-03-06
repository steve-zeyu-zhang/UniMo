import os
from os.path import join as pjoin
import json
import re

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from data.cmu_dataset import CmuDataset, setup_skeleton

from models.vq.pcmgvq import PCMGVQ
from models.vq.pvq_trainer import PVQTrainer

from options.vq_option import arg_parse

from utils.fixseed import fixseed
import utils.utils as utils_model
from utils.pcmg_loss import PcmgLoss

os.environ["OMP_NUM_THREADS"] = "1"

if __name__ == "__main__":

    args = arg_parse(True)
    fixseed(args.seed)

    # [Prepare description]
    desc = args.dataset_name  # dataset
    desc += f'-{args.vqvae_cfg}'

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
    args.eval_dir = pjoin(args.run_dir, 'visuals')
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
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    # save the training config
    args.args_save_dir = os.path.join(args.run_dir, 'train_config.json')
    args_dict = vars(args)
    with open(args.args_save_dir, 'wt') as f:
        json.dump(args_dict, f, indent=4)

    args.device = torch.device("cpu" if args.gpu_id == -1 else "cuda:" + str(args.gpu_id))
    print(f"Using Device: {args.device}")

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
        args.latent_dim = 256
        args.input_dim = 3
        args.cmu_path_train = '/root/autodl-tmp/pcmrl-vis/dataset/cmu/train'
        args.cmu_skeleton_train = '/root/autodl-tmp/pcmrl-vis/dataset/cmu/train/001/01_01.bvh'
        args.cmu_path_val = '/root/autodl-tmp/pcmrl-vis/dataset/cmu/val'
        args.density_k = 1024
        args.min_length = 128
        args.src_n_points = 256
        args.std_cloud = 0.05
        args.max_length = 128
        args.dim_pose = args.src_n_points * 3
        args.points_num = args.src_n_points
        args.num_frames = args.max_length

    else:
        raise KeyError('Dataset Does not Exists')

    net = PCMGVQ(args)

    pc_vq = sum(param.numel() for param in net.parameters())
    print(net)
    print("Total parameters of discriminator net: {}".format(pc_vq))
    print('Total parameters of all models: {}M'.format(pc_vq/1000_000))


    train_dataset = CmuDataset(args.cmu_path_train, args.min_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, num_workers=4,
                              shuffle=True, pin_memory=True)
    
    val_dataset = CmuDataset(args.cmu_path_val, args.min_length)
    val_loader = DataLoader(val_dataset, batch_size=5, drop_last=True, num_workers=4,
                              shuffle=True, pin_memory=True)

    skeleton = setup_skeleton(args.cmu_skeleton_train, args.device)


    trainer = PVQTrainer(args, vq_model=net, skeleton=skeleton, logger=logger)
    trainer.train(train_loader, val_loader, skeleton)

## xvfb-run -a python train_pcmrlvq.py --dataset_name cmu --batch_size 12 --name debug --gpu_id 0 --vqvae_cfg pcmg --max_epoch 50000 --eval_every_it 1000 --recons_loss emd
## xvfb-run -a python train_pcmrlvq.py --dataset_name cmu --batch_size 12 --name emd --gpu_id 0 --vqvae_cfg pcmg --max_epoch 8000 --eval_every_it 1000 --recons_loss emd
## xvfb-run -a python train_pcmrlvq.py --dataset_name cmu --batch_size 12 --name cd --gpu_id 0 --vqvae_cfg pcmg --max_epoch 8000 --eval_every_it 1000 --recons_loss cd_density
