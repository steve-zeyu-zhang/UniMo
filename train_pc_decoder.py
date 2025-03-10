import os
from os.path import join as pjoin
import json
import re

import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter


from data import uni_dataset

from models.pcmrl.pc_mrl import PointCloudDecoder

from models.pcmrl.pcmrl_trainer import PCDETrainer

from options.pcdecoder_option import arg_parse

from utils.fixseed import fixseed
import utils.utils as utils_model

os.environ["OMP_NUM_THREADS"] = "1"

if __name__ == "__main__":

    args = arg_parse(True)
    fixseed(args.seed)

    args.device = torch.device("cpu" if args.gpu_id == -1 else "cuda:" + str(args.gpu_id))

    # Pick output directory.
    desc = args.name  # dataset
    desc += f'-{args.desc}'

    prev_run_dirs = []
    outdir = args.out_dir
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    args.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{args.dataset_name}', f'{desc}')
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


    if args.dataset_name == 'Jaguar':
        args.min_length = 64
        args.max_length =196

        skeleton_train_path = 'dataset/Jaguar/data/Idle_expansion_edit_Idle_Walk_000059999_expansion_0.25_0.25_seed10_Jaguar-Walk_1.bvh'
        train_split_file = 'dataset/Jaguar/train.txt'
        val_split_file = 'dataset/Jaguar/val.txt'
        dataset_path = 'dataset/Jaguar/'
        
    else:
        raise KeyError('Dataset Does not Exists')
    
    ##### ---- Logger ---- #####
    logger = utils_model.get_logger(args.run_dir)
    writer = SummaryWriter(args.run_dir)

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



    train_loader = uni_dataset.DATALoader(dataset_path, train_split_file, min_length=args.min_length, max_length=196, step=4, batch_size=args.batch_size, drop_last=True, num_workers=4,
                              shuffle=True, pin_memory=True)
    train_loader_iter = uni_dataset.cycle(train_loader)

    val_loader = uni_dataset.DATALoader(dataset_path, val_split_file, min_length=args.min_length, max_length=196, step=4, batch_size=8, drop_last=True, num_workers=4,
                              shuffle=True, pin_memory=True)
    
    skeleton = uni_dataset.setup_skeleton(skeleton_train_path, args.device, args.dataset_name, args.src_n_points, args.std_cloud)

    train_loader_iters = [train_loader_iter]
    val_loaders = [val_loader]
    tgt_skeletons = [skeleton]

    valid_joints_list = []
    d_pose_outputs = []
    for skeleton in tgt_skeletons:
        d_pose_output = 3
        valid_joints = []
        for i in range(len(skeleton.joint_lengths)):
            if skeleton.joint_lengths[i] > 1e-6:
                d_pose_output += 4
                valid_joints.append(i)
        valid_joints_list.append(torch.tensor(valid_joints, dtype=torch.int64, device=args.device))
        d_pose_outputs.append(d_pose_output)
    max_d_pose = max(d_pose_outputs)

    net = PointCloudDecoder(args.n_bgroups,
            args.d_model,
            max_d_pose,  
            args.temporal_n_heads,
            args.temporal_n_layers,
            args.k_farthest_first,
            args.max_length)

    pc_pcde = sum(param.numel() for param in net.parameters())
    logger.info(net)
    logger.info('Total parameters of all models: {}M'.format(pc_pcde/1000_000))


    trainer = PCDETrainer(args, pcde_model=net, skeletons=tgt_skeletons, logger=logger)
    trainer.train(train_loader_iters, val_loaders)