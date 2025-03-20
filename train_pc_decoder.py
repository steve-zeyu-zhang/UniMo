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
        args.max_length = 196

        dataset_names = ['Jaguar']
        skeleton_train_paths = ['dataset/Jaguar/motions/Idle_expansion_edit_Idle_Walk_000059999_expansion_0.25_0.25_seed10_Jaguar-Walk_1.bvh']

        dataset_paths = ['dataset/Jaguar/']
        train_split_files = [pjoin(dataset_path, 'train.txt') for dataset_path in dataset_paths]
        val_split_files = [pjoin(dataset_path, 'val.txt') for dataset_path in dataset_paths]
    elif args.dataset_name == 'AAJ':
        args.min_length = 64
        args.max_length = 196

        dataset_names = ['Jaguar', 'AnimalML3D']
        skeleton_train_paths = ['dataset/Jaguar/motions/Idle_expansion_edit_Idle_Walk_000059999_expansion_0.25_0.25_seed10_Jaguar-Walk_1.bvh',
                                'dataset/AnimalML3D/motions/bear9AK_SwimIdleRM.bvh']

        dataset_paths = ['dataset/Jaguar/', 'dataset/AnimalML3D/']
        train_split_files = [pjoin(dataset_path, 'train.txt') for dataset_path in dataset_paths]
        val_split_files = [pjoin(dataset_path, 'val.txt') for dataset_path in dataset_paths]
    elif args.dataset_name == 'AnimalML3D':
        args.min_length = 64
        args.max_length = 196

        dataset_names = ['AnimalML3D']
        skeleton_train_paths = ['dataset/AnimalML3D/motions/bear9AK_SwimIdleRM.bvh']

        dataset_paths = ['dataset/AnimalML3D/']
        train_split_files = [pjoin(dataset_path, 'train.txt') for dataset_path in dataset_paths]
        val_split_files = [pjoin(dataset_path, 'val.txt') for dataset_path in dataset_paths]
    elif args.dataset_name == 'Dog':
        args.min_length = 64
        args.max_length = 196

        dataset_names = ['Dog']
        skeleton_train_paths = ['dataset/Dog/motions/Floor-Rise-Right_expansion_edit_Floor-Rise-Right_Running_000059999_expansion_0.25_0.25_seed10_Dog-Running_0.bvh']

        dataset_paths = ['dataset/Dog/']
        train_split_files = [pjoin(dataset_path, 'train.txt') for dataset_path in dataset_paths]
        val_split_files = [pjoin(dataset_path, 'val.txt') for dataset_path in dataset_paths]
    elif args.dataset_name == 'JAC':
        args.min_length = 64
        args.max_length = 196

        dataset_names = ['Jaguar', 'Coyote']
        skeleton_train_paths = ['dataset/Jaguar/motions/Idle_expansion_edit_Idle_Walk_000059999_expansion_0.25_0.25_seed10_Jaguar-Walk_1.bvh', 
                               'dataset/Coyote/motions/Attack3_expansion_edit_Attack3_Walking_000059999_expansion_0.25_0.25_seed10_Coyote-Walking_18.bvh']

        dataset_paths = ['dataset/Jaguar/', 'dataset/Coyote/']
        train_split_files = [pjoin(dataset_path, 'train.txt') for dataset_path in dataset_paths]
        val_split_files = [pjoin(dataset_path, 'val.txt') for dataset_path in dataset_paths]
    elif args.dataset_name == 'UniML3D':
        args.min_length = 64
        args.max_length = 196

        dataset_names = ['Jaguar', 'Coyote', 'Dog']
        skeleton_train_paths = ['dataset/Jaguar/motions/Idle_expansion_edit_Idle_Walk_000059999_expansion_0.25_0.25_seed10_Jaguar-Walk_1.bvh', 
                               'dataset/Coyote/motions/Attack3_expansion_edit_Attack3_Walking_000059999_expansion_0.25_0.25_seed10_Coyote-Walking_18.bvh',
                               'dataset/Dog/motions/Floor-Rise-Right_expansion_edit_Floor-Rise-Right_Running_000059999_expansion_0.25_0.25_seed10_Dog-Running_0.bvh']

        dataset_paths = ['dataset/Jaguar/', 'dataset/Coyote/', 'dataset/Dog/']
        train_split_files = [pjoin(dataset_path, 'train.txt') for dataset_path in dataset_paths]
        val_split_files = [pjoin(dataset_path, 'val.txt') for dataset_path in dataset_paths]
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


    src_skeletons = [uni_dataset.setup_skeleton(skeleton_train_path, args.device, dataset_name, args.src_n_points, args.std_cloud) for skeleton_train_path, dataset_name in zip(skeleton_train_paths, dataset_names)]
    tgt_skeletons = src_skeletons
    
    train_loader_iters = [uni_dataset.cycle(uni_dataset.DATALoader(dataset_path, train_split_file, skeleton, min_length=args.min_length, max_length=args.max_length, step=4, batch_size=args.batch_size, drop_last=True, num_workers=4,
                              shuffle=True, pin_memory=True)) for dataset_path, train_split_file, skeleton in zip(dataset_paths, train_split_files, src_skeletons)]
    
    val_loaders_iters = [uni_dataset.cycle(uni_dataset.DATALoader(dataset_path, val_split_file, skeleton, min_length=args.min_length, max_length=args.max_length, step=4, batch_size=6, drop_last=True, num_workers=4,
                              shuffle=True, pin_memory=True)) for dataset_path, val_split_file, skeleton in zip(dataset_paths, val_split_files, src_skeletons)]

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
    trainer.train(train_loader_iters, val_loaders_iters)

## xvfb-run -a python train_pc_decoder.py --dataset_name UniML3D --name PCDE --desc 12bc_50000ep --gpu_id 0 --max_epoch 50000 --eval_every_it 2000
## xvfb-run -a python train_pc_decoder.py --dataset_name Jaguar --name PCDE --desc 12bc_50000ep --gpu_id 0 --max_epoch 50000 --eval_every_it 500
## xvfb-run -a python train_pc_decoder.py --dataset_name AnimalML3D --name PCDE --desc 12bc_50000ep --gpu_id 0 --max_epoch 50000 --eval_every_it 2000