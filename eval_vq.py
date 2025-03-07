import os
from os.path import join as pjoin
import json
import re

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from data import cmu_dataset

from models.vq.pvqvae import PVQVAE
from models.vq.pcmgvq import PCMGVQ
from models.vq.model import PointVQVAE
from models.vq.pointcloud_AE import PCMGAE
from models.pcmrl.pc_mrl import PointCloudDecoder

from models.vq.pvq_trainer import PVQTrainer

from options.vq_option import arg_parse

from utils.fixseed import fixseed
import utils.utils as utils_model
from utils.eval_unimo import evaluation_unimo

os.environ["OMP_NUM_THREADS"] = "1"

if __name__ == "__main__":

    args = arg_parse(True)
    fixseed(args.seed)

    # [Prepare description]
    desc = args.dataset_name  # dataset
    desc += f'-{args.vqvae_cfg}'

    args.checkpoints_dir = 'checkpoints/00039-cmu-emb_density/VQVAE-emb_density-cmu-pvq'
    args.out_dir = pjoin(args.checkpoints_dir, 'eval')
    # Pick output directory.
    prev_run_dirs = []
    outdir = args.out_dir
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    args.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{args.dataset_name}-{args.name}')
    assert not os.path.exists(args.run_dir)

    args.eval_dir = pjoin(args.run_dir, 'eval')
    args.log_dir = pjoin(args.run_dir, 'logs')

    print('Creating directory...')
    os.makedirs(args.run_dir)

    os.makedirs(args.eval_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    ##### ---- Logger ---- #####
    logger = utils_model.get_logger(args.run_dir)
    writer = SummaryWriter(args.run_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    # save the training config
    args.args_save_dir = os.path.join(args.run_dir, 'eval_config.json')
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

        args.ck_vq_path = pjoin(args.checkpoints_dir, 'model', 'latest.tar')
        args.ck_pcmrl_path = 'checkpoints/pcmrl/PCMRL50000.pt'

        args.min_length = 128
        args.src_n_points = 256
        args.std_cloud = 0.05
        args.max_length = 128
        args.dim_pose = args.src_n_points * 3

        args.points_num = args.src_n_points
        args.Transformer_pointdecoder_k = 32
        args.Transformer_pointdecoder_num_branch = 1

        ###
        args.tgt_n_points = 1024
        args.point_sampling_std=0.05
        args.n_bgroups=5
        args.d_model_ = "Base PointTransformer dimension size, aggregated point cloud vector is 16x larger"
        args.d_model = 64
        args.k_farthest_first = 3
        args.temporal_n_heads = 4
        args.temporal_n_layers = 6
        args.learning_rate = 0.0001
        args.warmup_steps = 500
        args.knn_loss_k = 8
        
        ###


        args.vq_mode == 'pvq'
    else:
        raise KeyError('Dataset Does not Exists')
    


    val_loader = cmu_dataset.DATALoader(args.cmu_path_val, args.min_length, batch_size=12, drop_last=True, num_workers=4,
                              shuffle=True, pin_memory=True)
    val_loader_iter = cmu_dataset.cycle(val_loader)

    val_loaders = [val_loader]

    skeleton = cmu_dataset.setup_skeleton(args.cmu_skeleton_train, args.device, args.dataset_name, args.src_n_points, args.std_cloud)
    skeletons = [skeleton]



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
        net = PVQVAE(args)
    elif args.vq_mode == 'pcmgvq':
        net = PCMGVQ(args)
    elif args.vq_mode == 'pcmgae':
        net = PCMGAE(args)

    vq_trainer_ck = torch.load(args.ck_vq_path)
    net.load_state_dict(vq_trainer_ck['vq_model'])
    net.to(args.device)
    net.eval()

    pc_vq = sum(param.numel() for param in net.parameters())
    print(net)
    print("Total parameters of discriminator net: {}".format(pc_vq))
    print('Total parameters of all models: {}M'.format(pc_vq/1000_000))


    valid_joints_list = []
    d_pose_outputs = []
    for skeleton in skeletons:
        d_pose_output = 3
        valid_joints = []
        for i in range(len(skeleton.joint_lengths)):
            if skeleton.joint_lengths[i] > 1e-6:
                d_pose_output += 4
                valid_joints.append(i)
        valid_joints_list.append(torch.tensor(valid_joints, dtype=torch.int64, device=args.device))
        d_pose_outputs.append(d_pose_output)

    max_d_pose = max(d_pose_outputs)

    decoder = PointCloudDecoder(
        args.n_bgroups,
        args.d_model,
        max_d_pose,
        args.temporal_n_heads,
        args.temporal_n_layers,
        args.k_farthest_first,
        args.max_length
    ).to(args.device)

    decoder.load_state_dict(torch.load(args.ck_pcmrl_path))
    decoder.to(args.device)
    decoder.eval()

    
    pc_vq = sum(param.numel() for param in decoder.parameters())
    print(decoder)
    print("Total parameters of discriminator net: {}".format(pc_vq))
    print('Total parameters of all models: {}M'.format(pc_vq/1000_000))

    ep = 0
    while ep <= args.max_epoch:
        ep += 1
        evaluation_unimo(args.run_dir, val_loaders, net, decoder, skeletons, ep, writer, args.device)

## python eval_vq.py --dataset_name cmu


