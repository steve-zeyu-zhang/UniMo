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

from models.mask_transformer.transformer import MaskTransformer
from models.mask_transformer.transformer_trainer import MaskTransformerTrainer


from options.t2m_option import TrainT2MOptions
from options.vq_option import arg_parse

from utils.get_opt import get_opt
from utils.fixseed import fixseed
import utils.utils as utils_model

os.environ["OMP_NUM_THREADS"] = "1"

def load_vq_model():

    # vqvae training config
    # print('\nLoading training argument...\n')
    # args.vqvae_training_options_path = os.path.join(vqvae_train_dir, 'train_config.json')
    # with open(args.vqvae_training_options_path, 'r') as f:
    #     vqvae_train_args_dict = json.load(f)  # dict
    # vqvae_train_args = EasyDict(vqvae_train_args_dict)  # convert dict to easydict for convenience
    # args.vqvae_train_args = vqvae_train_args

    # opt_path = pjoin(args.vqvae_ck_dir, 'opt.txt')
    # vq_opt = get_opt(opt_path, args.device)

    net = PVQVAE(code_dim=256, nb_code=512, in_dim=3, mu=0.99)


    ckpt = torch.load(args.vqvae_ckpt_path, map_location='cpu')
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    net.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model')

    return net
    # return net, vq_opt

if __name__ == "__main__":

    ##### ---- Exp dirs ---- #####
    parser = TrainT2MOptions()
    args = parser.parse()
    fixseed(args.seed)

    # [Prepare description]
    desc = args.dataset_name  # dataset
    desc += f'-{args.t2m_cfg}'



    # Load VQVAE and its configs
    select_vqvae_ckpt = args.select_vqvae_ckpt

    vqvae_ck_dir = args.vqvae_ck_dir

    if select_vqvae_ckpt == 'last':
        args.vqvae_ckpt_path = os.path.join(vqvae_ck_dir, 'model', 'latest.tar')
    else:
        args.vqvae_ckpt_path = os.path.join(vqvae_ck_dir, 'model', select_vqvae_ckpt + '.pth')


    # Pick output directory.

    # Get running directory: args.run_dir
    prev_run_dirs = []
    parent_dir = os.path.dirname(vqvae_ck_dir)
    if os.path.isdir(parent_dir):
        prev_run_dirs = [x for x in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1

    args.run_dir = os.path.join(parent_dir, f'{cur_run_id:05d}-Trans-{args.name}-{desc}')
    assert not os.path.exists(args.run_dir)

    args.model_dir = pjoin(args.run_dir, 'model')
    args.eval_dir = pjoin(args.run_dir, 'eval')
    args.log_dir = pjoin(args.run_dir, 'logs')

    print('Creating directory...')
    os.makedirs(args.run_dir)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.eval_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    ##### ---- Logger ---- #####
    logger = utils_model.get_logger(args.run_dir)
    writer = SummaryWriter(args.run_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    args.device = torch.device("cpu" if args.gpu_id == -1 else "cuda:" + str(args.gpu_id))
    print(f"Using Device: {args.device}")

    # Preprocess configuration
    if args.dataset_name == "cmu":
        args.cmu_path_train = '/root/autodl-tmp/pcmrl-vis/dataset/cmu/train'
        args.skeleton_train_path = '/root/autodl-tmp/pcmrl-vis/dataset/Jaguar/data/Idle_expansion_edit_Idle_Walk_000059999_expansion_0.25_0.25_seed10_Jaguar-Walk_1.bvh'
        args.cmu_path_val = '/root/autodl-tmp/pcmrl-vis/dataset/cmu/val'
        args.min_length = 64
        args.src_n_points = 256
        args.std_cloud = 0.05

        train_split_file = 'dataset/Jaguar/train.txt'
        val_split_file = 'dataset/Jaguar/val.txt'
        dataset_path = 'dataset/Jaguar/'
    elif args.dataset_name == 'Jaguar':
        args.cmu_path_train = '/root/autodl-tmp/pcmrl-vis/dataset/cmu/train'
        args.skeleton_train_path = '/root/autodl-tmp/pcmrl-vis/dataset/Jaguar/data/Idle_expansion_edit_Idle_Walk_000059999_expansion_0.25_0.25_seed10_Jaguar-Walk_1.bvh'
        args.cmu_path_val = '/root/autodl-tmp/pcmrl-vis/dataset/cmu/val'
        args.min_length = 64
        args.src_n_points = 256
        args.std_cloud = 0.05

        train_split_file = 'dataset/Jaguar/train.txt'
        val_split_file = 'dataset/Jaguar/val.txt'
        dataset_path = 'dataset/Jaguar/'
    else:
        raise KeyError('Dataset Does not Exists')
    


    vq_model = load_vq_model()

    clip_version = 'ViT-B/32'

    args.num_tokens = 512

    t2m_transformer = MaskTransformer(code_dim=256,
                                      cond_mode='text',
                                      latent_dim=args.latent_dim,
                                      ff_size=args.ff_size,
                                      num_layers=args.n_layers,
                                      num_heads=args.n_heads,
                                      dropout=args.dropout,
                                      clip_dim=512,
                                      cond_drop_prob=args.cond_drop_prob,
                                      clip_version=clip_version,
                                      opt=args)

    all_params = 0
    pc_transformer = sum(param.numel() for param in t2m_transformer.parameters_wo_clip())
    # print(t2m_transformer)
    all_params += pc_transformer
    print('Total parameters of all models: {:.2f}M'.format(all_params / 1000_000))


    train_loader = uni_dataset.DATALoader(dataset_path, train_split_file, min_length=args.min_length, max_length=196, step=4, batch_size=args.batch_size, drop_last=True, num_workers=4,
                              shuffle=True, pin_memory=True)
    train_loader_iter = uni_dataset.cycle(train_loader)

    val_loader = uni_dataset.DATALoader(dataset_path, val_split_file, min_length=args.min_length, max_length=196, step=4, batch_size=8, drop_last=True, num_workers=4,
                              shuffle=True, pin_memory=True)


    skeleton = uni_dataset.setup_skeleton(args.skeleton_train_path, args.device, args.dataset_name, args.src_n_points, args.std_cloud)


    train_loader_iters = [train_loader_iter]
    val_loaders = [val_loader]
    skeletons = [skeleton]

    # trainer = PVQTrainer(args, vq_model=net, skeletons=skeletons, logger=logger)
    # trainer.train(train_loader_iters, val_loaders)


    trainer = MaskTransformerTrainer(args, t2m_transformer, vq_model, skeletons=skeletons, logger=logger)

    trainer.train(train_loader_iters, val_loaders, val_loader)

## xvfb-run -a python train_vq.py --dataset_name cmu --batch_size 12 --name point --gpu_id 0 --vqvae_cfg default --max_epoch 60000 --eval_every_it 1000 --recons_loss emd
## xvfb-run -a python train_vq.py --dataset_name cmu --batch_size 12 --name emd --gpu_id 0 --vqvae_cfg point --max_epoch 8000 --eval_every_it 1000 --recons_loss emd
## xvfb-run -a python train_vq.py --dataset_name cmu --batch_size 12 --name cd --gpu_id 0 --vqvae_cfg point --max_epoch 8000 --eval_every_it 1000 --recons_loss cd_density
## xvfb-run -a python train_vq.py --dataset_name cmu --batch_size 12 --name emb_density --gpu_id 0 --vqvae_cfg pvq --max_epoch 8000 --eval_every_it 1000 --recons_loss emb_density --vq_mode pvq
## xvfb-run -a python train_vq.py --dataset_name cmu --batch_size 12 --name emb_density --gpu_id 0 --vqvae_cfg pcmgae --max_epoch 40000 --eval_every_it 1000 --recons_loss emb_density --vq_mode pcmgae