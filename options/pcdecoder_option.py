import argparse
import os
import torch

def arg_parse(is_train=False):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## dataloader
    parser.add_argument('--dataset_name', type=str, default='humanml3d', help='dataset directory')
    parser.add_argument('--batch_size', default=12, type=int, help='batch size')
    parser.add_argument("--gpu_id", type=int, default=0, help='GPU id')

    ## optimization
    parser.add_argument('--max_epoch', default=5000, type=int, help='number of total epochs to run')
    parser.add_argument('--warm_up_iter', default=500, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=1e-4, type=float, help='max learning rate')
    parser.add_argument('--milestones', default=[150000, 250000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")
    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')


    ## decoder arch
    parser.add_argument("--src_n_points", default=256, type=int, help="sk -> pc Encoder The number of points in the point cloud")
    parser.add_argument("--tgt_n_points", default=1024, type=int, help="sk -> pc Decoder The number of points in the point cloud")
    parser.add_argument("--std_cloud", default=0.05, type=float, help="point sampling std")
    parser.add_argument("--n_bgroups", default=5, type=int, help=" ")
    parser.add_argument("--d_model", default=64, type=int, help="Base PointTransformer dimension size, aggregated point cloud vector is 16x larger")
    parser.add_argument("--k_farthest_first", default=8, type=int, help=" ")
    parser.add_argument("--temporal_n_heads", default=4, type=int, help=" ")
    parser.add_argument("--temporal_n_layers", default=6, type=int, help=" ")
    parser.add_argument("--knn_loss_k", default=8, type=int, help=" ")

    ## other
    parser.add_argument('--name', type=str, default="PCDE", help='Name of this trial')
    parser.add_argument('--desc', type=str, default="test", help='desc of this trail')
    parser.add_argument('--is_continue', action="store_true", help='is continue training')
    parser.add_argument('--out_dir', type=str, default='./output', help='output are saved here')
    parser.add_argument('--log_every_it', default=10, type=int, help='iter log frequency')
    parser.add_argument('--save_every_it', default=200, type=int, help='iter save latest model frequency')
    parser.add_argument('--save_every_e', default=2, type=int, help='save model every n epoch')
    parser.add_argument('--eval_every_it', default=100, type=int, help='save eval results every n epoch')

    parser.add_argument('--which_epoch', type=str, default="all", help='Name of this trial')

    parser.add_argument("--seed", default=42, type=int)


    opt = parser.parse_args()
    torch.cuda.set_device(opt.gpu_id)

    args = vars(opt)

    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    opt.is_train = is_train

    return opt