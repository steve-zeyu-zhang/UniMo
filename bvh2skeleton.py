from utils.plot_script_sinmdm import *


def plot_bvh(data, save_dir):
    data = train_data.inv_transformer(data)      ### data is  [dim_9]
    for i in range(len(data)):
        ### input dim_9
        quats = repr6d2quat(torch.tensor(dim_9[:,:,3:])).numpy()
        bvhdir = pjoin(opt.data_root,"bvh") # load template bvh
        anim, joint_names, frametime = BVH.load(os.path.join(bvhdir, [f for f in os.listdir(bvhdir) if f.endswith('.bvh')][0])) # load template bvh
        anim_out = Animation(rotations=Quaternions(quats), positions=dim_9[:, :, :3],
                                orients=anim.orients, offsets=anim.offsets, parents=anim.parents)
        joint = anim_pos(anim_out) 
        save_path = pjoin(save_dir, '%02d.mp4'%(i))

        kinematic_chain = get_kinematic_chain[anim.parents]
        plot_3d_motion_sinmdm(save_path, kinematic_chain, joint, dataset=opt.dataset_name, title="", fps=fps)
        
        
# BVH to data, how to acquisite dim_9 (nf,nj,9)


from utils.plot_script_sinmdm import *  # 导入自定义的绘图工具
import os  # 操作系统接口模块
import os.path as osp  # 路径处理模块
from Motion import BVH  # 自定义的BVH文件处理模块
import numpy as np  # 数值计算库
import torch  # PyTorch深度学习框架
from Motion.transforms import quat2repr6d  # 四元数转6D旋转表示
from glob import glob  # 文件路径匹配模块
from Motion.Animation import positions_global as anim_pos  # 全局位置计算模块
from Motion.Animation import Animation  # 动画数据结构
from Motion.Quaternions import Quaternions  # 四元数处理类
from Motion.transforms import repr6d2quat  # 6D旋转表示转四元数
from Motion.AnimationStructure import get_kinematic_chain  # 获取运动学链结构
from os.path import join as pjoin


bvh_files = glob(osp.join(in_dir,'*.bvh'))
for idx, in_file in enumerate(bvh_files):
    anim, joint_names, frametime = BVH.load(in_file)
    print(f'This pose shape is',anim.positions.shape)
    repr_6d = quat2repr6d(torch.tensor(anim.rotations.qs)) #6d
    dim_9 = np.concatenate([anim.positions, repr_6d], axis=2) # (nf, nj, 9)

