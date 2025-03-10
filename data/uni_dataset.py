from os.path import join as pjoin
import torch
from torch.utils import data
from torch.utils.data._utils.collate import default_collate
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_quaternion
import random
import glob
from data.skeleton import Skeleton
from data.bvh import parse_bvh_skeleton
import numpy as np
from tqdm import tqdm
import codecs as cs


def setup_skeleton(skeleton_file, device, skeleton_name, n_point=-1, std=-1):
        with open(skeleton_file, "r") as f:
            joint_names, joint_offsets, joint_hierarchy, end_sites = parse_bvh_skeleton(f.read())

        children = [list() for _ in range(len(joint_names))]
        for i, parent_idx in enumerate(joint_hierarchy):
            if parent_idx >= 0:
                children[parent_idx].append(i)

        # 检测身体部位分组
        joint_bgroups = torch.zeros(len(joint_names), 5, dtype=torch.float32, device=device)
        spine_keywords = {'spine', 'neck', 'head', 'hips', 'pelvis', 'back',
                          'tail',
                          'mouth', 'ear'}
        arm_keywords = {'arm', 'shoulder', 'hand', 'clavicle', 'finger',
                        'horse', 'hoof', 'paw'}
        leg_keywords = {'leg', 'thigh', 'calf', 'foot', 'toe', 'hip',
                        'hind', 'equine', 'quadruped'}

        # 自动识别对称关节对
        pairs = []
        left_joints = []
        right_joints = []

        for i, name in enumerate(joint_names):
            lower_name = name.lower()

            # 脊椎检测（包含尾巴、头部细节）
            if any(kw in lower_name for kw in spine_keywords):
                joint_bgroups[i, 0] = 1  # 脊椎组

            # 四肢检测
            is_left = any([tag in lower_name for tag in ['left', '_l_']])  # 支持多种左标记
            is_right = any([tag in lower_name for tag in ['right', '_r_']])

            # 前肢/后肢判断逻辑
            if is_left:
                if any(kw in lower_name for kw in arm_keywords):
                    joint_bgroups[i, 1] = 1  # 左前肢
                elif any(kw in lower_name for kw in leg_keywords):
                    joint_bgroups[i, 3] = 1  # 左后肢

            if is_right:
                if any(kw in lower_name for kw in arm_keywords):
                    joint_bgroups[i, 2] = 1  # 右前肢
                elif any(kw in lower_name for kw in leg_keywords):
                    joint_bgroups[i, 4] = 1  # 右后肢

        # 建立对称关节映射
        for li, lname in left_joints:
            rname = lname.replace("Left", "Right").replace("_L_", "_R_")
            if rname in joint_names:
                ri = joint_names.index(rname)
                pairs.append((li, ri))

        # 自动识别末端关节
        end_joints = [name for i, name in enumerate(joint_names)
                      if len(children[i]) == 0 or  # 没有子节点
                      any(kw in name.lower() for kw in {'hand', 'foot', 'toe'})]

        # ======
        # 先构建原始数据列表，再统一转换为Tensor
        joint_tails = []
        for i, name in enumerate(joint_names):
            if len(children[i]) == 0:
                # 确保end_sites返回的是三维坐标列表
                tail = end_sites[name]
            else:
                # 使用原始列表而不是Tensor索引
                tail = joint_offsets[children[i][0]]
            joint_tails.append(tail)


        joint_offsets = torch.tensor(joint_offsets, dtype=torch.float32, device=device)
        joint_tails = torch.tensor(joint_tails, dtype=torch.float32, device=device)
        # ====================
        return Skeleton(joint_names, joint_hierarchy, joint_offsets, joint_tails, joint_bgroups, end_joints, pairs, skeleton_name=skeleton_name, n_point=n_point, std=std, 
                    scale=0.056444, device=device, rotation_order="ZYX")

                        
class UniDataset(data.Dataset):
    def __init__(self, dataset_path, split_file, min_length=64, max_length=196, step=1):

        self.src_n_points = 256
        self.length = min_length
        self.SCALE = 0.056444
        self.std_cloud = 0.05
        self.min_length = min_length
        self.max_length = max_length
        self.pointer = 0
        self.step = step

        id_list = []
        with open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())


        data_dict = {}
        name_list = []
        for name in tqdm(id_list):
            
            bvh_file = pjoin(dataset_path, './data/', name + '.bvh')
            with open(bvh_file, "r") as f:

                while f.readline() != "MOTION\n":
                    continue

                motion = f.read()
                lines = motion.split('\n')
                sub_motion_data = []
                for line in lines[2:]:
                    words = line.strip().split(" ")
                    try:
                        sub_motion_data.append(list(map(float, words)))
                    except ValueError:
                        pass
                motion = torch.tensor(sub_motion_data, dtype=torch.float32)
                if motion.shape[0] < min_length:
                    continue 

                if step is not None:
                    data = motion[1::step].clone()
                else:
                    data = motion[1::4].clone()

                rotation_euler = torch.reshape(data[..., 3:], (data.shape[0], -1, 3))
                rotation_matrix = euler_angles_to_matrix(torch.deg2rad(rotation_euler), "ZYX")
                q = matrix_to_quaternion(rotation_matrix)

                # _xzy_to_xyz = torch.sqrt(torch.tensor([[2, 2, 0, 0]], dtype=torch.float32, device=data_device)) / 2
                # _xzy_to_xyz = _xzy_to_xyz.expand(q[:, 0].shape)
                # q[:, 0] = self.skeleton._qmul(_xzy_to_xyz, q[:, 0])
                # data[..., :3] = self.skeleton._qrot(data[..., :3], _xzy_to_xyz)

                sign = torch.gt(torch.linalg.vector_norm(q[1:] - q[:-1], dim=-1, keepdim=True),
                                torch.linalg.vector_norm(q[1:] + q[:-1], dim=-1, keepdim=True)).int()
                sign = (torch.cumsum(sign, dim=0) % 2) * -2 + 1
                q = torch.cat([q[:1], q[1:] * sign], dim=0)
                q = torch.reshape(q, (data.shape[0], -1))

                motion = torch.cat([data[..., :3] * self.SCALE, q], dim=-1)

                text_data = []
                with cs.open(pjoin(dataset_path, './texts/', name + '.txt')) as f:
                    for line in f.readlines():
                        if line == '' or line == '\n':
                            continue
                        text_dict = {}
                        # print(line)
                        caption = line

                        text_dict['caption'] = caption

                        text_data.append(text_dict)

                data_dict[name] = {'motion': motion,
                                    'text': text_data}
                name_list.append(name)
        
        self.data_dict = data_dict
        self.name_list = name_list

    
    def __len__(self):
        return len(self.data_dict) - self.pointer


    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]

        motion, text_list = data['motion'], data['text']

        text_data = random.choice(text_list)
        caption = text_data['caption']

        m_length = len(motion)
        if self.step < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'
        if coin2 == 'double':
            m_length = (m_length // self.step - 1) * self.step
        elif coin2 == 'single':
            m_length = (m_length // self.step) * self.step

        start = random.randint(0, motion.shape[0] - m_length)

        motion = motion[start:start + m_length]

        if m_length < self.max_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_length - m_length, motion.shape[1]))
                                     ], axis=0)
            
        motion = torch.from_numpy(motion)

        root_p = motion[..., :3]

        q = torch.reshape(motion[..., 3:].clone(), (*root_p.shape[:1], -1, 4))

        return caption, root_p, q, m_length

def DATALoader(dataset_path,
                split_file,
                min_length,
                max_length,
                step,
                batch_size,
                num_workers=8,
                drop_last=True, 
                shuffle=True, 
                pin_memory=True):
    
    trainSet = UniDataset(dataset_path, split_file, min_length=min_length, max_length=max_length, step=step)
    train_loader = torch.utils.data.DataLoader(trainSet,
                                              batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              drop_last=drop_last,
                                              pin_memory=pin_memory)
    
    return train_loader

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
