from os.path import join as pjoin
import torch
from torch.utils import data
from torch.utils.data._utils.collate import default_collate
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_quaternion
import random
import codecs as cs
import glob
from data.skeleton import Skeleton
from data.bvh import parse_bvh_skeleton

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

def setup_skeleton(skeleton_file, device):
    with open(skeleton_file, "r") as f:
        joint_names, joint_offsets, joint_hierarchy, end_sites = parse_bvh_skeleton(f.read())

    children = [list() for _ in range(len(joint_names))]
    for i, parent_idx in enumerate(joint_hierarchy):
        if parent_idx >= 0:
            children[parent_idx].append(i)

    joint_tails = []

    joint_bgroups = torch.zeros(len(joint_names), 5, dtype=torch.float32, device=device)
    bgroups_spine = ["Hips", "LHipJoint", "RHipJoint", "LowerBack", "Spine", "Spine1", "Neck", "Neck1", "Head"]
    bgroups_leftarm = ["LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftFingerBase", "LeftHandIndex1"]
    bgroups_rightarm = ["RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightFingerBase", "RightHandIndex1"]
    bgroups_leftleg = ["LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase"]
    bgroups_rightleg = ["RightUpLeg", "RightLeg", "RightFoot", "RightToeBase"]

    end_joints = ["LeftHand", "RightHand", "LeftToeBase", "RightToeBase"]

    for i, name in enumerate(joint_names):
        if len(children[i]) == 0:
            joint_tails.append(end_sites[name])
        elif name == "Hips":
            joint_tails.append(joint_offsets[children[i][-1]])
        else:
            joint_tails.append(joint_offsets[children[i][0]])

        if name in bgroups_spine:
            joint_bgroups[i, 0] = 1
        elif name in bgroups_leftarm:
            joint_bgroups[i, 1] = 1
        elif name in bgroups_rightarm:
            joint_bgroups[i, 2] = 1
        elif name in bgroups_leftleg:
            joint_bgroups[i, 3] = 1
        elif name in bgroups_rightleg:
            joint_bgroups[i, 4] = 1

    pairs = []
    for i in range(len(bgroups_leftarm)):
        pairs.append((joint_names.index(bgroups_leftarm[i]), joint_names.index(bgroups_rightarm[i])))

    for i in range(len(bgroups_leftleg)):
        pairs.append((joint_names.index(bgroups_leftleg[i]), joint_names.index(bgroups_rightleg[i])))

    joint_offsets = torch.tensor(joint_offsets, dtype=torch.float32, device=device)
    joint_tails = torch.tensor(joint_tails, dtype=torch.float32, device=device)

    return Skeleton(joint_names, joint_hierarchy, joint_offsets, joint_tails, joint_bgroups, end_joints, pairs,
                    scale=0.056444, device=device,rotation_order="ZYX")
                        
class CmuDataset(data.Dataset):
    def __init__(self, data_folder, min_length=0, step=1):

        self.data_folder = data_folder
        self.src_n_points = 256
        self.length = min_length
        self.SCALE = 0.056444
        self.std_cloud = 0.05

        raw_data = self.load_files()

        motion_data = [self.process_data(d) for d in raw_data]

        self.min_length = min_length
        self.motion_data = []
        for data in motion_data:
            if data.shape[0] // step >= min_length:
                self.motion_data.append(data)


        files = glob.glob(self.data_folder + "/**/*.bvh", recursive=True)
        fps60 = ["33_", "34_", "42_", "60_", "61_", "62_", "74_", "75_", "77_", "79_", "80_", "87_", "88_", "89_"]

        # quaternion conversion
        for i in range(len(self.motion_data)):
            if step is not None:
                data = self.motion_data[i][1::step].clone()
            elif any(s in files[i] for s in fps60):
                data = self.motion_data[i][1::2].clone()
            else:
                data = self.motion_data[i][1::4].clone()
            # rotation_euler = torch.flip(torch.reshape(data[..., 3:], (data.shape[0], -1, 3)), [2])
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

            self.motion_data[i] = torch.cat([data[..., :3] * self.SCALE, q], dim=-1)

    def load_files(self):
        raw_data = []

        for file in glob.glob(self.data_folder + "/**/*.bvh", recursive=True):
            with open(file, "r") as f:
                while f.readline() != "MOTION\n":
                    continue
                raw_data.append(f.read())

        return raw_data

    def process_data(self, data):
        lines = data.split('\n')
        motion_data = []
        for line in lines[2:]:
            words = line.strip().split(" ")
            try:
                motion_data.append(list(map(float, words)))
            except ValueError:
                pass

        return torch.tensor(motion_data, dtype=torch.float32)

    def __len__(self):
        return len(self.motion_data)

    def __getitem__(self, item):

        motion = self.motion_data[item]

        start = random.randint(0, motion.shape[0] - self.length)

        motion = motion[start:start + self.length]

        root_p = motion[..., :3]

        q = torch.reshape(motion[..., 3:].clone(), (*root_p.shape[:1], -1, 4))

        return root_p, q
