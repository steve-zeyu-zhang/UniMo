import numpy as np
import pickle
from ylib.skeleton_prior import smal_35_parents as parents
from ylib.skeleton_prior import smal_35_kinematic_chain
from ylib.maa import convert_maa_motion_to_positions
import numpy as np
import torch
from Motion.transforms import quat2euler, repr6d2quat
from pytorch3d.transforms import matrix_to_euler_angles
from Motion.Animation import positions_global
import os
from ylib.rotation_conversions import rotation_6d_to_matrix, matrix_to_euler_angles

SMAL_NAME2ID35 = {
    'root':       0,
    'pelvis0':    1,
    'spine':      2,
    'spine0':     3,
    'spine1':     4,
    'spine2':     5,
    'spine3':     6,
    'Left_Leg1':      7,
    'Left_Leg2':      8,
    'Left_Leg3':      9,
    'Left_Foot':     10,
    'Right_Leg1':     11,
    'Right_Leg2':     12,
    'Right_Leg3':     13,
    'Right_Foot':     14,
    'Neck':      15,
    'Head':      16,
    'Left_LegBack1': 17,
    'Left_LegBack2': 18,
    'Left_LegBack3': 19,
    'Left_FootBack': 20,
    'Right_LegBack1': 21,
    'Right_LegBack2': 22,
    'Right_LegBack3': 23,
    'Right_FootBack': 24,
    'Tail1':     25,
    'Tail2':     26,
    'Tail3':     27,
    'Tail4':     28,
    'Tail5':     29,
    'Tail6':     30,
    'Tail7':     31,
    'Mouth':     32,
    'Left_Ear':      33,
    'Right_Ear':      34,
}
SMAL_NAMES = [k for k in SMAL_NAME2ID35.keys()]

# rotation with shape frame * J * 3
def write_bvh(parent, offset, rotation, position, names, frametime, order, path, endsite=None):
    file = open(path, 'w')
    frame = len(rotation)
    joint_num = len(rotation[0])
    order = order.upper()

    file_string = 'HIERARCHY\n'

    seq = []

    def write_static(idx, prefix):
        nonlocal parent, offset, rotation, names, order, endsite, file_string, seq
        seq.append(idx)
        if idx == 0:
            name_label = 'ROOT ' + names[idx]
            channel_label = 'CHANNELS 6 Xposition Yposition Zposition {}rotation {}rotation {}rotation'.format(*order)
        else:
            name_label = 'JOINT ' + names[idx]
            channel_label = 'CHANNELS 3 {}rotation {}rotation {}rotation'.format(*order)
        offset_label = 'OFFSET %.6f %.6f %.6f' % (offset[idx][0], offset[idx][1], offset[idx][2])

        file_string += prefix + name_label + '\n'
        file_string += prefix + '{\n'
        file_string += prefix + '\t' + offset_label + '\n'
        file_string += prefix + '\t' + channel_label + '\n'

        has_child = False
        for y in range(idx+1, joint_num):
            if parent[y] == idx:
                has_child = True
                write_static(y, prefix + '\t')
        if not has_child:
            file_string += prefix + '\t' + 'End Site\n'
            file_string += prefix + '\t' + '{\n'
            file_string += prefix + '\t\t' + 'OFFSET 0 0 0\n'
            file_string += prefix + '\t' + '}\n'

        file_string += prefix + '}\n'

    write_static(0, '')

    file_string += 'MOTION\n' + 'Frames: {}\n'.format(frame) + 'Frame Time: %.8f\n' % frametime
    for i in range(frame):
        file_string += '%.6f %.6f %.6f ' % (position[i][0], position[i][1], position[i][2])
        for j in range(joint_num):
            idx = seq[j]
            file_string += '%.6f %.6f %.6f ' % (rotation[i][idx][0], rotation[i][idx][1], rotation[i][idx][2])
        file_string += '\n'

    file.write(file_string)
    return file_string


class WriterWrapper:
    def __init__(self, parents, offset=None):
        self.parents = parents
        self.offset = offset
        self.n_joints = offset.shape[0]
        self.joint_hierarchy = parents

    def write(self, filename, rot, pos, offset=None, names=None, repr='quat'):
        """
        Write animation to bvh file
        :param filename:
        :param rot: Quaternion as (w, x, y, z)
        :param pos:
        :param offset:
        :return:
        """
        if repr not in ['euler', 'quat', 'quaternion', 'repr6d']:
            raise Exception('Unknown rotation representation')
        if offset is None:
            offset = self.offset
        if not isinstance(offset, torch.Tensor):
            offset = torch.tensor(offset)
        n_bone = offset.shape[0]

        if repr == 'repr6d':
            rot = rot.reshape(rot.shape[0], -1, 6)
            
        if repr == 'repr6d' or repr == 'quat' or repr == 'quaternion':
            rot = rotation_6d_to_matrix(rot)
            # rot = rot.reshape(rot.shape[0], -1, 4)
            # rot = rot / (rot.norm(dim=-1, keepdim=True) ** 0.5)
            # 转换为 Euler 角，单位为度，顺序为 'xyz'
            euler = matrix_to_euler_angles(rot, order='xyz')
            # --- 在这里对全局旋转进行处理 ---
            # 假设根节点对应的 joint index 为 0，即 euler[:, 0, :]
            # 对根节点的 Z 轴旋转角加 90°
            euler[:, 0, 1] += 90.0
            # -----------------------------------------
            rot = euler

            # rot = torch.cat([-euler[..., 1].unsqueeze(dim=-1), euler[..., 0].unsqueeze(dim=-1), euler[..., 2].unsqueeze(dim=-1)], dim=-1)
            # local_rotations = []
            # for frame_idx in range(global_rotations.shape[0]):
            #     frame_rot = {}
            #     for joint_idx in range(self.n_joints):
            #         parent_idx = self.joint_hierarchy[joint_idx]

            #         q_global = torch.tensor(global_rotations[frame_idx, joint_idx], dtype=torch.float32)

            #         if parent_idx == -1:
            #             local_rot = q_global
            #         else:
            #             q_parent_inv = torch.tensor([
            #                 global_rotations[frame_idx, parent_idx][0],
            #                 -global_rotations[frame_idx, parent_idx][1],
            #                 -global_rotations[frame_idx, parent_idx][2],
            #                 -global_rotations[frame_idx, parent_idx][3]
            #             ])

            #             # q_parent^{-1} * q_global
            #             local_rot = self._qmul(q_parent_inv, q_global)

            #         frame_rot[joint_idx] = local_rot.numpy()
            #     local_rotations.append(frame_rot)

            # local_rotations = rot
            # euler_angles = []
            # for frame_idx in range(rot.shape[0]):
            #     joint_euler = []
            #     for joint_idx in range(self.n_joints):
            #         q = torch.tensor(local_rotations[frame_idx][joint_idx])
            #         q_normalized = q / torch.norm(q)

            #         matrix = torch.zeros(3, 3)
            #         w, x, y, z = q_normalized
            #         theta = np.deg2rad(90)
            #         matrix[0, 0] = 1 - 2 * y ** 2 - 2 * z ** 2 
            #         matrix[0, 1] = 2 * x * y - 2 * z * w 
            #         matrix[0, 2] = 2 * x * z + 2 * y * w
            #         matrix[1, 0] = 2 * x * y + 2 * z * w 
            #         matrix[1, 1] = 1 - 2 * x ** 2 - 2 * z ** 2
            #         matrix[1, 2] = 2 * y * z - 2 * x * w
            #         matrix[2, 0] = 2 * x * z - 2 * y * w
            #         matrix[2, 1] = 2 * y * z + 2 * x * w
            #         matrix[2, 2] = 1 - 2 * x ** 2 - 2 * y ** 2 + 1

            #         Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
            #                                         [np.sin(theta),  np.cos(theta), 0],
            #                                         [0,              0,             1]])

            #         euler = torch.rad2deg(matrix_to_euler_angles(matrix, "XYZ"))

            #         joint_euler.append(Rz@euler.numpy())
            #     euler_angles.append(np.array(joint_euler))
            # rot = euler_angles

        if names is None:
            names = ['%02d' % i for i in range(n_bone)]
        write_bvh(self.parents, offset, rot, pos, names, 1, 'xyz', filename)



    def _qmul(self, u, v):
        # u, v: [..., 4], same dim, real first
        u = u.to(torch.float)
        v = v.to(torch.float)
        original_shape = u.shape
        terms = torch.bmm(torch.reshape(v, (-1, 4, 1)), torch.reshape(u, (-1, 1, 4)))

        w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
        x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
        y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
        z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
        return torch.reshape(torch.stack((w, x, y, z), dim=1), original_shape).contiguous()
    
def vis(file_name):
    dataset = file_name.split('_')[0]
    motion = np.load(f"dataset/animals_maa_motions/{file_name}")
    outfile = os.path.join('output/vis-tmp', f"{file_name.replace('.npy', '.bvh')}")
    with open(f"dataset/animals_maa_offsets/{dataset}/offset.pkl", 'rb') as f:
        offset = pickle.load(f)

    pos = convert_maa_motion_to_positions(motion, offset, smal_35_kinematic_chain)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)

    if isinstance(motion, np.ndarray):
        motion = torch.from_numpy(motion)

    rot = motion[:, :-1]

    WriterWrapper(parents, offset).write(outfile, rot, pos, offset, names=SMAL_NAMES, repr='repr6d')

def get_all_files(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_list.append(file)
    return file_list

in_dir = 'dataset/animals_maa_motions'
bvh_files = get_all_files(in_dir)

num = 0
for fn in bvh_files:
    
    vis(fn)
    num += 1
    if num >= 5:
        break