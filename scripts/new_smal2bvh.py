import numpy as np
import pickle
from ylib.skeleton_prior import smal_35_parents as parents
from ylib.skeleton_prior import smal_35_kinematic_chain
from ylib.maa import convert_maa_motion_to_positions
import numpy as np
import torch
from Motion.transforms import quat2euler, repr6d2quat
from Motion.Animation import positions_global
import os
from glob import glob
import json
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

DT4D_TEMPLATE_DIR = 'dataset/animals_smal_template'

def load_scale(animal_name) -> float:
    '''
    load scale from the template
    '''
    json_file = glob(f'{DT4D_TEMPLATE_DIR}/{animal_name}_*.json')

    # make sure that there is only one json file
    assert len(json_file) == 1

    with open(json_file[0], 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    scale = json_data['scale'][0][0][0]

    return scale

# rotation with shape frame * J * 3
def write_bvh(parent, offset, rotation, position, names, frametime, order, path, endsite=None, scale = 1.0):
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
        offset_label = 'OFFSET %.6f %.6f %.6f' % (offset[idx][0]*scale, offset[idx][1]*scale, offset[idx][2]*scale)

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
        file_string += '%.6f %.6f %.6f ' % (position[i][0]*scale, position[i][1]*scale, position[i][2]*scale)
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

    def write(self, filename, rot, pos, offset=None, names=None, repr='quat', scale = 1.0):
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

        from ylib.quaternion import cont6d_to_matrix
        matrix = cont6d_to_matrix(rot)

        euler = torch.rad2deg(matrix_to_euler_angles(matrix, convention="XYZ"))
        euler[:, 0, 1] += 90.0

        rot = euler


        write_bvh(self.parents, offset, rot, pos, names, 1.0 / 20, 'xyz', filename, 1.0)

    
def vis(file_name):
    dataset = file_name.split('_')[0]
    motion = np.load(f"dataset/animals_maa_motions/{file_name}")
    outfile = os.path.join('dataset/AnimalML3D/motions', f"{file_name.replace('.npy', '.bvh')}")
    with open(f"dataset/animals_maa_offsets/{dataset}/offset.pkl", 'rb') as f:
        offset = pickle.load(f)


    

    scale = load_scale(dataset)


    pos = motion[:, -1, :3]

    if isinstance(motion, np.ndarray):
        motion = torch.from_numpy(motion)

    rot = motion[:, :-1]

    WriterWrapper(parents, offset).write(outfile, rot, pos, offset, names=SMAL_NAMES, repr='repr6d', scale=1.0)

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