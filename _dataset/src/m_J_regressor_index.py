import os
import pickle as pkl
import sys
from argparse import ArgumentParser

import torch


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BASE_DIR)
BASE_DIR = os.path.dirname(BASE_DIR)
#PROJECT_ROOT_DIR = os.path.dirname(BASE_DIR)
PROJECT_ROOT_DIR=BASE_DIR
print(BASE_DIR)
sys.path.append(PROJECT_ROOT_DIR)

from _dataset.src.utils import farthest_point_sample

def add_options(parser):
    group = parser.add_argument_group('options')
    
    # dataset
    group.add_argument("--dataset", default="humanact12",choices=[ "uestc","4Dcompelte", "humanact12","cmu"], help="Dataset to load")
    group.add_argument("--num_points", default=1024,type=int, help="Dataset to load")


parser = ArgumentParser()
add_options(parser)
args = parser.parse_args(args=[])

SMPL_DATA_PATH = "/root/autodl-tmp/UniMo/_dataset/data/models"
SMPL_MODEL_PATH = os.path.join(SMPL_DATA_PATH, "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl")

with open(SMPL_MODEL_PATH, 'rb') as f:
    params = pkl.load(f,encoding='latin1')
    J_regressor = params['J_regressor']
    v_template = params['v_template']

v_template=torch.tensor(v_template).cuda()   #[6890,3]
J_regressor = torch.tensor(J_regressor.toarray()).cuda()  #[24,6890]

nozero=torch.nonzero(J_regressor)

newindex_to_oldindex=dict()
oldindex_to_newindex=dict()
new_index=0
m_J_regressor=torch.zeros([24,232])
old_index_tensor=torch.zeros([232]).to(torch.long)
for i in range(len(nozero)):
    old_index=nozero[i][1].item()
    row=nozero[i,0]
    value = J_regressor[nozero[i,0],nozero[i,1]]
    #print(value)
    if(old_index not in newindex_to_oldindex.values()):
        newindex_to_oldindex[new_index]=old_index
        oldindex_to_newindex[old_index]=new_index
        m_J_regressor[row,new_index]=value
        old_index_tensor[new_index]=old_index
        new_index+=1
    else:
        temp_new_index=oldindex_to_newindex[old_index]
        m_J_regressor[row,temp_new_index]=value
        # print(old_index)
        # print(temp_new_index)
    
#m_J_regressor_points= v_template[old_index_tensor,:]  #[24,232]
    
all_points_index=torch.ones([6890])
all_points_index[old_index_tensor]=0
other_index=torch.nonzero(all_points_index).squeeze(1)
other_points= v_template[other_index,:].unsqueeze(0).to(torch.float).cuda()

sample_points_num=args.num_points-232

point_index=farthest_point_sample(other_points,sample_points_num)
other_index = other_index.cuda()  # 让 other_index 在 GPU 上
point_index = point_index.cuda()  # 确保 point_index 也在 GPU 上
point_index = other_index[point_index[0].cuda()].unsqueeze(0)  # 确保索引时都在 GPU

# print(other_index.shape)
# print(other_points.shape)
# print(point_index)
#print(m_J_regressor.shape)

m_J_regressor=torch.cat([m_J_regressor.cuda(),torch.zeros(24,sample_points_num).cuda()],dim=1)
old_index_tensor=old_index_tensor.unsqueeze(0).cuda()
point_index=torch.cat([old_index_tensor,point_index],dim=1)


print(m_J_regressor.shape)
# print(old_index_tensor.shape)
print(point_index.shape)


# 定义保存路径
SAVE_DIR = "/root/autodl-tmp/UniMo/_dataset/data/smpl_1024vertices_fps_index"
# 确保保存路径存在
os.makedirs(SAVE_DIR, exist_ok=True)

# 保存 point_index
with open(os.path.join(SAVE_DIR, "_m_J_regressor_1024_point_index.pkl"), 'wb') as f:
    pkl.dump(point_index, f)

# 保存 m_J_regressor
with open(os.path.join(SAVE_DIR, "_m_1024_J_regressor.pkl"), 'wb') as f:
    pkl.dump(m_J_regressor, f)