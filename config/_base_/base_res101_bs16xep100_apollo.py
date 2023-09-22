import os
import os.path as osp
import numpy as np

# ========DATA SETTING======== #
dataset_name = 'apollo'
dataset = 'standard'

data_dir = osp.join('./data/apollosyn_gen-lanenet/data_splits', dataset)
dataset_dir = './data/apollosyn_gen-lanenet/Apollo_Sim_3D_Lane_Release'

output_dir = 'apollo'

rewrite_pred = True
save_best = False

output_dir = dataset_name

org_h = 1080
org_w = 1920
crop_y = 0

cam_height = 1.55
pitch = 3
fix_cam = False
pred_cam = False

model_name = 'LATR'
mod = None

ipm_h = 208
ipm_w = 128
resize_h = 360
resize_w = 480

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

K = np.array([[2015., 0., 960.],
            [0., 2015., 540.],
            [0., 0., 1.]])

position_embedding = 'learned'

max_lanes = 6
num_category = 2
prob_th = 0.5
num_class = 2 # 1 bgd | 1 lanes

batch_size = 16
nepochs = 210
nworkers = 16

# ddp setting
dist = True
sync_bn = True
cudnn = True

distributed = True
local_rank = None #TODO
gpu = 0
world_size = 1
nodes = 1

# for reload ckpt
eval_ckpt = ''
resume = '' # ckpt number as input
resume_from = '' # ckpt path as input

no_cuda = False

# tensorboard
no_tb = False

start_epoch = 0
channels_in = 3

# args input
test_mode = False # 'store_true' # TODO 
evaluate = False # TODO
evaluate_case = ''

# print & save
print_freq = 50
save_freq = 50
eval_freq = 20 # eval freq during training

# top view
top_view_region = np.array([[-10, 103], [10, 103], [-10, 3], [10, 3]])
anchor_y_steps = np.linspace(3, 103, 25)
num_y_steps = len(anchor_y_steps)

save_path = None
save_json_path = None

