import os
import os.path as osp
import numpy as np


dataset_name = 'openlane'
dataset = '300' # '300' | '1000'

#  The path of dataset json files (annotations)
data_dir = './data/openlane/lane3d_300/'
# The path of dataset image files (images)
dataset_dir = './data/openlane/images/'
output_dir = dataset_name

org_h = 1280
org_w = 1920
crop_y = 0

ipm_h = 208
ipm_w = 128
resize_h = 360
resize_w = 480

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

cam_height = 1.55
pitch = 3
fix_cam = False
pred_cam = False

model_name = 'LATR'
weight_init = 'normal'
mod = None

position_embedding = 'learned'
max_lanes = 20
num_category = 21
prob_th = 0.5
num_class = 21 # 1 bgd | 1 lanes

# top view
top_view_region = np.array([[-10, 103], [10, 103], [-10, 3], [10, 3]])
anchor_y_steps = np.linspace(3, 103, 25)
num_y_steps = len(anchor_y_steps)

# placeholder, not used
K = np.array([[1000., 0., 960.],
            [0., 1000., 640.],
            [0., 0., 1.]])

# persformer anchor
use_default_anchor = False

batch_size = 16
nepochs = 100

no_cuda = False
nworkers = 16

start_epoch = 0
channels_in = 3

# args input
test_mode = False # 'store_true' # TODO 
evaluate = False # TODO
resume = '' # resume latest saved run.

# tensorboard
no_tb = False

# print & save
print_freq = 50
save_freq = 50

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
resume_from = ''
output_dir = 'openlane'
evaluate_case = ''
eval_freq = 8 # eval freq during training

save_json_path = None
save_root = 'work_dirs'
save_prefix = osp.join(os.getcwd(), save_root)
save_path = osp.join(save_prefix, output_dir)