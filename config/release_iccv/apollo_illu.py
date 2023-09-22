import numpy as np
from mmcv.utils import Config
import os.path as osp

_base_ = [
    '../_base_/base_res101_bs16xep100_apollo.py',
    '../_base_/optimizer.py',
]

mod = 'release_iccv/apollo_illu'
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


dataset_name = 'apollo'
dataset = 'illus_chg'
data_dir = osp.join('./3D_Lane_Synthetic_Dataset/data_splits', dataset)
dataset_dir = './data/Apollo_Sim_3D_Lane_Release'
output_dir = 'apollo'
num_category = 2
max_lanes = 6

h_org, w_org = 1080, 1920

batch_size = 8
nworkers = 10
pos_threshold = 0.3
top_view_region = np.array([
    [-10, 103], [10, 103], [-10, 3], [10, 3]])
enlarge_length = 20
position_range = [
    top_view_region[0][0] - enlarge_length,
    top_view_region[2][1] - enlarge_length,
    -5,
    top_view_region[1][0] + enlarge_length,
    top_view_region[0][1] + enlarge_length,
    5.]
anchor_y_steps = np.linspace(3, 103, 20)
num_y_steps = len(anchor_y_steps)

_dim_ = 256
num_query = 12
num_pt_per_line = 20
latr_cfg = dict(
    fpn_dim = _dim_,
    num_query = num_query,
    num_group = 1,
    sparse_num_group = 4,
    encoder = dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    neck = dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True
    ),
    head=dict(
        pt_as_query=True,
        num_pt_per_line=num_pt_per_line,
    ),
    trans_params=dict(init_z=0, bev_h=150, bev_w=70),
)

ms2one=dict(
    type='DilateNaive',
    inc=_dim_, outc=_dim_, num_scales=4,
    dilations=(1, 2, 5, 9))

transformer=dict(
    type='LATRTransformer',
    decoder=dict(
        type='LATRTransformerDecoder',
        embed_dims=_dim_,
        num_layers=6,
        enlarge_length=enlarge_length,
        M_decay_ratio=1,
        num_query=num_query,
        num_anchor_per_query=num_pt_per_line,
        anchor_y_steps=anchor_y_steps,
        transformerlayers=dict(
            type='LATRDecoderLayer',
            attn_cfgs=[
                dict(
                    type='MultiheadAttention',
                    embed_dims=_dim_,
                    num_heads=4,
                    dropout=0.1),
                dict(
                    type='MSDeformableAttention3D',
                    embed_dims=_dim_,
                    num_heads=4,
                    num_levels=1,
                    num_points=8,
                    batch_first=False,
                    num_query=num_query,
                    num_anchor_per_query=num_pt_per_line,
                    anchor_y_steps=anchor_y_steps,
                    dropout=0.1),
                ],
            ffn_cfgs=dict(
                type='FFN',
                embed_dims=_dim_,
                feedforward_channels=_dim_*8,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True),
            ),
            feedforward_channels=_dim_ * 8,
            operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                            'ffn', 'norm')),
))

sparse_ins_decoder=Config(
    dict(
        encoder=dict(
            out_dims=_dim_),
        decoder=dict(
            num_query=latr_cfg['num_query'],
            num_group=latr_cfg['num_group'],
            sparse_num_group=latr_cfg['sparse_num_group'],
            hidden_dim=_dim_,
            kernel_dim=_dim_,
            num_classes=num_category,
            num_convs=4,
            output_iam=True,
            scale_factor=1.,
        ),
        sparse_decoder_weight=5.0,
))

resize_h = 720
resize_w = 960