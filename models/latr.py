import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import *
from mmdet3d.models import build_backbone, build_neck
from .latr_head import LATRHead
from mmcv.utils import Config
from .ms2one import build_ms2one
from .utils import deepFeatureExtractor_EfficientNet

from mmdet.models.builder import BACKBONES


# overall network
class LATR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.no_cuda = args.no_cuda
        self.batch_size = args.batch_size
        self.num_lane_type = 1  # no centerline
        self.num_y_steps = args.num_y_steps
        self.max_lanes = args.max_lanes
        self.num_category = args.num_category
        _dim_ = args.latr_cfg.fpn_dim
        num_query = args.latr_cfg.num_query
        num_group = args.latr_cfg.num_group
        sparse_num_group = args.latr_cfg.sparse_num_group

        self.encoder = build_backbone(args.latr_cfg.encoder)
        if getattr(args.latr_cfg, 'neck', None):
            self.neck = build_neck(args.latr_cfg.neck)
        else:
            self.neck = None
        self.encoder.init_weights()
        self.ms2one = build_ms2one(args.ms2one)

        # build 2d query-based instance seg
        self.head = LATRHead(
            args=args,
            dim=_dim_,
            num_group=num_group,
            num_convs=4,
            in_channels=_dim_,
            kernel_dim=_dim_,
            position_range=args.position_range,
            top_view_region=args.top_view_region,
            positional_encoding=dict(
                type='SinePositionalEncoding',
                num_feats=_dim_// 2, normalize=True),
            num_query=num_query,
            pred_dim=self.num_y_steps,
            num_classes=args.num_category,
            embed_dims=_dim_,
            transformer=args.transformer,
            sparse_ins_decoder=args.sparse_ins_decoder,
            **args.latr_cfg.get('head', {}),
            trans_params=args.latr_cfg.get('trans_params', {})
        )

    def forward(self, image, _M_inv=None, is_training=True, extra_dict=None):
        out_featList = self.encoder(image)
        neck_out = self.neck(out_featList)
        neck_out = self.ms2one(neck_out)

        output = self.head(
            dict(
                x=neck_out,
                lane_idx=extra_dict['seg_idx_label'],
                seg=extra_dict['seg_label'],
                lidar2img=extra_dict['lidar2img'],
                pad_shape=extra_dict['pad_shape'],
                ground_lanes=extra_dict['ground_lanes'] if is_training else None,
                ground_lanes_dense=extra_dict['ground_lanes_dense'] if is_training else None,
                image=image,
            ),
            is_training=is_training,
        )
        return output