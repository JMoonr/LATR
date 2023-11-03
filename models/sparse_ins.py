import math

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill
from .sparse_inst_loss import SparseInstCriterion, SparseInstMatcher

def _make_stack_3x3_convs(num_convs, in_channels, out_channels):
    convs = []
    for _ in range(num_convs):
        convs.append(
            nn.Conv2d(in_channels, out_channels, 3, padding=1))
        convs.append(nn.ReLU(True))
        in_channels = out_channels
    return nn.Sequential(*convs)


class MaskBranch(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        dim = cfg.hidden_dim
        num_convs = cfg.num_convs
        kernel_dim = cfg.kernel_dim
        self.mask_convs = _make_stack_3x3_convs(num_convs, in_channels, dim)
        self.projection = nn.Conv2d(dim, kernel_dim, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.mask_convs.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)
        c2_msra_fill(self.projection)

    def forward(self, features):
        # mask features (x4 convs)
        features = self.mask_convs(features)
        return self.projection(features)


class InstanceBranch(nn.Module):
    def __init__(self, cfg, in_channels, **kwargs):
        super().__init__()
        num_mask = cfg.num_query
        dim = cfg.hidden_dim
        num_classes = cfg.num_classes
        kernel_dim = cfg.kernel_dim
        num_convs = cfg.num_convs
        num_group = cfg.get('num_group', 1)
        sparse_num_group = cfg.get('sparse_num_group', 1)
        self.num_group = num_group
        self.sparse_num_group = sparse_num_group
        self.num_mask = num_mask
        self.inst_convs = _make_stack_3x3_convs(
                            num_convs=num_convs, 
                            in_channels=in_channels, 
                            out_channels=dim)

        self.iam_conv = nn.Conv2d(
            dim * num_group,
            num_group * num_mask * sparse_num_group,
            3, padding=1, groups=num_group * sparse_num_group)
        self.fc = nn.Linear(dim * sparse_num_group, dim)
        # output
        self.mask_kernel = nn.Linear(
            dim, kernel_dim)
        self.cls_score = nn.Linear(
            dim, num_classes)
        self.objectness = nn.Linear(
            dim, 1)
        self.prior_prob = 0.01
        self._init_weights()

    def _init_weights(self):
        for m in self.inst_convs.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        for module in [self.iam_conv, self.cls_score]:
            init.constant_(module.bias, bias_value)
        init.normal_(self.iam_conv.weight, std=0.01)
        init.normal_(self.cls_score.weight, std=0.01)

        init.normal_(self.mask_kernel.weight, std=0.01)
        init.constant_(self.mask_kernel.bias, 0.0)
        c2_xavier_fill(self.fc)

    def forward(self, seg_features, is_training=True):
        out = {}
        # SparseInst part
        seg_features = self.inst_convs(seg_features)
        # predict instance activation maps
        iam = self.iam_conv(seg_features.tile(
            (1, self.num_group, 1, 1)))
        if not is_training:
            iam = iam.view(
                iam.shape[0],
                self.num_group,
                self.num_mask * self.sparse_num_group,
                *iam.shape[-2:])
            iam = iam[:, 0, ...]
            num_group = 1
        else:
            num_group = self.num_group

        iam_prob = iam.sigmoid()
        B, N = iam_prob.shape[:2]
        C = seg_features.size(1)
        # BxNxHxW -> BxNx(HW)
        iam_prob = iam_prob.view(B, N, -1)
        normalizer = iam_prob.sum(-1).clamp(min=1e-6)
        iam_prob_norm_hw = iam_prob / normalizer[:, :, None]

        # aggregate features: BxCxHxW -> Bx(HW)xC
        # (B x N x HW) @ (B x HW x C) -> B x N x C
        all_inst_features = torch.bmm(
            iam_prob_norm_hw,
            seg_features.view(B, C, -1).permute(0, 2, 1)) #BxNxC

        # concat sparse group features
        inst_features = all_inst_features.reshape(
            B, num_group,
            self.sparse_num_group,
            self.num_mask, -1
        ).permute(0, 1, 3, 2, 4).reshape(
            B, num_group,
            self.num_mask, -1)
        inst_features = F.relu_(
            self.fc(inst_features))

        # avg over sparse group
        iam_prob = iam_prob.view(
            B, num_group,
            self.sparse_num_group,
            self.num_mask,
            iam_prob.shape[-1])
        iam_prob = iam_prob.mean(dim=2).flatten(1, 2)
        inst_features = inst_features.flatten(1, 2)
        out.update(dict(
            iam_prob=iam_prob,
            inst_features=inst_features))

        if self.training:
            pred_logits = self.cls_score(inst_features)
            pred_kernel = self.mask_kernel(inst_features)
            pred_scores = self.objectness(inst_features)
            out.update(dict(
                pred_logits=pred_logits,
                pred_kernel=pred_kernel,
                pred_scores=pred_scores))
        return out

class SparseInsDecoder(nn.Module):
    def __init__(self, cfg, **kargs) -> None:
        super().__init__()
        in_channels = cfg.encoder.out_dims + 2
        self.output_iam = cfg.decoder.output_iam
        self.scale_factor = cfg.decoder.scale_factor
        self.sparse_decoder_weight = cfg.sparse_decoder_weight
        self.inst_branch = InstanceBranch(cfg.decoder, in_channels)
        # dim, num_convs, kernel_dim, in_channels
        self.mask_branch = MaskBranch(cfg.decoder, in_channels)
        self.sparse_inst_crit = SparseInstCriterion(
            num_classes=cfg.decoder.num_classes,
            matcher=SparseInstMatcher(),
            cfg=cfg)
        self._init_weights()

    def _init_weights(self):
        self.inst_branch._init_weights()
        self.mask_branch._init_weights()

    @torch.no_grad()
    def compute_coordinates(self, x):
        h, w = x.size(2), x.size(3)
        y_loc = -1.0 + 2.0 * torch.arange(h, device=x.device) / (h - 1)
        x_loc = -1.0 + 2.0 * torch.arange(w, device=x.device) / (w - 1)
        y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
        y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
        x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
        locations = torch.cat([x_loc, y_loc], 1)
        return locations.to(x)

    def forward(self, features, is_training=True, **kwargs):
        output = {}
        coord_features = self.compute_coordinates(features)
        features = torch.cat([coord_features, features], dim=1)
        inst_output = self.inst_branch(
            features, is_training=is_training)
        output.update(inst_output)

        if is_training:
            mask_features = self.mask_branch(features)
            pred_kernel = inst_output['pred_kernel']
            N = pred_kernel.shape[1]
            B, C, H, W = mask_features.shape

            pred_masks = torch.bmm(pred_kernel, mask_features.view(
            B, C, H * W)).view(B, N, H, W)
            pred_masks = F.interpolate(
                pred_masks, scale_factor=self.scale_factor,
                mode='bilinear', align_corners=False)
            output.update(dict(
                pred_masks=pred_masks))
        
        if self.training:
            sparse_inst_losses, matched_indices = self.loss(
                    output,
                    lane_idx_map=kwargs.get('lane_idx_map'),
                    input_shape=kwargs.get('input_shape')
            )
            for k, v in sparse_inst_losses.items():
                sparse_inst_losses[k] = self.sparse_decoder_weight * v
            output.update(sparse_inst_losses)
            output['matched_indices'] = matched_indices
        return output

    def loss(self, output, lane_idx_map, input_shape):
        """
        output : from self.forward
        lane_idx_map : instance-level segmentation map, [20, H, W] where 20=max_lanes
        """
        pred_masks = output['pred_masks']
        pred_masks = output['pred_masks'].view(
            pred_masks.shape[0],
            self.inst_branch.num_group,
            self.inst_branch.num_mask,
            *pred_masks.shape[2:])
        pred_logits = output['pred_logits']
        pred_logits = output['pred_logits'].view(
            pred_logits.shape[0],
            self.inst_branch.num_group,
            self.inst_branch.num_mask,
            *pred_logits.shape[2:])
        pred_scores = output['pred_scores']
        pred_scores = output['pred_scores'].view(
            pred_scores.shape[0],
            self.inst_branch.num_group,
            self.inst_branch.num_mask,
            *pred_scores.shape[2:])

        out = {}
        all_matched_indices = []
        for group_idx in range(self.inst_branch.num_group):
            sparse_inst_losses, matched_indices = \
                self.sparse_inst_crit(
                    outputs=dict(
                        pred_masks=pred_masks[:, group_idx, ...].contiguous(),
                        pred_logits=pred_logits[:, group_idx, ...].contiguous(),
                        pred_scores=pred_scores[:, group_idx, ...].contiguous(),
                    ),
                    targets=self.prepare_targets(lane_idx_map),
                    input_shape=input_shape, # seg_bev
                )
            for k, v in sparse_inst_losses.items():
                out['%s_%d' % (k, group_idx)] = v
            all_matched_indices.append(matched_indices)
        return out, all_matched_indices

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            target = {}
            cls_labels = targets_per_image.flatten(-2).max(-1)[0]
            pos_mask = cls_labels > 0

            target["labels"] = cls_labels[pos_mask].long()
            target["masks"] = targets_per_image[pos_mask] > 0
            new_targets.append(target)
        return new_targets
        return output
