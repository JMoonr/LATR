# Copyright (c) Tianheng Cheng and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from scipy.optimize import linear_sum_assignment
from fvcore.nn import sigmoid_focal_loss_jit

from typing import Optional, List

import torch
from torch import Tensor
import torch.distributed as dist
import torch.nn.functional as F
import torchvision


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.


@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i]
                                            for img in tensor_list]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


def nested_masks_from_list(tensor_list: List[Tensor], input_shape=None):
    if tensor_list[0].ndim == 3:
        dim_size = sum([img.shape[0] for img in tensor_list])
        if input_shape is None:
            max_size = _max_by_axis([list(img.shape[-2:]) for img in tensor_list])
        else:
            max_size = [input_shape[0], input_shape[1]]
        batch_shape = [dim_size] + max_size
        # b, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.zeros(batch_shape, dtype=torch.bool, device=device)
        idx = 0
        for img in tensor_list:
            c = img.shape[0]
            c_ = idx + c
            tensor[idx: c_, :img.shape[1], : img.shape[2]].copy_(img)
            mask[idx: c_, :img.shape[1], :img.shape[2]] = True
            idx = c_
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def aligned_bilinear(tensor, factor):
    # borrowed from Adelaidet: https://github1s.com/aim-uofa/AdelaiDet/blob/HEAD/adet/utils/comm.py
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )

    return tensor[:, :, :oh - 1, :ow - 1]



def compute_mask_iou(inputs, targets):
    inputs = inputs.sigmoid()
    # thresholding
    binarized_inputs = (inputs >= 0.4).float()
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score


def dice_score(inputs, targets):
    inputs = inputs.sigmoid()
    numerator = 2 * torch.matmul(inputs, targets.t())
    denominator = (
        inputs * inputs).sum(-1)[:, None] + (targets * targets).sum(-1)
    score = numerator / (denominator + 1e-4)
    return score


def dice_loss(inputs, targets, reduction='sum'):
    inputs = inputs.sigmoid()
    assert inputs.shape == targets.shape
    numerator = 2 * (inputs * targets).sum(1)
    denominator = (inputs * inputs).sum(-1) + (targets * targets).sum(-1)
    loss = 1 - (numerator) / (denominator + 1e-4)
    if reduction == 'none':
        return loss
    return loss.sum()


# @SPARSE_INST_CRITERION_REGISTRY.register()
class SparseInstCriterion(nn.Module):
    # This part is partially derivated from: https://github.com/facebookresearch/detr/blob/main/models/detr.py

    def __init__(self, num_classes=4, cfg=None, matcher=None):
        super().__init__()
        self.matcher = matcher
        self.losses = ("labels", "masks") # cfg.MODEL.SPARSE_INST.LOSS.ITEMS
        self.weight_dict = self.get_weight_dict(cfg)
        self.num_classes = num_classes # cfg.MODEL.SPARSE_INST.DECODER.NUM_CLASSES

    def get_weight_dict(self, cfg):
        losses = ("loss_ce", "loss_mask", "loss_dice", "loss_objectness")
        weight_dict = {}

        ce_weight = cfg.get('ce_weight', 2.0)
        mask_weight = cfg.get('mask_weight', 5.0)
        dice_weight = cfg.get('dice_weight', 2.0)
        objectness_weight = cfg.get('objectness_weight', 1.0)
        weight_dict = dict(
            zip(losses, (ce_weight, mask_weight, dice_weight, objectness_weight)))
        return weight_dict

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i)
                              for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i)
                              for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def loss_labels(self, outputs, targets, indices, num_instances, input_shape=None):
        assert "pred_logits" in outputs
        src_logits = outputs['pred_logits']
        target_classes = torch.full(src_logits.shape[:2], 0, # self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        if sum([tmp[0].shape[0] for tmp in indices]) > 0:
            idx = self._get_src_permutation_idx(indices)
            target_classes_o = torch.cat([t["labels"][J]
                                         for t, (_, J) in zip(targets, indices)])
            target_classes[idx] = target_classes_o

        src_logits = src_logits.flatten(0, 1)
        # prepare one_hot target.
        target_classes = target_classes.flatten(0, 1)
        pos_inds = torch.nonzero(
            target_classes != self.num_classes, as_tuple=True)[0]
        labels = torch.zeros_like(src_logits)
        labels[pos_inds, target_classes[pos_inds]] = 1
        # comp focal loss.
        class_loss = sigmoid_focal_loss_jit(
            src_logits,
            labels,
            alpha=0.25,
            gamma=2.0,
            reduction="sum",
        ) / num_instances
        losses = {'loss_ce': class_loss}
        return losses

    def loss_masks_with_iou_objectness(self, outputs, targets, indices, num_instances, input_shape):
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        # Bx100xHxW
        assert "pred_masks" in outputs
        assert "pred_scores" in outputs
        src_iou_scores = outputs["pred_scores"]
        src_masks = outputs["pred_masks"]
        with torch.no_grad():
            target_masks, _ = nested_masks_from_list(
                [t["masks"] for t in targets], input_shape).decompose()
        num_masks = [len(t["masks"]) for t in targets]
        target_masks = target_masks.to(src_masks)
        if len(target_masks) == 0:
            losses = {
                "loss_dice": src_masks.sum() * 0.0,
                "loss_mask": src_masks.sum() * 0.0,
                "loss_objectness": src_iou_scores.sum() * 0.0
            }
            return losses

        src_masks = src_masks[src_idx]
        target_masks = F.interpolate(
            target_masks[:, None], size=src_masks.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)

        src_masks = src_masks.flatten(1)
        # FIXME: tgt_idx
        mix_tgt_idx = torch.zeros_like(tgt_idx[1])
        cum_sum = 0
        for num_mask in num_masks:
            mix_tgt_idx[cum_sum: cum_sum + num_mask] = cum_sum
            cum_sum += num_mask
        mix_tgt_idx += tgt_idx[1]

        target_masks = target_masks[mix_tgt_idx].flatten(1)

        with torch.no_grad():
            ious = compute_mask_iou(src_masks, target_masks)

        tgt_iou_scores = ious
        src_iou_scores = src_iou_scores[src_idx]
        tgt_iou_scores = tgt_iou_scores.flatten(0)
        src_iou_scores = src_iou_scores.flatten(0)

        losses = {
            "loss_objectness": F.binary_cross_entropy_with_logits(src_iou_scores, tgt_iou_scores, reduction='mean'),
            "loss_dice": dice_loss(src_masks, target_masks) / num_instances,
            "loss_mask": F.binary_cross_entropy_with_logits(src_masks, target_masks, reduction='mean')
        }
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_instances, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "masks": self.loss_masks_with_iou_objectness,
        }
        if loss == "loss_objectness":
            # NOTE: loss_objectness will be calculated in `loss_masks_with_iou_objectness`
            return {}
        assert loss in loss_map
        return loss_map[loss](outputs, targets, indices, num_instances, **kwargs)

    def forward(self, outputs, targets, input_shape):

        outputs_without_aux = {k: v for k,
                               v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, input_shape)
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_instances = sum(len(t["labels"]) for t in targets)
        num_instances = torch.as_tensor(
            [num_instances], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_instances)
        num_instances = torch.clamp(
            num_instances / get_world_size(), min=1).item()
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            # try:
            losses.update(self.get_loss(loss, outputs, targets, indices,
                                        num_instances, input_shape=input_shape))
            # except Exception as e:
            #     import pdb; pdb.set_trace()

        for k in losses.keys():
            if k in self.weight_dict:
                losses[k] *= self.weight_dict[k]
        return losses, indices


# @SPARSE_INST_MATCHER_REGISTRY.register()
class SparseInstMatcherV1(nn.Module):

    def __init__(self, cfg=None):
        super().__init__()
        self.alpha = 0.8 # cfg.MODEL.SPARSE_INST.MATCHER.ALPHA
        self.beta = 0.2 # cfg.MODEL.SPARSE_INST.MATCHER.BETA
        self.mask_score = dice_score

    @torch.no_grad()
    def forward(self, outputs, targets, input_shape):
        B, N, H, W = outputs["pred_masks"].shape
        pred_masks = outputs['pred_masks']
        pred_logits = outputs['pred_logits'].sigmoid()

        indices = []

        for i in range(B):
            tgt_ids = targets[i]["labels"]
            # no annotations
            if tgt_ids.shape[0] == 0:
                indices.append((torch.as_tensor([]),
                                torch.as_tensor([])))
                continue

            tgt_masks = targets[i]['masks'].tensor.to(pred_masks)
            pred_logit = pred_logits[i]
            out_masks = pred_masks[i]

            # upsampling:
            # (1) padding/
            # (2) upsampling to 1x input size (input_shape)
            # (3) downsampling to 0.25x input size (output mask size)
            ori_h, ori_w = tgt_masks.size(1), tgt_masks.size(2)
            tgt_masks_ = torch.zeros(
                (1, tgt_masks.size(0), input_shape[0], input_shape[1])).to(pred_masks)
            tgt_masks_[0, :, :ori_h, :ori_w] = tgt_masks
            tgt_masks = F.interpolate(
                tgt_masks_, size=out_masks.shape[-2:], mode='bilinear', align_corners=False)[0]

            # compute dice score and classification score
            tgt_masks = tgt_masks.flatten(1)
            out_masks = out_masks.flatten(1)

            mask_score = self.mask_score(out_masks, tgt_masks)
            # Nx(Number of gts)
            matching_prob = pred_logit[:, tgt_ids]
            C = (mask_score ** self.alpha) * (matching_prob ** self.beta)
            # hungarian matching
            inds = linear_sum_assignment(C.cpu(), maximize=True)
            indices.append(inds)
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


# @SPARSE_INST_MATCHER_REGISTRY.register()
class SparseInstMatcher(nn.Module):

    def __init__(self, cfg=None):
        super().__init__()
        self.alpha = 0.8 # cfg.MODEL.SPARSE_INST.MATCHER.ALPHA
        self.beta = 0.2 # cfg.MODEL.SPARSE_INST.MATCHER.BETA
        self.mask_score = dice_score

    def forward(self, outputs, targets, input_shape):
        with torch.no_grad():
            # B x 40 x 90 x 120 
            B, N, H, W = outputs["pred_masks"].shape
            pred_masks = outputs['pred_masks']
            pred_logits = outputs['pred_logits'].sigmoid()
            tgt_ids = torch.cat([v["labels"] for v in targets])

            if tgt_ids.shape[0] == 0:
                return [(torch.as_tensor([]).to(pred_logits), torch.as_tensor([]).to(pred_logits))] * B
            tgt_masks, _ = nested_masks_from_list(
                [t["masks"] for t in targets], input_shape).decompose()
            device = pred_masks.device
            tgt_masks = tgt_masks.to(pred_masks)

            tgt_masks = F.interpolate(
                tgt_masks[:, None], size=pred_masks.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)

            pred_masks = pred_masks.view(B * N, -1)
            tgt_masks = tgt_masks.flatten(1)
            with autocast(enabled=False):
                pred_masks = pred_masks.float()
                tgt_masks = tgt_masks.float()
                pred_logits = pred_logits.float()
                mask_score = self.mask_score(pred_masks, tgt_masks)
                # Nx(Number of gts)
                matching_prob = pred_logits.view(B * N, -1)[:, tgt_ids]
                C = (mask_score ** self.alpha) * (matching_prob ** self.beta)

            C = C.view(B, N, -1).cpu()
            # hungarian matching
            sizes = [len(v["masks"]) for v in targets]
            indices = [linear_sum_assignment(c[i], maximize=True)
                       for i, c in enumerate(C.split(sizes, -1))]
            indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(
                j, dtype=torch.int64)) for i, j in indices]
            return indices


# def build_sparse_inst_matcher(cfg):
#     name = cfg.MODEL.SPARSE_INST.MATCHER.NAME
#     return SPARSE_INST_MATCHER_REGISTRY.get(name)(cfg)


# def build_sparse_inst_criterion(cfg):
#     matcher = build_sparse_inst_matcher(cfg)
#     name = cfg.MODEL.SPARSE_INST.LOSS.NAME
#     return SPARSE_INST_CRITERION_REGISTRY.get(name)(cfg, matcher)
