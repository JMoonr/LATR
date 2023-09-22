# ==============================================================================
# Copyright (c) 2022 The PersFormer Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import re
import os
import sys
import copy
import json
import glob
import random
import pickle
import warnings
from pathlib import Path
import numpy as np
from numpy import int32#, result_type
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from utils.utils import *
sys.path.append('./')
warnings.simplefilter('ignore', np.RankWarning)
matplotlib.use('Agg')

from .transform import PhotoMetricDistortionMultiViewImage 

from tqdm import tqdm


class ApolloLaneDataset(Dataset):
    def __init__(self, dataset_base_dir, json_file_path, args, data_aug=False, **kwargs):
        # define image pre-processor
        self.totensor = transforms.ToTensor()
        # expect same mean/std for all torchvision models
        mean = [0.485, 0.456, 0.406] if args.mean is None else args.mean
        std = [0.229, 0.224, 0.225] if args.std is None else args.std
        self.normalize = transforms.Normalize(mean, std)

        self.data_aug = data_aug
        if data_aug:
            if hasattr(args, 'photo_aug'):
                self.photo_aug = PhotoMetricDistortionMultiViewImage(**args.photo_aug)
            else:
                self.photo_aug = False
        
        self.dataset_base_dir = dataset_base_dir
        self.json_file_path = json_file_path

        # dataset parameters
        self.dataset_name = args.dataset_name
        self.num_category = args.num_category

        self.h_org = args.org_h
        self.w_org = args.org_w
        self.h_crop = args.crop_y

        # parameters related to service network
        self.h_net = args.resize_h
        self.w_net = args.resize_w
        self.ipm_h = args.ipm_h
        self.ipm_w = args.ipm_w
        self.u_ratio = float(self.w_net) / float(self.w_org)
        self.v_ratio = float(self.h_net) / float(self.h_org - self.h_crop)
        self.top_view_region = args.top_view_region
        
        self.max_lanes = args.max_lanes

        self.K = args.K
        self.H_crop = homography_crop_resize([args.org_h, args.org_w], args.crop_y, [args.resize_h, args.resize_w])
        self.fix_cam = False
        
        self.x_min, self.x_max = self.top_view_region[0, 0], self.top_view_region[1, 0]
        self.y_min, self.y_max = self.top_view_region[2, 1], self.top_view_region[0, 1]
        
        self.anchor_y_steps = args.anchor_y_steps
        self.num_y_steps = len(self.anchor_y_steps)

        self.anchor_y_steps_dense = args.get(
            'anchor_y_steps_dense',
            np.linspace(3, 103, 200))
        args.anchor_y_steps_dense = self.anchor_y_steps_dense
        self.num_y_steps_dense = len(self.anchor_y_steps_dense)

        self.anchor_dim = 3 * self.num_y_steps + args.num_category

        self.save_json_path = args.save_json_path

        # parse ground-truth file
        self.processed_info_dict = None

        self.label_list = self.gen_single_file_json()
        self.n_samples = len(self.label_list)
        self.processed_info_dict = self.init_dataset_3D(dataset_base_dir, json_file_path)

    def gen_single_file_json(self):
        gt_labels_json_dict = [json.loads(line) for line in open(self.json_file_path, 'r').readlines()]
        
        # e.g., xxx/standard/train
        json_save_dir = self.json_file_path.split('.json')[0]

        mkdir_if_missing(json_save_dir)
        
        label_list_path = self.json_file_path.rsplit('/', 1)[0] + '/%s_json_list.txt' % self.json_file_path.rsplit('/', 1)[-1].split('.json')[0]

        if os.path.isfile(label_list_path):
            with open(label_list_path, 'r') as f:
                label_list = f.readlines()
                label_list = list(map(lambda x: os.path.join(json_save_dir, x.strip()), label_list))
        else:
            label_list = []
            
            for single_info in tqdm(gt_labels_json_dict):
                img_p = Path(single_info['raw_file'])
                json_dir = os.path.join(json_save_dir, img_p.parent.name)
                mkdir_if_missing(json_dir)
                json_p = os.path.join(json_dir, img_p.stem + '.json')
                single_info['file_path'] = json_p.split(json_save_dir + '/')[-1]
                json.dump(single_info, open(json_p, 'w'), separators=(',', ': '), indent=4)
                label_list.append(json_p)
        
            with open(label_list_path, 'w') as f:
                for label_js in label_list:
                    f.write(label_js)
                    f.write('\n')
        
        return label_list
    
        
    def parse_processed_info_dict_apollo(self, idx):
        keys = self.processed_info_dict.keys()
        keys = list(keys)
        res = []
        for k in keys:
            res.append(self.processed_info_dict[k][idx])
        return res


    def __len__(self):
        """
        Conventional len method
        """
        return len(self.label_list)

    # new getitem, WIP
    def WIP__getitem__(self, idx):
        """
        Args: idx (int): Index in list to load image
        """
        
        # preprocess data from json file

        _label_image_path, _label_cam_height, _label_cam_pitch, \
        cam_extrinsics, cam_intrinsics, \
        _label_laneline, _label_laneline_org, \
        _gt_laneline_visibility, _gt_laneline_category, \
        _gt_laneline_category_org, gt_laneline_img = self.parse_processed_info_dict_apollo(idx)
        
        if not self.fix_cam:
            gt_cam_height = _label_cam_height
            gt_cam_pitch = _label_cam_pitch
            intrinsics = cam_intrinsics
            extrinsics = cam_extrinsics
        else:
            raise ValueError('check release with training, fix_cam=False')
        img_name = _label_image_path

        with open(img_name, 'rb') as f:
            image = (Image.open(f).convert('RGB'))

        # image preprocess with crop and resize
        image = F.crop(image, self.h_crop, 0, self.h_org-self.h_crop, self.w_org)
        image = F.resize(image, size=(self.h_net, self.w_net), interpolation=InterpolationMode.BILINEAR)

        gt_category_2d = _gt_laneline_category_org

        if self.data_aug:
            img_rot, aug_mat = data_aug_rotate(image)
            if self.photo_aug:
                img_rot = self.photo_aug(
                    dict(img=img_rot.copy().astype(np.float32))
                )['img']
            
            image = Image.fromarray(np.clip(img_rot, 0, 255).astype(np.uint8))
        image = self.totensor(image).float()
        image = self.normalize(image)
        gt_cam_height = torch.tensor(gt_cam_height, dtype=torch.float32)
        gt_cam_pitch = torch.tensor(gt_cam_pitch, dtype=torch.float32)
        intrinsics = torch.from_numpy(intrinsics)
        extrinsics = torch.from_numpy(extrinsics)

        # prepare binary segmentation label map
        seg_label = np.zeros((self.h_net, self.w_net), dtype=np.uint8)
        seg_idx_label = np.zeros((self.max_lanes, self.h_net, self.w_net), dtype=np.uint8)
        ground_lanes = np.zeros((self.max_lanes, self.anchor_dim), dtype=np.float32)
        ground_lanes_dense = np.zeros(
            (self.max_lanes, self.num_y_steps_dense * 3), dtype=np.float32)
        
        gt_lanes = _label_laneline_org
        H_g2im, P_g2im, H_crop = self.transform_mats_impl(_label_cam_pitch, 
                                                                    _label_cam_height)
        M = np.matmul(H_crop, P_g2im)
        # update transformation with image augmentation
        if self.data_aug:
            M = np.matmul(aug_mat, M)
        
        lidar2img = np.eye(4).astype(np.float32)
        lidar2img[:3] = M
            
        SEG_WIDTH = 80
        thickness_st = int(SEG_WIDTH / 2550 * self.h_net / 2)

        for i, lane in enumerate(gt_lanes):
            if i >= self.max_lanes:
                break

            # TODO remove this
            if lane.shape[0] < 2:
                continue

            if _gt_laneline_category_org[i] > self.num_category:
                continue

            xs, zs = resample_laneline_in_y(lane, self.anchor_y_steps)
            vis = np.logical_and(
                self.anchor_y_steps > lane[:, 1].min() - 5,
                self.anchor_y_steps < lane[:, 1].max() + 5)

            ground_lanes[i][0: self.num_y_steps] = xs
            ground_lanes[i][self.num_y_steps:2*self.num_y_steps] = zs
            ground_lanes[i][2*self.num_y_steps:3*self.num_y_steps] = vis * 1.0
            ground_lanes[i][self.anchor_dim - self.num_category] = 0.0
            ground_lanes[i][self.anchor_dim - self.num_category + 1] = 1.0

            xs_dense, zs_dense = resample_laneline_in_y(
                lane, self.anchor_y_steps_dense)
            vis_dense = np.logical_and(
                self.anchor_y_steps_dense > lane[:, 1].min(),
                self.anchor_y_steps_dense < lane[:, 1].max())
            ground_lanes_dense[i][0: self.num_y_steps_dense] = xs_dense
            ground_lanes_dense[i][1*self.num_y_steps_dense: 2*self.num_y_steps_dense] = zs_dense
            ground_lanes_dense[i][2*self.num_y_steps_dense: 3*self.num_y_steps_dense] = vis_dense * 1.0

            x_2d, y_2d = projective_transformation(M, 
                                                   lane[:, 0],
                                                   lane[:, 1], 
                                                   lane[:, 2])
            
            for j in range(len(x_2d) - 1):
                # empirical setting.
                k = 2.7e-2 - ((2.5e-2 - 5e-5) / 600) * y_2d[j]
                thickness = max(round(thickness_st - k * (self.h_net - y_2d[j])), 2)
                if thickness >= 6:
                    thickness += 1

                seg_label = cv2.line(seg_label,
                                    (int(x_2d[j]), int(y_2d[j])), 
                                    (int(x_2d[j+1]), int(y_2d[j+1])),
                                    color=1,
                                    thickness=thickness)
                seg_idx_label[i] = cv2.line(seg_idx_label[i],
                                        (int(x_2d[j]), int(y_2d[j])),
                                        (int(x_2d[j+1]), int(y_2d[j+1])),
                                        color=gt_category_2d[i].item(),
                                        thickness=thickness,
                                        lineType=cv2.LINE_AA
                                        )

        seg_label = torch.from_numpy(seg_label.astype(np.float32))
        seg_label.unsqueeze_(0)
        
        extra_dict = {}
        
        extra_dict['seg_label'] = seg_label
        extra_dict['seg_idx_label'] = seg_idx_label
        extra_dict['ground_lanes'] = ground_lanes
        extra_dict['ground_lanes_dense'] = ground_lanes_dense
        extra_dict['lidar2img'] = lidar2img
        extra_dict['pad_shape'] = torch.Tensor(seg_idx_label.shape[-2:]).float()
        extra_dict['idx_json_file'] = self.label_list[idx]

        extra_dict['image'] = image
        if self.data_aug:
            aug_mat = torch.from_numpy(aug_mat.astype(np.float32))
            extra_dict['aug_mat'] = aug_mat
        
        extra_dict['cam_extrinsics'] = cam_extrinsics
        extra_dict['cam_intrinsics'] = cam_intrinsics
        return extra_dict

    # old getitem, workable
    def __getitem__(self, idx):
        """
        Args: idx (int): Index in list to load image
        """
        return self.WIP__getitem__(idx)

    def init_dataset_3D(self, dataset_base_dir, json_file_path):
        """
        :param dataset_info_file:
        :return: image paths, labels in unormalized net input coordinates

        data processing:
        ground truth labels map are scaled wrt network input sizes
        """

        # load image path, and lane pts
        label_image_path = []
        gt_laneline_pts_all = []
        gt_centerline_pts_all = []
        gt_laneline_visibility_all = []
        gt_centerline_visibility_all = []
        gt_laneline_category_all = []
        gt_cam_height_all = []
        gt_cam_pitch_all = []

        assert ops.exists(json_file_path), '{:s} not exist'.format(json_file_path)
        
        with open(json_file_path, 'r') as file:
            for idx, line in enumerate(file):
                
                info_dict = json.loads(line)
                # print('load json : %s | %s' % (idx, info_dict['raw_file']))
                image_path = ops.join(dataset_base_dir, info_dict['raw_file'])
                assert ops.exists(image_path), '{:s} not exist'.format(image_path)

                label_image_path.append(image_path)

                gt_lane_pts = info_dict['laneLines']
                gt_lane_visibility = info_dict['laneLines_visibility']
                for i, lane in enumerate(gt_lane_pts):
                    lane = np.array(lane)
                    gt_lane_pts[i] = lane
                    gt_lane_visibility[i] = np.array(gt_lane_visibility[i])
                
                gt_laneline_pts_all.append(gt_lane_pts)
                gt_laneline_visibility_all.append(gt_lane_visibility)
                
                if 'category' in info_dict:
                    gt_laneline_category = info_dict['category']
                    gt_laneline_category_all.append(np.array(gt_laneline_category, dtype=np.int32))
                else:
                    gt_laneline_category_all.append(np.ones(len(gt_lane_pts), dtype=np.int32))

                if not self.fix_cam:
                    gt_cam_height = info_dict['cam_height']
                    gt_cam_height_all.append(gt_cam_height)
                    gt_cam_pitch = info_dict['cam_pitch']
                    gt_cam_pitch_all.append(gt_cam_pitch)
        
        label_image_path = np.array(label_image_path)
        gt_cam_height_all = np.array(gt_cam_height_all)
        gt_cam_pitch_all = np.array(gt_cam_pitch_all)
        gt_laneline_pts_all_org = copy.deepcopy(gt_laneline_pts_all)
        gt_laneline_category_all_org = copy.deepcopy(gt_laneline_category_all)
        
        visibility_all_flat = []
        gt_laneline_im_all = []
        gt_centerline_im_all = []
        cam_extrinsics_all = []
        cam_intrinsics_all = []
        for idx in range(len(gt_laneline_pts_all)):
            # fetch camera height and pitch
            gt_cam_height = gt_cam_height_all[idx]
            gt_cam_pitch = gt_cam_pitch_all[idx]
            if not self.fix_cam:
                P_g2im = projection_g2im(gt_cam_pitch, gt_cam_height, self.K)
                H_g2im = homograpthy_g2im(gt_cam_pitch, gt_cam_height, self.K)
                H_im2g = np.linalg.inv(H_g2im)
            else:
                P_g2im = self.P_g2im
                H_im2g = self.H_im2g

            gt_lanes = gt_laneline_pts_all[idx]
            gt_visibility = gt_laneline_visibility_all[idx]

            # prune gt lanes by visibility labels
            gt_lanes = [prune_3d_lane_by_visibility(gt_lane, gt_visibility[k]) for k, gt_lane in enumerate(gt_lanes)]
            gt_laneline_pts_all_org[idx] = gt_lanes
            
            # project gt laneline to image plane
            gt_laneline_im = []
            for gt_lane in gt_lanes:
                x_vals, y_vals = projective_transformation(P_g2im, gt_lane[:,0], gt_lane[:,1], gt_lane[:,2])
                gt_laneline_im_oneline = np.array([x_vals, y_vals]).T.tolist()
                gt_laneline_im.append(gt_laneline_im_oneline)
            gt_laneline_im_all.append(gt_laneline_im)

            # generate ex/in from apollo
            cam_intrinsics = self.K
            cam_extrinsics = np.zeros((4,4))
            cam_extrinsics[-1, -1] = 1
            cam_extrinsics[2,3] = gt_cam_height
            cam_extrinsics_all.append(cam_extrinsics)
            cam_intrinsics_all.append(cam_intrinsics)
        
        visibility_all_flat = np.array(visibility_all_flat)
        
        processed_info_dict = {}
        processed_info_dict['label_json_path'] = label_image_path
        
        processed_info_dict['gt_cam_height_all'] = gt_cam_height_all
        processed_info_dict['gt_cam_pitch_all'] = gt_cam_pitch_all
        processed_info_dict['cam_extrinsics'] = cam_extrinsics_all
        processed_info_dict['cam_intrinsics'] = cam_intrinsics_all

        processed_info_dict['gt_laneline_pts_all'] = gt_laneline_pts_all
        processed_info_dict['gt_laneline_pts_all_org'] = gt_laneline_pts_all_org
        processed_info_dict['gt_laneline_visibility_all'] = gt_laneline_visibility_all

        processed_info_dict['gt_laneline_category_all'] = gt_laneline_category_all
        processed_info_dict['gt_laneline_category_all_org'] = gt_laneline_category_all_org
        processed_info_dict['gt_laneline_im_all'] = gt_laneline_im_all
        return processed_info_dict

    def transform_mats_impl(self, cam_pitch, cam_height):
        if not self.fix_cam:
            H_g2im = homograpthy_g2im(cam_pitch, cam_height, self.K)
            P_g2im = projection_g2im(cam_pitch, cam_height, self.K)
            return H_g2im, P_g2im, self.H_crop
        else:
            return self.H_g2im, self.P_g2im, self.H_crop

def data_aug_rotate(img):
    # assume img in PIL image format
    rot = random.uniform(-np.pi/18, np.pi/18)
    center_x = img.width / 2
    center_y = img.height / 2
    rot_mat = cv2.getRotationMatrix2D((center_x, center_y), rot, 1.0)
    img_rot = np.array(img)
    img_rot = cv2.warpAffine(img_rot, rot_mat, (img.width, img.height), flags=cv2.INTER_LINEAR)
    rot_mat = np.vstack([rot_mat, [0, 0, 1]])
    return img_rot, rot_mat


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_loader(transformed_dataset, args):
    """
        create dataset from ground-truth
        return a batch sampler based ont the dataset
    """

    sample_idx = range(transformed_dataset.n_samples)

    g = torch.Generator()
    g.manual_seed(0)

    discarded_sample_start = len(sample_idx) // args.batch_size * args.batch_size
    if args.proc_id == 0:
        print("Discarding images:")
    if args.proc_id == 0:
        if hasattr(transformed_dataset, '_label_image_path'):
            print(transformed_dataset._label_image_path[discarded_sample_start: len(sample_idx)])
        else:
            print(len(sample_idx) - discarded_sample_start)
    sample_idx = sample_idx[0 : discarded_sample_start]
    
    if args.dist:
        if args.proc_id == 0:
            print('use distributed sampler')
        if 'standard' in args.dataset_name or 'rare_subset' in args.dataset_name or 'illus_chg' in args.dataset_name:
            data_sampler = torch.utils.data.distributed.DistributedSampler(transformed_dataset, shuffle=True, drop_last=True)
            data_loader = DataLoader(transformed_dataset,
                                        batch_size=args.batch_size, 
                                        sampler=data_sampler,
                                        num_workers=args.nworkers, 
                                        pin_memory=True,
                                        persistent_workers=True,
                                        worker_init_fn=seed_worker,
                                        generator=g,
                                        drop_last=True)
        else:
            data_sampler = torch.utils.data.distributed.DistributedSampler(transformed_dataset)
            data_loader = DataLoader(transformed_dataset,
                                        batch_size=args.batch_size, 
                                        sampler=data_sampler,
                                        num_workers=args.nworkers, 
                                        pin_memory=True,
                                        persistent_workers=True,
                                        worker_init_fn=seed_worker,
                                        generator=g)
    else:
        if args.proc_id == 0:
            print("use default sampler")
        data_sampler = torch.utils.data.sampler.SubsetRandomSampler(sample_idx)
        data_loader = DataLoader(transformed_dataset,
                                batch_size=args.batch_size, sampler=data_sampler,
                                num_workers=args.nworkers, pin_memory=True,
                                persistent_workers=True,
                                worker_init_fn=seed_worker,
                                generator=g)

    if args.dist:
        return data_loader, data_sampler
    return data_loader
