import re
import os
import sys
import copy
import json
import glob
import random
import warnings
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from utils.utils import *
from experiments.gpu_utils import is_main_process

from .transform import PhotoMetricDistortionMultiViewImage

sys.path.append('./')
warnings.simplefilter('ignore', np.RankWarning)
matplotlib.use('Agg')

import yaml

class LaneDataset(Dataset):
    """
    Dataset with labeled lanes
        This implementation considers:
        w/o laneline 3D attributes
        w/o centerline annotations
        default considers 3D laneline, including centerlines

        This new version of data loader prepare ground-truth anchor tensor in flat ground space.
        It is assumed the dataset provides accurate visibility labels. Preparing ground-truth tensor depends on it.
    """
    # dataset_base_dir is image path, json_file_path is json file path,
    def __init__(self, dataset_base_dir, json_file_path, args, data_aug=False, save_std=False):
        """

        :param dataset_info_file: json file list
        """
        self.totensor = transforms.ToTensor()
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
        self.u_ratio = float(self.w_net) / float(self.w_org)
        self.v_ratio = float(self.h_net) / float(self.h_org - self.h_crop)
        self.top_view_region = args.top_view_region
        self.max_lanes = args.max_lanes

        self.K = args.K
        self.H_crop = homography_crop_resize([args.org_h, args.org_w], args.crop_y, [args.resize_h, args.resize_w])

        if args.fix_cam:
            self.fix_cam = True
            # compute the homography between image and IPM, and crop transformation
            self.cam_height = args.cam_height
            self.cam_pitch = np.pi / 180 * args.pitch
            self.P_g2im = projection_g2im(self.cam_pitch, self.cam_height, args.K)
        else:
            self.fix_cam = False

        # compute anchor steps
        self.use_default_anchor = args.use_default_anchor
        
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
        if 'openlane' in self.dataset_name:
            label_list = glob.glob(json_file_path + '**/*.json', recursive=True)
            self._label_list = label_list
        elif 'once' in self.dataset_name:
            label_list = glob.glob(json_file_path + '*/*/*.json', recursive=True)
            self._label_list = []
            for js_label_file in label_list:
                if not os.path.getsize(js_label_file):
                    continue
                image_path = map_once_json2img(js_label_file)
                if not os.path.exists(image_path):
                    continue
                self._label_list.append(js_label_file)
        else: 
            raise ValueError("to use ApolloDataset for apollo")
        
        if hasattr(self, '_label_list'):
            self.n_samples = len(self._label_list)
        else:
            self.n_samples = self._label_image_path.shape[0]

    def preprocess_data_from_json_once(self, idx_json_file):
        _label_image_path = None
        _label_cam_height = None
        _label_cam_pitch = None
        cam_extrinsics = None
        cam_intrinsics = None
        _label_laneline_org = None
        _gt_laneline_category_org = None

        image_path = map_once_json2img(idx_json_file)

        assert ops.exists(image_path), '{:s} not exist'.format(image_path)
        _label_image_path = image_path

        with open(idx_json_file, 'r') as file:
            file_lines = [line for line in file]
            if len(file_lines) != 0:
                info_dict = json.loads(file_lines[0])
            else:
                print('Empty label_file:', idx_json_file)
                return

            if not self.fix_cam:
                cam_pitch = 0.3/180*np.pi
                cam_height = 1.5
                cam_extrinsics = np.array([[np.cos(cam_pitch), 0, -np.sin(cam_pitch), 0],
                                            [0, 1, 0, 0],
                                            [np.sin(cam_pitch), 0,  np.cos(cam_pitch), cam_height],
                                            [0, 0, 0, 1]], dtype=float)
                R_vg = np.array([[0, 1, 0],
                                    [-1, 0, 0],
                                    [0, 0, 1]], dtype=float)
                R_gc = np.array([[1, 0, 0],
                                    [0, 0, 1],
                                    [0, -1, 0]], dtype=float)
                cam_extrinsics[:3, :3] = np.matmul(np.matmul(
                                            np.matmul(np.linalg.inv(R_vg), cam_extrinsics[:3, :3]),
                                                R_vg), R_gc)
                cam_extrinsics[0:2, 3] = 0.0

                gt_cam_height = cam_extrinsics[2, 3] 
                gt_cam_pitch = 0

                if 'calibration' in info_dict:
                    cam_intrinsics = info_dict['calibration']
                    cam_intrinsics = np.array(cam_intrinsics)
                    cam_intrinsics = cam_intrinsics[:, :3]
                else:
                    cam_intrinsics = self.K

            _label_cam_height = gt_cam_height
            _label_cam_pitch = gt_cam_pitch

            gt_lanes_packed = info_dict['lanes']
            gt_lane_pts, gt_lane_visibility, gt_laneline_category = [], [], []
            for i, gt_lane_packed in enumerate(gt_lanes_packed):
                lane = np.array(gt_lane_packed).T

                # Coordinate convertion for openlane_300 data
                lane = np.vstack((lane, np.ones((1, lane.shape[1]))))
                lane = np.matmul(cam_extrinsics, lane)

                lane = lane[0:3, :].T
                lane = lane[lane[:,1].argsort()] #TODO:make y mono increase
                gt_lane_pts.append(lane)
                gt_lane_visibility.append(1.0)
                gt_laneline_category.append(1)

        _gt_laneline_category_org = copy.deepcopy(np.array(gt_laneline_category))

        if not self.fix_cam:
            cam_K = cam_intrinsics
            if 'openlane' in self.dataset_name or 'once' in self.dataset_name:
                cam_E = cam_extrinsics
                P_g2im = projection_g2im_extrinsic(cam_E, cam_K)
                H_g2im = homograpthy_g2im_extrinsic(cam_E, cam_K)
            else:
                gt_cam_height = _label_cam_height
                gt_cam_pitch = _label_cam_pitch
                P_g2im = projection_g2im(gt_cam_pitch, gt_cam_height, cam_K)
                H_g2im = homograpthy_g2im(gt_cam_pitch, gt_cam_height, cam_K)
            H_im2g = np.linalg.inv(H_g2im)
        else:
            P_g2im = self.P_g2im
            H_im2g = self.H_im2g
        P_g2gflat = np.matmul(H_im2g, P_g2im)

        gt_lanes = gt_lane_pts
        gt_visibility = gt_lane_visibility
        gt_category = gt_laneline_category

        # prune gt lanes by visibility labels
        gt_lanes = [prune_3d_lane_by_visibility(gt_lane, gt_visibility[k]).squeeze(0) for k, gt_lane in enumerate(gt_lanes)]
        _label_laneline_org = copy.deepcopy(gt_lanes)
        return _label_image_path, _label_cam_height, _label_cam_pitch, \
               cam_extrinsics, cam_intrinsics, \
               _label_laneline_org, \
               _gt_laneline_category_org, info_dict
               #    _label_laneline, \
               #    _gt_laneline_visibility, _gt_laneline_category, \

    def preprocess_data_from_json_openlane(self, idx_json_file):
        _label_image_path = None
        _label_cam_height = None
        _label_cam_pitch = None
        cam_extrinsics = None
        cam_intrinsics = None
        # _label_laneline = None
        _label_laneline_org = None
        # _gt_laneline_visibility = None
        # _gt_laneline_category = None
        _gt_laneline_category_org = None
        # _laneline_ass_id = None

        with open(idx_json_file, 'r') as file:
            file_lines = [line for line in file]
            info_dict = json.loads(file_lines[0])

            image_path = ops.join(self.dataset_base_dir, info_dict['file_path'])
            assert ops.exists(image_path), '{:s} not exist'.format(image_path)
            _label_image_path = image_path

            if not self.fix_cam:
                cam_extrinsics = np.array(info_dict['extrinsic'])
                # Re-calculate extrinsic matrix based on ground coordinate
                R_vg = np.array([[0, 1, 0],
                                    [-1, 0, 0],
                                    [0, 0, 1]], dtype=float)
                R_gc = np.array([[1, 0, 0],
                                    [0, 0, 1],
                                    [0, -1, 0]], dtype=float)
                cam_extrinsics[:3, :3] = np.matmul(np.matmul(
                                            np.matmul(np.linalg.inv(R_vg), cam_extrinsics[:3, :3]),
                                                R_vg), R_gc)
                cam_extrinsics[0:2, 3] = 0.0
                
                # gt_cam_height = info_dict['cam_height']
                gt_cam_height = cam_extrinsics[2, 3]
                if 'cam_pitch' in info_dict:
                    gt_cam_pitch = info_dict['cam_pitch']
                else:
                    gt_cam_pitch = 0

                if 'intrinsic' in info_dict:
                    cam_intrinsics = info_dict['intrinsic']
                    cam_intrinsics = np.array(cam_intrinsics)
                else:
                    cam_intrinsics = self.K  

            _label_cam_height = gt_cam_height
            _label_cam_pitch = gt_cam_pitch

            gt_lanes_packed = info_dict['lane_lines']
            gt_lane_pts, gt_lane_visibility, gt_laneline_category = [], [], []
            for i, gt_lane_packed in enumerate(gt_lanes_packed):
                # A GT lane can be either 2D or 3D
                # if a GT lane is 3D, the height is intact from 3D GT, so keep it intact here too
                lane = np.array(gt_lane_packed['xyz'])
                lane_visibility = np.array(gt_lane_packed['visibility'])

                # Coordinate convertion for openlane_300 data
                lane = np.vstack((lane, np.ones((1, lane.shape[1]))))
                cam_representation = np.linalg.inv(
                                        np.array([[0, 0, 1, 0],
                                                    [-1, 0, 0, 0],
                                                    [0, -1, 0, 0],
                                                    [0, 0, 0, 1]], dtype=float))  # transformation from apollo camera to openlane camera
                lane = np.matmul(cam_extrinsics, np.matmul(cam_representation, lane))

                lane = lane[0:3, :].T
                gt_lane_pts.append(lane)
                gt_lane_visibility.append(lane_visibility)

                if 'category' in gt_lane_packed:
                    lane_cate = gt_lane_packed['category']
                    if lane_cate == 21:  # merge left and right road edge into road edge
                        lane_cate = 20
                    gt_laneline_category.append(lane_cate)
                else:
                    gt_laneline_category.append(1)
        
        # _label_laneline_org = copy.deepcopy(gt_lane_pts)
        _gt_laneline_category_org = copy.deepcopy(np.array(gt_laneline_category))

        gt_lanes = gt_lane_pts
        gt_visibility = gt_lane_visibility
        gt_category = gt_laneline_category

        # prune gt lanes by visibility labels
        gt_lanes = [prune_3d_lane_by_visibility(gt_lane, gt_visibility[k]) for k, gt_lane in enumerate(gt_lanes)]
        _label_laneline_org = copy.deepcopy(gt_lanes)

        return _label_image_path, _label_cam_height, _label_cam_pitch, \
               cam_extrinsics, cam_intrinsics, \
               _label_laneline_org, \
               _gt_laneline_category_org, info_dict

    def __len__(self):
        """
        Conventional len method
        """
        return self.n_samples

    # new getitem, WIP
    def WIP__getitem__(self, idx):
        """
        Args: idx (int): Index in list to load image
        """
        extra_dict = {}

        idx_json_file = self._label_list[idx]
        # preprocess data from json file
        if 'openlane' in self.dataset_name:
            _label_image_path, _label_cam_height, _label_cam_pitch, \
            cam_extrinsics, cam_intrinsics, \
            _label_laneline_org, \
            _gt_laneline_category_org, info_dict = self.preprocess_data_from_json_openlane(idx_json_file)
        elif 'once' in self.dataset_name:
            _label_image_path, _label_cam_height, _label_cam_pitch, \
            cam_extrinsics, cam_intrinsics, \
            _label_laneline_org, \
            _gt_laneline_category_org, info_dict = self.preprocess_data_from_json_once(idx_json_file)

        # fetch camera height and pitch
        if not self.fix_cam:
            gt_cam_height = _label_cam_height
            gt_cam_pitch = _label_cam_pitch
            if 'openlane' in self.dataset_name or 'once' in self.dataset_name:
                intrinsics = cam_intrinsics
                extrinsics = cam_extrinsics
            else:
                # should not be used
                intrinsics = self.K
                extrinsics = np.zeros((3,4))
                extrinsics[2,3] = gt_cam_height
        else:
            gt_cam_height = self.cam_height
            gt_cam_pitch = self.cam_pitch
            # should not be used
            intrinsics = self.K
            extrinsics = np.zeros((3,4))
            extrinsics[2,3] = gt_cam_height

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
            image = Image.fromarray(
                np.clip(img_rot, 0, 255).astype(np.uint8))
        image = self.totensor(image).float()
        image = self.normalize(image)
        intrinsics = torch.from_numpy(intrinsics)
        extrinsics = torch.from_numpy(extrinsics)

        # prepare binary segmentation label map
        seg_label = np.zeros((self.h_net, self.w_net), dtype=np.int8)
        # seg idx has the same order as gt_lanes
        seg_idx_label = np.zeros((self.max_lanes, self.h_net, self.w_net), dtype=np.uint8)
        ground_lanes = np.zeros((self.max_lanes, self.anchor_dim), dtype=np.float32)
        ground_lanes_dense = np.zeros(
            (self.max_lanes, self.num_y_steps_dense * 3), dtype=np.float32)
        gt_lanes = _label_laneline_org # ground
        gt_laneline_img = [[0]] * len(gt_lanes)

        H_g2im, P_g2im, H_crop = self.transform_mats_impl(cam_extrinsics, \
                                            cam_intrinsics, _label_cam_pitch, _label_cam_height)
        M = np.matmul(H_crop, P_g2im)
        # update transformation with image augmentation
        if self.data_aug:
            M = np.matmul(aug_mat, M)

        lidar2img = np.eye(4).astype(np.float32)
        lidar2img[:3] = M

        SEG_WIDTH = 80
        thickness = int(SEG_WIDTH / 2650 * self.h_net / 2)

        for i, lane in enumerate(gt_lanes):
            if i >= self.max_lanes:
                break

            if lane.shape[0] <= 2:
                continue

            if _gt_laneline_category_org[i] >= self.num_category:
                continue

            xs, zs = resample_laneline_in_y(lane, self.anchor_y_steps)
            vis = np.logical_and(
                self.anchor_y_steps > lane[:, 1].min() - 5,
                self.anchor_y_steps < lane[:, 1].max() + 5)

            ground_lanes[i][0: self.num_y_steps] = xs
            ground_lanes[i][self.num_y_steps:2*self.num_y_steps] = zs
            ground_lanes[i][2*self.num_y_steps:3*self.num_y_steps] = vis * 1.0
            ground_lanes[i][self.anchor_dim - self.num_category] = 0.0
            ground_lanes[i][self.anchor_dim - self.num_category + _gt_laneline_category_org[i]] = 1.0

            xs_dense, zs_dense = resample_laneline_in_y(
                lane, self.anchor_y_steps_dense)
            vis_dense = np.logical_and(
                self.anchor_y_steps_dense > lane[:, 1].min(),
                self.anchor_y_steps_dense < lane[:, 1].max())
            ground_lanes_dense[i][0: self.num_y_steps_dense] = xs_dense
            ground_lanes_dense[i][1*self.num_y_steps_dense: 2*self.num_y_steps_dense] = zs_dense
            ground_lanes_dense[i][2*self.num_y_steps_dense: 3*self.num_y_steps_dense] = vis_dense * 1.0

            x_2d, y_2d = projective_transformation(M, lane[:, 0],
                                                   lane[:, 1], lane[:, 2])
            gt_laneline_img[i] = np.array([x_2d, y_2d]).T.tolist()
            for j in range(len(x_2d) - 1):
                seg_label = cv2.line(seg_label,
                                     (int(x_2d[j]), int(y_2d[j])), (int(x_2d[j+1]), int(y_2d[j+1])),
                                     color=np.asscalar(np.array([1])),
                                     thickness=thickness)
                seg_idx_label[i] = cv2.line(
                    seg_idx_label[i],
                    (int(x_2d[j]), int(y_2d[j])), (int(x_2d[j+1]), int(y_2d[j+1])),
                    color=gt_category_2d[i].item(),
                    thickness=thickness)

        seg_label = torch.from_numpy(seg_label.astype(np.float32))
        seg_label.unsqueeze_(0)
        extra_dict['seg_label'] = seg_label
        extra_dict['seg_idx_label'] = seg_idx_label
        extra_dict['ground_lanes'] = ground_lanes
        extra_dict['ground_lanes_dense'] = ground_lanes_dense
        extra_dict['lidar2img'] = lidar2img
        extra_dict['pad_shape'] = torch.Tensor(seg_idx_label.shape[-2:]).float()
        extra_dict['idx_json_file'] = idx_json_file
        extra_dict['image'] = image
        if self.data_aug:
            aug_mat = torch.from_numpy(aug_mat.astype(np.float32))
            extra_dict['aug_mat'] = aug_mat
        return extra_dict

    # old getitem, workable
    def __getitem__(self, idx):
        """
        Args: idx (int): Index in list to load image
        """
        return self.WIP__getitem__(idx)

    def transform_mats_impl(self, cam_extrinsics, cam_intrinsics, cam_pitch, cam_height):
        if not self.fix_cam:
            if 'openlane' in self.dataset_name or 'once' in self.dataset_name:
                H_g2im = homograpthy_g2im_extrinsic(cam_extrinsics, cam_intrinsics)
                P_g2im = projection_g2im_extrinsic(cam_extrinsics, cam_intrinsics)
            else:
                H_g2im = homograpthy_g2im(cam_pitch, cam_height, self.K)
                P_g2im = projection_g2im(cam_pitch, cam_height, self.K)
            return H_g2im, P_g2im, self.H_crop
        else:
            return self.H_g2im, self.P_g2im, self.H_crop

def make_lane_y_mono_inc(lane):
    """
        Due to lose of height dim, projected lanes to flat ground plane may not have monotonically increasing y.
        This function trace the y with monotonically increasing y, and output a pruned lane
    :param lane:
    :return:
    """
    idx2del = []
    max_y = lane[0, 1]
    for i in range(1, lane.shape[0]):
        # hard-coded a smallest step, so the far-away near horizontal tail can be pruned
        if lane[i, 1] <= max_y + 3:
            idx2del.append(i)
        else:
            max_y = lane[i, 1]
    lane = np.delete(lane, idx2del, 0)
    return lane

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

    # transformed_dataset = LaneDataset(dataset_base_dir, json_file_path, args)
    sample_idx = range(transformed_dataset.n_samples)

    g = torch.Generator()
    g.manual_seed(0)

    discarded_sample_start = len(sample_idx) // args.batch_size * args.batch_size
    if is_main_process():
        print("Discarding images:")
        if hasattr(transformed_dataset, '_label_image_path'):
            print(transformed_dataset._label_image_path[discarded_sample_start: len(sample_idx)])
        else:
            print(len(sample_idx) - discarded_sample_start)
    sample_idx = sample_idx[0 : discarded_sample_start]
    
    if args.dist:
        if is_main_process():
            print('use distributed sampler')
        if 'standard' in args.dataset_name or 'rare_subset' in args.dataset_name or 'illus_chg' in args.dataset_name:
            data_sampler = torch.utils.data.distributed.DistributedSampler(transformed_dataset, shuffle=True, drop_last=True)
            data_loader = DataLoader(transformed_dataset,
                                        batch_size=args.batch_size, 
                                        sampler=data_sampler,
                                        num_workers=args.nworkers, 
                                        pin_memory=True,
                                        persistent_workers=args.nworkers > 0,
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
                                        persistent_workers=args.nworkers > 0,
                                        worker_init_fn=seed_worker,
                                        generator=g)
    else:
        if is_main_process():
            print("use default sampler")
        data_sampler = torch.utils.data.sampler.SubsetRandomSampler(sample_idx)
        data_loader = DataLoader(transformed_dataset,
                                batch_size=args.batch_size, sampler=data_sampler,
                                num_workers=args.nworkers, pin_memory=True,
                                persistent_workers=args.nworkers > 0,
                                worker_init_fn=seed_worker,
                                generator=g)

    if args.dist:
        return data_loader, data_sampler
    return data_loader

def map_once_json2img(json_label_file):
    if 'train' in json_label_file:
        split_name = 'train'
    elif 'val' in json_label_file:
        split_name = 'val'
    elif 'test' in json_label_file:
        split_name = 'test'
    else:
        raise ValueError("train/val/test not in the json path")
    image_path = json_label_file.replace(split_name, 'data').replace('.json', '.jpg')
    return image_path
