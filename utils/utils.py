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

import argparse
import errno
import os
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.init as init
import torch.optim
from torch.optim import lr_scheduler
import os.path as ops


from scipy.interpolate import interp1d
from scipy.special import softmax
import logging, datetime

from experiments.gpu_utils import is_main_process
from mmdet.utils import get_root_logger as get_mmdet_root_logger


def create_logger(args):
    datenow = datetime.datetime.now()
    ymd = '-'.join(list(map(str, [datenow.year, datenow.month, datenow.day])))
    hms = ':'.join(list(map(str, [datenow.hour, datenow.minute, datenow.second])))
    logname = '%s_%s' % (ymd, hms)
    logdir = os.path.join(args.save_path, 'logs')
    os.makedirs(logdir, exist_ok=True)

    ckpt_name = Path(args.eval_ckpt).stem.split('checkpoint_model_epoch_')[-1]
    logtype = 'eval_{}'.format(ckpt_name)  if args.evaluate else 'train'
    filename = os.path.join(logdir, '%s_%s.log' % (logtype, logname))

    logging.basicConfig(level=logging.INFO, 
                        format ='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d-%b-%Y %H:%W:%S',
                        filename=filename,
                        filemode= 'w'
                        )

    # logger = logging.getLogger(filename)
    logger = get_mmdet_root_logger(log_file=filename, log_level=logging.INFO)

    return logger


def define_args():
    parser = argparse.ArgumentParser(description='PersFormer_3DLane_Detection')
    
    # CUDNN usage
    parser.add_argument("--cudnn", type=str2bool, nargs='?', const=True, default=True, help="cudnn optimization active")
    
    # DDP setting
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--gpu', type=int, default = 0)
    parser.add_argument('--world_size', type=int, default = 1)
    parser.add_argument('--nodes', type=int, default = 1)
    parser.add_argument('--eval_ckpt', type=str, default='')
    parser.add_argument('--resume_from', type=str, default='')
    parser.add_argument('--no_eval', action='store_true')

    # General model settings
    parser.add_argument('--nworkers', type=int, default=0, help='num of threads')
    parser.add_argument('--test_mode', action='store_true', help='prevents loading latest saved model')
    parser.add_argument('--start_epoch', type=int, default=0, help='prevents loading latest saved model')
    parser.add_argument('--evaluate', action='store_true', default=False, help='only perform evaluation')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--resume', type=str, default='', help='resume latest saved run')
    parser.add_argument('--output_dir', default='openlane', type=str, 
                        help='output_dir name under `work_dirs`')
    parser.add_argument('--evaluate_case', default='', type=str, 
                        help='scene name, some are in shor.')
    parser.add_argument('--eval_freq', type=int, default=2,
                        help='evaluation frequency during training, 0 means no eval', )
    
    # eval using gen-laneNet
    parser.add_argument('--rewrite_pred', default=False, action='store_true', help='whether rewrite existing pred .json file.')
    parser.add_argument('--save_best', default=False, action='store_true', help='only save best ckpt.')
    
    # workdir
    parser.add_argument('--save_root', default='work_dirs', type=str)
    # dataset
    parser.add_argument('--dataset', default='300', type=str, help='1000 | 300 openlane dataset')
    return parser


def prune_3d_lane_by_visibility(lane_3d, visibility):
    lane_3d = lane_3d[visibility > 0, ...]
    return lane_3d


def prune_3d_lane_by_range(lane_3d, x_min, x_max):
    # TODO: solve hard coded range later
    # remove points with y out of range
    # 3D label may miss super long straight-line with only two points: Not have to be 200, gt need a min-step
    # 2D dataset requires this to rule out those points projected to ground, but out of meaningful range
    lane_3d = lane_3d[np.logical_and(lane_3d[:, 1] > 0, lane_3d[:, 1] < 200), ...]

    # remove lane points out of x range
    lane_3d = lane_3d[np.logical_and(lane_3d[:, 0] > x_min,
                                     lane_3d[:, 0] < x_max), ...]
    return lane_3d


def resample_laneline_in_y(input_lane, y_steps, out_vis=False):
    """
        Interpolate x, z values at each anchor grid, including those beyond the range of input lnae y range
    :param input_lane: N x 2 or N x 3 ndarray, one row for a point (x, y, z-optional).
                       It requires y values of input lane in ascending order
    :param y_steps: a vector of steps in y
    :param out_vis: whether to output visibility indicator which only depends on input y range
    :return:
    """

    # at least two points are included
    assert(input_lane.shape[0] >= 2)

    y_min = np.min(input_lane[:, 1])-5
    y_max = np.max(input_lane[:, 1])+5

    if input_lane.shape[1] < 3:
        input_lane = np.concatenate([input_lane, np.zeros([input_lane.shape[0], 1], dtype=np.float32)], axis=1)

    f_x = interp1d(input_lane[:, 1], input_lane[:, 0], fill_value="extrapolate")
    f_z = interp1d(input_lane[:, 1], input_lane[:, 2], fill_value="extrapolate")

    x_values = f_x(y_steps)
    z_values = f_z(y_steps)

    if out_vis:
        output_visibility = np.logical_and(y_steps >= y_min, y_steps <= y_max)
        return x_values, z_values, output_visibility.astype(np.float32) + 1e-9
    return x_values, z_values


def resample_laneline_in_y_with_vis(input_lane, y_steps, vis_vec):
    """
        Interpolate x, z values at each anchor grid, including those beyond the range of input lnae y range
    :param input_lane: N x 2 or N x 3 ndarray, one row for a point (x, y, z-optional).
                       It requires y values of input lane in ascending order
    :param y_steps: a vector of steps in y
    :param out_vis: whether to output visibility indicator which only depends on input y range
    :return:
    """

    # at least two points are included
    assert(input_lane.shape[0] >= 2)

    if input_lane.shape[1] < 3:
        input_lane = np.concatenate([input_lane, np.zeros([input_lane.shape[0], 1], dtype=np.float32)], axis=1)

    f_x = interp1d(input_lane[:, 1], input_lane[:, 0], fill_value="extrapolate")
    f_z = interp1d(input_lane[:, 1], input_lane[:, 2], fill_value="extrapolate")
    f_vis = interp1d(input_lane[:, 1], vis_vec, fill_value="extrapolate")

    x_values = f_x(y_steps)
    z_values = f_z(y_steps)
    vis_values = f_vis(y_steps)

    x_values = x_values[vis_values > 0.5]
    y_values = y_steps[vis_values > 0.5]
    z_values = z_values[vis_values > 0.5]
    return np.array([x_values, y_values, z_values]).T


def homograpthy_g2im(cam_pitch, cam_height, K):
    # transform top-view region to original image region
    R_g2c = np.array([[1, 0, 0],
                      [0, np.cos(np.pi / 2 + cam_pitch), -np.sin(np.pi / 2 + cam_pitch)],
                      [0, np.sin(np.pi / 2 + cam_pitch), np.cos(np.pi / 2 + cam_pitch)]])
    H_g2im = np.matmul(K, np.concatenate([R_g2c[:, 0:2], [[0], [cam_height], [0]]], 1))
    return H_g2im


def projection_g2im(cam_pitch, cam_height, K):
    P_g2c = np.array([[1,                             0,                              0,          0],
                      [0, np.cos(np.pi / 2 + cam_pitch), -np.sin(np.pi / 2 + cam_pitch), cam_height],
                      [0, np.sin(np.pi / 2 + cam_pitch),  np.cos(np.pi / 2 + cam_pitch),          0]])
    P_g2im = np.matmul(K, P_g2c)
    return P_g2im


def homograpthy_g2im_extrinsic(E, K):
    """E: extrinsic matrix, 4*4"""
    E_inv = np.linalg.inv(E)[0:3, :]
    H_g2c = E_inv[:, [0,1,3]]
    H_g2im = np.matmul(K, H_g2c)
    return H_g2im


def projection_g2im_extrinsic(E, K):
    E_inv = np.linalg.inv(E)[0:3, :]
    P_g2im = np.matmul(K, E_inv)
    return P_g2im


def homography_crop_resize(org_img_size, crop_y, resize_img_size):
    """
        compute the homography matrix transform original image to cropped and resized image
    :param org_img_size: [org_h, org_w]
    :param crop_y:
    :param resize_img_size: [resize_h, resize_w]
    :return:
    """
    # transform original image region to network input region
    ratio_x = resize_img_size[1] / org_img_size[1]
    ratio_y = resize_img_size[0] / (org_img_size[0] - crop_y)
    H_c = np.array([[ratio_x, 0, 0],
                    [0, ratio_y, -ratio_y*crop_y],
                    [0, 0, 1]])
    return H_c


def homographic_transformation(Matrix, x, y):
    """
    Helper function to transform coordinates defined by transformation matrix

    Args:
            Matrix (multi dim - array): 3x3 homography matrix
            x (array): original x coordinates
            y (array): original y coordinates
    """
    ones = np.ones((1, len(y)))
    coordinates = np.vstack((x, y, ones))
    trans = np.matmul(Matrix, coordinates)

    x_vals = trans[0, :]/trans[2, :]
    y_vals = trans[1, :]/trans[2, :]
    return x_vals, y_vals


def projective_transformation(Matrix, x, y, z):
    """
    Helper function to transform coordinates defined by transformation matrix

    Args:
            Matrix (multi dim - array): 3x4 projection matrix
            x (array): original x coordinates
            y (array): original y coordinates
            z (array): original z coordinates
    """
    ones = np.ones((1, len(z)))
    coordinates = np.vstack((x, y, z, ones))
    trans = np.matmul(Matrix, coordinates)

    x_vals = trans[0, :]/trans[2, :]
    y_vals = trans[1, :]/trans[2, :]
    return x_vals, y_vals


def first_run(save_path):
    txt_file = os.path.join(save_path,'first_run.txt')
    if not os.path.exists(txt_file):
        open(txt_file, 'w').close()
    else:
        saved_epoch = open(txt_file).read()
        if saved_epoch is None:
            print('You forgot to delete [first run file]')
            return '' 
        return saved_epoch
    return ''


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


# trick from stackoverflow
def str2bool(argument):
    if argument.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif argument.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Wrong argument in argparse, should be a boolean')


class Logger(object):
    """
    Source https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        self.fpath = fpath
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def define_optim(optim, params, lr, weight_decay):
    if optim == 'adam':
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optim == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optim == 'sgd':
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optim == 'rmsprop':
        optimizer = torch.optim.RMSprop(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise KeyError("The requested optimizer: {} is not implemented".format(optim))
    return optimizer


def cosine_schedule_with_warmup(k, args, dataset_size=None):
    # k : iter num
    num_gpu = args.world_size
    dataset_size = dataset_size
    batch_size = args.batch_size
    num_epochs = args.nepochs

    if num_gpu == 1:
        warmup_iters = 0
    else:
        warmup_iters = 1000 // num_gpu

    if k < warmup_iters:
        return (k + 1) / warmup_iters
    else:
        iter_per_epoch = (dataset_size + batch_size - 1) // batch_size
        return 0.5 * (1 + np.cos(np.pi * (k - warmup_iters) / (num_epochs * iter_per_epoch)))


def define_scheduler(optimizer, args, dataset_size=None):
    if args.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - args.niter) / float(args.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=args.lr_decay_iters, gamma=args.gamma)
    elif args.lr_policy == 'multi_step':
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, step_size=args.lr_multi_steps, gamma=args.gamma)
    elif args.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=args.T_max, eta_min=args.eta_min)
    elif args.lr_policy == 'cosine_warm':
        '''
        lr_config = dict(
            policy='CosineAnnealing',
            warmup='linear',
            warmup_iters=500,
            warmup_ratio=1.0 / 3,
            min_lr_ratio=1e-3)
        '''
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                             T_0=args.T_0, T_mult=args.T_mult, eta_min=args.eta_min)

    # elif args.lr_policy == 'plateau':
    #     scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
    #                                                factor=args.gamma,
    #                                                threshold=0.0001,
    #                                                patience=args.lr_decay_iters)
    elif args.lr_policy == 'cosine_warmup':
        from functools import partial
        cosine_warmup = partial(cosine_schedule_with_warmup, args=args, dataset_size=dataset_size)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_warmup)
        
    elif args.lr_policy == 'None':
        scheduler = None
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


def define_init_weights(model, init_w='normal', activation='relu'):
    # print('Init weights in network with [{}]'.format(init_w))
    if init_w == 'normal':
        model.apply(weights_init_normal)
    elif init_w == 'xavier':
        model.apply(weights_init_xavier)
    elif init_w == 'kaiming':
        model.apply(weights_init_kaiming)
    elif init_w == 'orthogonal':
        model.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [{}] is not implemented'.format(init_w))


def weights_init_normal(m):
    classname = m.__class__.__name__
#    print(classname)
    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
        try:
            init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        except:
            print("{} not support init".format(str(classname)))
    elif classname.find('Linear') != -1:
        try:
            init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        except:
            print("{} not support init".format(str(classname)))
    elif classname.find('BatchNorm2d') != -1:
        try:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
        except:
            print("{} not support init".format(str(classname)))

def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
#    print(classname)
    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
        init.orthogonal(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
