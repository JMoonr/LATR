import torch
import torch.optim
import torch.nn as nn
import numpy as np
import glob
import time
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter
import traceback
import shutil

from data.Load_Data import *
from data.apollo_dataset import ApolloLaneDataset
from models.latr import LATR
from experiments.gpu_utils import is_main_process
from utils import eval_3D_lane, eval_3D_once
from utils import eval_3D_lane_apollo
from utils.utils import *

# ddp related
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .ddp import *
import os.path as osp
from .gpu_utils import gpu_available
from mmcv.runner.optimizer import build_optimizer


class Runner:
    def __init__(self, args):
        self.args = args
        set_work_dir(self.args)
        self.logger = create_logger(args)

        # Check GPU availability
        if is_main_process():
            if not gpu_available():
                raise Exception("No gpu available for usage")
            if int(os.getenv('WORLD_SIZE', 1)) >= 1:
                self.logger.info("Let's use %s" % os.environ['WORLD_SIZE'] + "GPUs!")
                torch.cuda.empty_cache()
        
        # Get Dataset
        if is_main_process():
            self.logger.info("Loading Dataset ...")

        self.val_gt_file = ops.join(args.save_path, 'test.json')
        if not args.evaluate:
            self.train_dataset, self.train_loader, self.train_sampler = self._get_train_dataset()
        else:
            self.train_dataset, self.train_loader, self.train_sampler = [],[],[]
        self.valid_dataset, self.valid_loader, self.valid_sampler = self._get_valid_dataset()

        if 'openlane' in args.dataset_name:
            self.evaluator = eval_3D_lane.LaneEval(args, logger=self.logger)
        elif 'apollo' in args.dataset_name:
            self.evaluator = eval_3D_lane_apollo.LaneEval(args, logger=self.logger)
        elif 'once' in args.dataset_name:
            self.evaluator = eval_3D_once.LaneEval()
        else:
            assert False
        # Tensorboard writer
        if not args.no_tb and is_main_process():
            tensorboard_path = os.path.join(args.save_path, 'Tensorboard/')
            mkdir_if_missing(tensorboard_path)
            self.writer = SummaryWriter(tensorboard_path)
        
        if is_main_process():
            self.logger.info("Init Done!")
        
        self.is_apollo = False
        if 'apollo' in args.dataset_name:
            self.is_apollo = True

    def train(self):
        args = self.args

        # Get Dataset
        train_loader = self.train_loader
        train_sampler = self.train_sampler

        global lowest_loss, best_f1_epoch, best_val_f1, best_epoch
        # Define model or resume
        
        model, optimizer, scheduler, best_epoch, \
            lowest_loss, best_f1_epoch, best_val_f1 = self._get_model_ddp()
        
        self._log_model_info(model)
        
        def save_cur_ckpt(
                loss,
                with_eval=True,
                eval_stats=None):
            # Save model
            if not with_eval:
                self.save_checkpoint({
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, False, epoch+1, self.args.save_path)
            else:
                total_score = loss.item() # loss_list[0].avg
                if is_main_process():
                    # File to keep latest epoch
                    with open(os.path.join(args.save_path, 'first_run.txt'), 'w') as f:
                        f.write(str(epoch + 1))
                global best_val_f1, best_f1_epoch, lowest_loss, best_epoch

                to_copy, to_save = False, True # False if args.save_best else True

                if total_score < lowest_loss:
                    best_epoch = epoch + 1
                    lowest_loss = total_score
                if eval_stats[0] > best_val_f1:
                    to_copy = True
                    best_f1_epoch = epoch + 1
                    best_val_f1 = eval_stats[0]
                    to_save = True
                self.log_eval_stats(eval_stats)
                self.logger.info("===> Last best F1 was {:.8f} in epoch {}".format(best_val_f1, best_f1_epoch))
                if not to_save:
                    return
                self.save_checkpoint({
                        'state_dict': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                    }, to_copy, epoch+1, self.args.save_path)

        # Start training and validation for nepochs
        for epoch in range(args.start_epoch, args.nepochs):
            if is_main_process():
                self.logger.info("\n => Start train set for EPOCH {}".format(epoch + 1))
                self.logger.info('lr is set to {}'.format(optimizer.param_groups[0]['lr']))
            
            if args.distributed:
                train_sampler.set_epoch(epoch)

            # Define container objects to keep track of multiple losses/metrics
            batch_time = AverageMeter()
            data_time = AverageMeter()         # compute FPS
            epoch_time = AverageMeter()
            
            loss = 0

            # Specify operation modules
            model.train()
            # compute timing
            end = time.time()
            epoch_time.start = end
            # Start training loop
            train_pbar = tqdm(total=len(train_loader), ncols=60)
            
            for i, extra_dict in enumerate(train_loader):
                train_pbar.update(1)
                data_time.update(time.time() - end)
                if gpu_available():
                    json_files = extra_dict.pop('idx_json_file')
                    for k, v in extra_dict.items():
                        extra_dict[k] = v.cuda()
                    image = extra_dict['image']
                image = image.contiguous().float()
                # Run model
                optimizer.zero_grad()

                output = model(image=image, extra_dict=extra_dict, is_training=True)
                
                loss, loss_info = self._log_training_loss(
                    output, epoch, step=i, data_loader=train_loader)

                train_pbar.set_postfix(loss=loss.item())
                
                if is_main_process():
                    self.writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
                
                # Setup backward pass
                loss.backward()

                # Clip gradients (usefull for instabilities or mistakes in ground truth)
                if args.clip_grad_norm != 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

                # update params
                optimizer.step()

                if args.lr_policy == 'cosine_warmup':
                    scheduler.step(epoch + i / len(train_loader))
                elif args.lr_policy == 'PolyLR':
                    scheduler.step()

                # Time trainig iteration
                batch_time.update(time.time() - end)
                end = time.time()

                # Print info
                if (i + 1) % args.print_freq == 0 and is_main_process():
                    self.logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Batch Time / Avg Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss:.8f} {loss_info}'.format(
                            epoch+1, i+1, len(train_loader), 
                            batch_time=batch_time, data_time=data_time,
                            loss=loss.item(), loss_info=loss_info))
            train_pbar.close()

            epoch_time.update(time.time() - epoch_time.start)

            if is_main_process():
                self.logger.info('Epoch time : {:.3f} hours.'.format(epoch_time.val / 60 / 60))
            
            # Adjust learning rate
            if args.lr_policy != 'cosine_warmup':
                scheduler.step()
            
            meet_eval_freq = args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0
            last_ep = (epoch == args.nepochs - 1)

            if meet_eval_freq or last_ep:
                loss_valid_list, eval_stats = self.validate(model)
                if eval_stats[0] >= best_val_f1:
                    self.logger.info(' >>> to save new best model at ep : %s with F1 %s' % ((epoch+1), eval_stats[0]))
                    save_cur_ckpt(loss, with_eval=True, eval_stats=eval_stats)
                elif last_ep:
                    self.logger.info(' >>> to save the last model at ep : %s with F1 %s' % ((epoch+1), eval_stats[0]))
                    save_cur_ckpt(loss, with_eval=True, eval_stats=eval_stats)
                else:
                    self.logger.info(' >>> skip model at ep : %s with lower F1 : %s' % ((epoch+1), eval_stats[0]))
            
                self.log_eval_stats(eval_stats)

            dist.barrier()
            torch.cuda.empty_cache()

        # at the end of training
        if not args.no_tb and is_main_process():
            self.writer.close()

    def _log_model_info(self, model):
        args = self.args
        if not is_main_process():
            return
        
        self.logger.info(40*"="+"\nArgs:{}\n".format(args)+40*"=")
        self.logger.info("Init model: '{}'".format(args.mod))
        self.logger.info("Number of parameters in model {} is {:.3f}M".format(args.mod, sum(tensor.numel() for tensor in model.parameters())/1e6))

    def _log_training_loss(self, output, epoch, step, data_loader):
        loss = 0.0
        loss_info = ''
        for k, v in output.items():
            if 'loss' in k:
                loss = loss + v
                loss_info = loss_info + '| %s:%.4f ' % (k, v.item() if isinstance(v, torch.Tensor) else v)
                if isinstance(v, torch.Tensor):
                    v = v.item()
                if is_main_process():
                    self.writer.add_scalar(k, v, epoch*len(data_loader) + step)
        return loss, loss_info

    def save_checkpoint(self, state, to_copy, epoch, save_path):
        if is_main_process():
            self.logger.info('Saving checkpoint to {}'.format(save_path))

            if to_copy:
                file_pre = f'model_best_epoch_{epoch}.pth.tar'
                self.logger.info('save the best model : %s' % epoch)
            else:
                file_pre = f'checkpoint_model_epoch_{epoch}.path.tar'

            filepath = os.path.join(save_path, file_pre)
            torch.save(state, filepath)

    def validate(self, model, **kwargs):
        args = self.args
        loader = self.valid_loader
        
        pred_lines_sub = []
        gt_lines_sub = []

        model.eval()

        # Start validation loop
        with torch.no_grad():
            val_pbar = tqdm(total=len(loader), ncols=50)
            
            for i, extra_dict in enumerate(loader):
                val_pbar.update(1)

                if not args.no_cuda:
                    json_files = extra_dict.pop('idx_json_file')
                    for k, v in extra_dict.items():
                        extra_dict[k] = v.cuda()
                    image = extra_dict['image']
                image = image.contiguous().float()
                
                output = model(image=image, extra_dict=extra_dict, is_training=False)
                all_line_preds = output['all_line_preds'] # in ground coordinate system
                all_cls_scores = output['all_cls_scores']

                all_line_preds = all_line_preds[-1]
                all_cls_scores = all_cls_scores[-1]
                num_el = all_cls_scores.shape[0]
                if 'cam_extrinsics' in extra_dict:
                    cam_extrinsics_all = extra_dict['cam_extrinsics']
                    cam_intrinsics_all = extra_dict['cam_intrinsics']
                else:
                    cam_extrinsics_all, cam_intrinsics_all = None, None

                # Print info
                if (i + 1) % args.print_freq == 0 and is_main_process():
                    self.logger.info('Test: [{0}/{1}]'.format(i+1, len(loader)))

                # Write results
                for j in range(num_el):
                    json_file = json_files[j]
                    if cam_extrinsics_all is not None:
                        extrinsic = cam_extrinsics_all[j].cpu().numpy()
                        intrinsic = cam_intrinsics_all[j].cpu().numpy()

                    with open(json_file, 'r') as file:
                        if 'apollo' in args.dataset_name:
                            json_line = json.loads(file.read())
                            if 'extrinsic' not in json_line:
                                json_line['extrinsic'] = extrinsic
                            if 'intrinsic' not in json_line:
                                json_line['intrinsic'] = intrinsic
                        else:
                            file_lines = [line for line in file]
                            json_line = json.loads(file_lines[0])

                    json_line['json_file'] = json_file
                    if 'once' in args.dataset_name:
                        if 'train' in json_file:
                            img_path = json_file.replace('train', 'data').replace('.json', '.jpg')
                        elif 'val' in json_file:
                            img_path = json_file.replace('val', 'data').replace('.json', '.jpg')
                        elif 'test' in json_file:
                            img_path = json_file.replace('test', 'data').replace('.json', '.jpg')
                        json_line["file_path"] = img_path

                    gt_lines_sub.append(copy.deepcopy(json_line))

                    # pred in ground
                    lane_pred = all_line_preds[j].cpu().numpy()
                    cls_pred = torch.argmax(all_cls_scores[j], dim=-1).cpu().numpy()
                    pos_lanes = lane_pred[cls_pred > 0]

                    if self.args.num_category > 1:
                        scores_pred = torch.softmax(all_cls_scores[j][cls_pred > 0], dim=-1).cpu().numpy()
                    else:
                        scores_pred = torch.sigmoid(all_cls_scores[j][cls_pred > 0]).cpu().numpy()

                    if pos_lanes.shape[0]:
                        lanelines_pred = []
                        lanelines_prob = []
                        xs = pos_lanes[:, 0:args.num_y_steps]
                        ys = np.tile(args.anchor_y_steps.copy()[None, :], (xs.shape[0], 1))
                        zs = pos_lanes[:, args.num_y_steps:2*args.num_y_steps]
                        vis = pos_lanes[:, 2*args.num_y_steps:]

                        for tmp_idx in range(pos_lanes.shape[0]):
                            cur_vis = vis[tmp_idx] > 0
                            cur_xs = xs[tmp_idx][cur_vis]
                            cur_ys = ys[tmp_idx][cur_vis]
                            cur_zs = zs[tmp_idx][cur_vis]

                            if cur_vis.sum() < 2:
                                continue

                            lanelines_pred.append([])
                            for tmp_inner_idx in range(cur_xs.shape[0]):
                                lanelines_pred[-1].append(
                                    [cur_xs[tmp_inner_idx],
                                     cur_ys[tmp_inner_idx],
                                     cur_zs[tmp_inner_idx]])
                            lanelines_prob.append(scores_pred[tmp_idx].tolist())
                    else:
                        lanelines_pred = []
                        lanelines_prob = []

                    json_line["pred_laneLines"] = lanelines_pred
                    json_line["pred_laneLines_prob"] = lanelines_prob

                    pred_lines_sub.append(copy.deepcopy(json_line))
                    img_path = json_line['file_path']
                    
                    if args.dataset_name == 'once':
                        self.save_eval_result_once(args, img_path, lanelines_pred, lanelines_prob)
            val_pbar.close()

            if 'openlane' in args.dataset_name:
                eval_stats = self.evaluator.bench_one_submit_ddp(
                    pred_lines_sub, gt_lines_sub, args.model_name,
                    args.pos_threshold, vis=False)
            elif 'once' in args.dataset_name:
                eval_stats = self.evaluator.lane_evaluation(
                    args.data_dir + 'val', '%s/once_pred/test' % (args.save_path),
                    args.eval_config_dir, args)
            elif 'apollo' in args.dataset_name:
                self.logger.info(' >>> eval mAP | [0.05, 0.95]')
                eval_stats = self.evaluator.bench_one_submit_ddp(
                    pred_lines_sub, gt_lines_sub,
                    args.model_name, args.pos_threshold, vis=False)
            else:
                assert False
                
            if any(name in args.dataset_name for name in ['openlane', 'apollo']):
                gather_output = [None for _ in range(args.world_size)]
                # all_gather all eval_stats and calculate mean
                dist.all_gather_object(gather_output, eval_stats)
                dist.barrier()
                eval_stats = self._recal_gpus_val(gather_output, eval_stats)

                loss_list = []
                return loss_list, eval_stats
            elif 'once' in args.dataset_name:
                loss_list = []
                return loss_list, eval_stats

    def _recal_gpus_val(self, gather_output, eval_stats):
        args = self.args

        apollo_metrics = {
            'r_lane': 0, 
            'p_lane': 0, 
            'cnt_gt': 0, 
            'cnt_pred': 0
        }
        openlane_metrics = {
            'r_lane': 0, 
            'p_lane': 0, 
            'c_lane': 0, 
            'cnt_gt': 0, 
            'cnt_pred': 0,
            'match_num': 0
        }

        if 'apollo' in self.args.dataset_name:
            # apollo no category accuracy.
            start_idx = 7
            gather_metrics = apollo_metrics
        else:
            start_idx = 8
            gather_metrics = openlane_metrics
        
        for i, k in enumerate(gather_metrics.keys()):
            gather_metrics[k] = np.sum(
                [eval_stats_sub[start_idx + i] for eval_stats_sub in gather_output])

        if gather_metrics['cnt_gt']!=0 :
            Recall = gather_metrics['r_lane'] / gather_metrics['cnt_gt']
        else:
            Recall = gather_metrics['r_lane'] / (gather_metrics['cnt_gt'] + 1e-6)
        if gather_metrics['cnt_pred'] !=0 :
            Precision = gather_metrics['p_lane'] / gather_metrics['cnt_pred']
        else:
            Precision = gather_metrics['p_lane'] / (gather_metrics['cnt_pred'] + 1e-6)
        if (Recall + Precision)!=0:
            f1_score = 2 * Recall * Precision / (Recall + Precision)
        else:
            f1_score = 2 * Recall * Precision / (Recall + Precision + 1e-6)
        
        if 'apollo' not in self.args.dataset_name:
            if gather_metrics['match_num']!=0:
                category_accuracy = gather_metrics['c_lane'] / gather_metrics['match_num']
            else:
                category_accuracy = gather_metrics['c_lane'] / (gather_metrics['match_num'] + 1e-6)
        
        eval_stats[0] = f1_score
        eval_stats[1] = Recall
        eval_stats[2] = Precision
        if self.is_apollo:
            err_start_idx = 3
        else:
            eval_stats[3] = category_accuracy
            err_start_idx = 4
        for i in range(4):
            err_idx = err_start_idx + i
            eval_stats[err_idx] = np.sum([eval_stats_sub[err_idx] for eval_stats_sub in gather_output]) / args.world_size
        return eval_stats

    def _get_model_from_cfg(self):
        args = self.args
        model = LATR(args)
        
        if args.sync_bn:
            if is_main_process():
                self.logger.info("Convert model with Sync BatchNorm")
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            
        if gpu_available():
            device = torch.device("cuda", args.local_rank)
            model = model.to(device)

        return model

    def _load_ckpt_from_workdir(self, model):
        args = self.args
        if args.eval_ckpt:
            best_file_name = args.eval_ckpt
        else:
            best_file_name = glob.glob(os.path.join(args.save_path, 'model_best*'))
            if len(best_file_name) > 0:
                best_file_name = best_file_name[0]
            else:
                best_file_name = ''
        if os.path.isfile(best_file_name):
            checkpoint = torch.load(best_file_name)
            if is_main_process():
                self.logger.info("=> loading checkpoint '{}'".format(best_file_name))
                model.load_state_dict(checkpoint['state_dict'])
        else:
            self.logger.info("=> no checkpoint found at '{}'".format(best_file_name))

    def eval(self):
        self.logger.info('>>>>>  start eval <<<<< \n')
        args = self.args
        
        model = self._get_model_from_cfg()
        self._load_ckpt_from_workdir(model)

        dist.barrier()
        # DDP setting
        if args.distributed:
            model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        _, eval_stats = self.validate(model)

        if is_main_process() and (eval_stats is not None):
            self.log_eval_stats(eval_stats)

    def _get_train_dataset(self):
        args = self.args
        if 'openlane' in args.dataset_name:
            train_dataset = LaneDataset(args.dataset_dir, args.data_dir + 'training/', args, data_aug=True)

        elif 'once' in args.dataset_name:
            train_dataset = LaneDataset(args.dataset_dir, ops.join(args.data_dir, 'train/'), args, data_aug=True)
        else:
            self.logger.info('using Apollo Dataset')
            train_dataset = ApolloLaneDataset(args.dataset_dir, ops.join(args.data_dir, 'train.json'), args, data_aug=True)
        
        train_loader, train_sampler = get_loader(train_dataset, args)

        return train_dataset, train_loader, train_sampler

    def _get_model_ddp(self):
        args = self.args
        # define network
        model = LATR(args)
        
        # if args.sync_bn:
        if is_main_process():
            self.logger.info("Convert model with Sync BatchNorm")
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
        if gpu_available():
            # Load model on gpu before passing params to optimizer
            device = torch.device("cuda", args.local_rank)
            model = model.to(device)

        """
            first load param to model, then model = DDP(model)
        """

        # resume model
        args.resume = first_run(args.save_path)

        model, best_epoch, lowest_loss, best_f1_epoch, best_val_f1, \
            optim_saved_state, schedule_saved_state = self.resume_model(model)
        dist.barrier()
        # DDP setting
        if args.distributed:
            model = DDP(
                model, device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True
            )

        # Define optimizer and scheduler
        optimizer = build_optimizer(
            model,
            args.optimizer_cfg)
        scheduler = define_scheduler(
            optimizer, args, dataset_size=len(self.train_loader))

        return model, optimizer, scheduler, best_epoch, lowest_loss, best_f1_epoch, best_val_f1

    def resume_model(self, model, path=''):
        args = self.args
        
        best_epoch = 0
        lowest_loss = np.inf
        best_f1_epoch = 0
        best_val_f1 = -1e-5
        optim_saved_state = None
        schedule_saved_state = None
            
        if len(path) == 0 and args.resume:
            # try the latest ckpt
            path = os.path.join(args.save_path, 'checkpoint_model_epoch_{}.pth.tar'.format(int(args.resume)))
            # try the best ckpt saved
            if not os.path.isfile(path):
                path = os.path.join(args.save_path, f'model_best_epoch_{args.resume}.pth.tar')
            
        if os.path.isfile(path):
            self.logger.info("=> loading checkpoint from {}".format(path))
            checkpoint = torch.load(path, map_location='cpu')
            if is_main_process():
                model.load_state_dict(checkpoint['state_dict'])
                self.logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, args.start_epoch))
            
            optim_saved_state = checkpoint['optimizer']
            schedule_saved_state = checkpoint['scheduler']
            
            args.start_epoch = int(args.resume)
        else:
            if is_main_process():
                self.logger.info("=> Warning: no checkpoint found at '{}'".format(path))
            
        return model, best_epoch, lowest_loss, best_f1_epoch, best_val_f1, optim_saved_state, schedule_saved_state

    def _get_valid_dataset(self):
        args = self.args
        if 'openlane' in args.dataset_name:
            if not args.evaluate_case:
                valid_dataset = LaneDataset(args.dataset_dir, args.data_dir + 'validation/', args)
            else:
                # TODO eval case
                valid_dataset = LaneDataset(args.dataset_dir, args.data_dir + 'test/up_down_case/', args)

        elif 'once' in args.dataset_name:
            valid_dataset = LaneDataset(args.dataset_dir, ops.join(args.data_dir, 'val/'), args)
        else:
            valid_dataset = ApolloLaneDataset(args.dataset_dir, os.path.join(args.data_dir, 'test.json'), args)

        valid_loader, valid_sampler = get_loader(valid_dataset, args)
        return valid_dataset, valid_loader, valid_sampler

    def save_eval_result_once(self, args, img_path, lanelines_pred, lanelines_prob):
        # 3d eval result
        result = {}
        result_dir = os.path.join(args.save_path, 'once_pred/')
        mkdir_if_missing(result_dir)
        result_dir = os.path.join(result_dir, 'test/')
        mkdir_if_missing(result_dir)
        file_path_splited = img_path.split('/')
        result_dir = os.path.join(result_dir, file_path_splited[-3]) # sequence
        mkdir_if_missing(result_dir)
        result_dir = os.path.join(result_dir, 'cam01/')
        mkdir_if_missing(result_dir)
        result_file_path = ops.join(result_dir, file_path_splited[-1][:-4]+'.json')

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

        # write lane result
        lane_lines = []
        for k in range(len(lanelines_pred)):
            lane = np.array(lanelines_pred[k])
            lane = np.flip(lane, axis=0)
            lane = lane.T
            lane = np.vstack((lane, np.ones((1, lane.shape[1]))))
            lane = np.matmul(np.linalg.inv(cam_extrinsics), lane)
            lane = lane[0:3,:].T
            lane_lines.append({'points': lane.tolist(),
                               'score': np.max(lanelines_prob[k])})
        result['lanes'] = lane_lines

        with open(result_file_path, 'w') as result_file:
            json.dump(result, result_file)

    def log_eval_stats(self, eval_stats):
        if self.is_apollo:
            return self._log_genlane_eval_info(eval_stats)

        if is_main_process():
            self.logger.info("===> Evaluation laneline F-measure: {:.8f}".format(eval_stats[0]))
            self.logger.info("===> Evaluation laneline Recall: {:.8f}".format(eval_stats[1]))
            self.logger.info("===> Evaluation laneline Precision: {:.8f}".format(eval_stats[2]))
            self.logger.info("===> Evaluation laneline Category Accuracy: {:.8f}".format(eval_stats[3]))
            self.logger.info("===> Evaluation laneline x error (close): {:.8f} m".format(eval_stats[4]))
            self.logger.info("===> Evaluation laneline x error (far): {:.8f} m".format(eval_stats[5]))
            self.logger.info("===> Evaluation laneline z error (close): {:.8f} m".format(eval_stats[6]))
            self.logger.info("===> Evaluation laneline z error (far): {:.8f} m".format(eval_stats[7]))

    def _log_genlane_eval_info(self, eval_stats):
        if is_main_process():
            self.logger.info("===> Evaluation on validation set: \n"
                "laneline F-measure {:.8} \n"
                "laneline Recall  {:.8} \n"
                "laneline Precision  {:.8} \n"
                "laneline x error (close)  {:.8} m\n"
                "laneline x error (far)  {:.8} m\n"
                "laneline z error (close)  {:.8} m\n"
                "laneline z error (far)  {:.8} m\n".format(eval_stats[0], eval_stats[1],
                                                            eval_stats[2], eval_stats[3],
                                                            eval_stats[4], eval_stats[5],
                                                            eval_stats[6]))


def set_work_dir(cfg):
    # =========output path========== #
    save_prefix = osp.join(os.getcwd(), 'work_dirs')
    save_root = osp.join(save_prefix, cfg.output_dir)

    # cur work dirname
    cfg_path = Path(cfg.config)

    if cfg.mod is None:
        cfg.mod = os.path.join(cfg_path.parent.name, cfg_path.stem)
    
    save_ppath = Path(save_root, cfg.mod)
    save_ppath.mkdir(parents=True, exist_ok=True)

    cfg.save_path = save_ppath.as_posix()
    cfg.save_json_path = cfg.save_path
    
    seg_output_dir = Path(cfg.save_path, 'seg_vis')
    seg_output_dir.mkdir(parents=True, exist_ok=True)

    # cp config into cur_work_dir
    shutil.copy(cfg_path.as_posix(), cfg.save_path)
    