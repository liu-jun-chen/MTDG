import random
import itertools
import glob
import os
import numpy as np
import torch
from torch import nn
from torch.cuda import amp
from torchpack.train import Trainer
from torchpack.utils.typing import Optimizer, Scheduler

import time
from typing import Any, Dict, List, Optional, Callable
from torch.utils.data import DataLoader

# from torchpack.callbacks import (Callback, Callbacks)
from torchpack.train.exception import StopTraining
from torchpack.train.summary import Summary
from torchpack.utils import humanize
from torchpack.utils.logging import logger
from torchpack.utils.config import configs
from torchpack.utils import fs, io
from core.callbacks import MeanIoU
from core.inference import InferenceRunner
import pdb
import tqdm
from collections import OrderedDict, deque
import operator
from numbers import Number
import copy
from torchpack.callbacks import (Callback, Callbacks, ConsoleWriter,
                                 EstimatedTimeLeft, JSONLWriter, MetaInfoSaver,
                                 ProgressBar, TFEventWriter)
from torch.utils.data import DataLoader, DistributedSampler

from torch.utils.tensorboard import SummaryWriter

from itertools import chain
from core.algorithms import HGP

import statistics
import torch.autograd as autograd

__all__ = ['SemanticSTFTrainer']

def cosine_similarity(gra1, gra2):
    dot_product = torch.dot(gra1, gra2)
    norm1 = torch.norm(gra1)
    norm2 = torch.norm(gra2)
    return dot_product / (norm1 * norm2)


class SemanticSTFTrainer(Trainer):

    def __init__(self,
                 model: nn.Module,
                 criterion: Callable,
                 optimizer: Optimizer,
                 scheduler: Scheduler,
                 num_workers: int,
                 seed: int,
                 amp_enabled: bool = False) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_workers = num_workers
        self.seed = seed
        self.amp_enabled = amp_enabled
        self.scaler = amp.GradScaler(enabled=self.amp_enabled)
        self.epoch_num = 1
        self.epoch_number = 0
        self.eval_interval = 500
        self.writer = None


        self.lamda = 0.1
        self.T = 0.07
        self.miou = None
        self.weight_deque = deque(maxlen=10)
        self.miou_deque = deque(maxlen=10)
        self.sim_avg_deque_4 = deque(maxlen=4)
        self.sim_avg_deque_16 = deque(maxlen=16)
        self.sim_avg_deque_64 = deque(maxlen=64)
        self.sim_mean = 0
        self.n = 0

    def _before_train(self) -> None:
        dir = os.path.join(configs.run_dir, 'checkpoints')
        load_dir = fs.normpath(dir)
        checkpoints = glob.glob(os.path.join(load_dir, 'step-*.pt'))
        if not checkpoints:
            logger.warning(f'No checkpoints found: "{load_dir}".')
            return

        load_path = max(checkpoints, key=os.path.getmtime)

        self._load_previous_checkpoint(load_path)


    def _before_epoch(self) -> None:
        self.model.train()
        self.dataflow_kitti.sampler.set_epoch(self.epoch_num - 1)
        self.dataflow_synlidar.sampler.set_epoch(self.epoch_num - 1)
        self.dataflow_poss.sampler.set_epoch(self.epoch_num - 1)

        self.dataflow_kitti.worker_init_fn = lambda worker_id: np.random.seed(
            self.seed + (self.epoch_num - 1) * self.num_workers + worker_id)
        self.dataflow_synlidar.worker_init_fn = lambda worker_id: np.random.seed(
            self.seed + (self.epoch_num - 1) * self.num_workers + worker_id)
        self.dataflow_poss.worker_init_fn = lambda worker_id: np.random.seed(
            self.seed + (self.epoch_num - 1) * self.num_workers + worker_id)
        
    def before_epoch(self) -> None:
        self.epoch_number += 1
        if isinstance(self.dataflow_kitti, DataLoader) and \
                isinstance(self.dataflow_kitti.sampler, DistributedSampler):
            self.dataflow_kitti.sampler.set_epoch(self.epoch_num)
        if isinstance(self.dataflow_synlidar, DataLoader) and isinstance(self.dataflow_synlidar.sampler, DistributedSampler):
            self.dataflow_synlidar.sampler.set_epoch(self.epoch_num)
        if isinstance(self.dataflow_poss, DataLoader) and isinstance(self.dataflow_poss.sampler, DistributedSampler):
            self.dataflow_poss.sampler.set_epoch(self.epoch_num)

        log_dir = "logs/gradient_match_corrected_0.15" + str(self.epoch_number)   # You can change this to your preferred log directory
        self.writer = SummaryWriter(log_dir)
    
        self._before_epoch()
        self.callbacks.before_epoch()

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:
        _inputs_kitti = {}
        _inputs_synlidar = {}
        _inputs_poss = {}
        total_loss = torch.tensor(0.0).cuda()



        if feed_dict['kitti'] != None:
            for key, value in feed_dict['kitti'].items():
                if 'name' not in key and 'ids' not in key:
                    _inputs_kitti[key] = value.cuda()

            inputs_lidar_kitti = _inputs_kitti['lidar']
            # targets_kitti = feed_dict['kitti']['targets'].F.long().cuda(non_blocking=True)
            targets_kitti_1 = _inputs_kitti['targets'].F.long().cuda(non_blocking=True)

            with amp.autocast(enabled=self.amp_enabled):
                output_kitti, feat_kitti = self.model(inputs_lidar_kitti)
                if output_kitti.requires_grad:
                    loss_kitti = self.criterion(output_kitti, targets_kitti_1)

            invs_kitti = feed_dict['kitti']['inverse_map_dense']
            all_labels_kitti = feed_dict['kitti']['targets_mapped']
            _outputs_kitti = []
            _targets_kitti = []
            for idx in range(invs_kitti.C[:, -1].max() + 1):
                cur_scene_pts = (inputs_lidar_kitti.C[:, -1] == idx).cpu().numpy()
                cur_inv = invs_kitti.F[invs_kitti.C[:, -1] == idx].cpu().numpy()
                cur_label = (all_labels_kitti.C[:, -1] == idx).cpu().numpy()
                outputs_mapped = output_kitti[cur_scene_pts][cur_inv].argmax(1)
                targets_mapped = all_labels_kitti.F[cur_label]
                _outputs_kitti.append(outputs_mapped)
                _targets_kitti.append(targets_mapped)
            outputs_kitti = torch.cat(_outputs_kitti, 0)
            targets_kitti = torch.cat(_targets_kitti, 0)
                
        
        if feed_dict['synlidar'] != None:
            for key, value in feed_dict['synlidar'].items():
                if 'name' not in key and 'ids' not in key:
                    _inputs_synlidar[key] = value.cuda()
            
            inputs_lidar_synlidar = _inputs_synlidar['lidar']
            # targets_synlidar = feed_dict['synlidar']['targets'].F.long().cuda(non_blocking=True)
            targets_synlidar_1 = _inputs_synlidar['targets'].F.long().cuda(non_blocking=True)

            with amp.autocast(enabled=self.amp_enabled):
                output_synlidar, feat_synlidar = self.model(inputs_lidar_synlidar)
                if output_synlidar.requires_grad:
                    loss_synlidar = self.criterion(output_synlidar, targets_synlidar_1)

            invs_synlidar = feed_dict['synlidar']['inverse_map_dense']
            all_labels_synlidar = feed_dict['synlidar']['targets_mapped']
            _outputs_synlidar = []
            _targets_synlidar = []
            for idx in range(invs_synlidar.C[:, -1].max() + 1):
                cur_scene_pts = (inputs_lidar_synlidar.C[:, -1] == idx).cpu().numpy()
                cur_inv = invs_synlidar.F[invs_synlidar.C[:, -1] == idx].cpu().numpy()
                cur_label = (all_labels_synlidar.C[:, -1] == idx).cpu().numpy()
                outputs_mapped = output_synlidar[cur_scene_pts][cur_inv].argmax(1)
                targets_mapped = all_labels_synlidar.F[cur_label]
                _outputs_synlidar.append(outputs_mapped)
                _targets_synlidar.append(targets_mapped)
            outputs_synlidar = torch.cat(_outputs_synlidar, 0)
            targets_synlidar = torch.cat(_targets_synlidar, 0)



        if feed_dict['poss'] != None:
            for key, value in feed_dict['poss'].items():
                if 'name' not in key and 'ids' not in key:
                    _inputs_poss[key] = value.cuda()

            inputs_lidar_poss = _inputs_poss['lidar']
            # targets_poss = feed_dict['poss']['targets'].F.long().cuda(non_blocking=True)
            targets_poss_1 = _inputs_poss['targets'].F.long().cuda(non_blocking=True)

            with amp.autocast(enabled=self.amp_enabled):
                output_poss, feat_poss = self.model(inputs_lidar_poss)
                if output_poss.requires_grad:
                    loss_poss = self.criterion(output_poss, targets_poss_1)

            invs_poss = feed_dict['poss']['inverse_map_dense']
            all_labels_poss = feed_dict['poss']['targets_mapped']
            _outputs_poss = []
            _targets_poss = []
            for idx in range(invs_poss.C[:, -1].max() + 1):
                cur_scene_pts = (inputs_lidar_poss.C[:, -1] == idx).cpu().numpy()
                cur_inv = invs_poss.F[invs_poss.C[:, -1] == idx].cpu().numpy()
                cur_label = (all_labels_poss.C[:, -1] == idx).cpu().numpy()
                outputs_mapped = output_poss[cur_scene_pts][cur_inv].argmax(1)
                targets_mapped = all_labels_poss.F[cur_label]
                _outputs_poss.append(outputs_mapped)
                _targets_poss.append(targets_mapped)
            outputs_poss = torch.cat(_outputs_poss, 0)
            targets_poss = torch.cat(_targets_poss, 0)

        kitti_gra = autograd.grad(loss_kitti, self.model.classifier.parameters(),create_graph=True, retain_graph=True)
        synlidar_gra = autograd.grad(loss_synlidar, self.model.classifier.parameters(),create_graph=True, retain_graph=True)
        poss_gra = autograd.grad(loss_poss, self.model.classifier.parameters(),create_graph=True, retain_graph=True)
        kitti_gra_tensor = torch.cat([grad.view(-1) for grad in kitti_gra])
        synlidar_gra_tensor = torch.cat([grad.view(-1) for grad in synlidar_gra])
        poss_gra_tensor = torch.cat([grad.view(-1) for grad in poss_gra])

        similarity_kitti_synlidar = cosine_similarity(kitti_gra_tensor, synlidar_gra_tensor)
        similarity_kitti_poss = cosine_similarity(kitti_gra_tensor, poss_gra_tensor)
        similarity_synlidar_poss = cosine_similarity(synlidar_gra_tensor, poss_gra_tensor)
        average_similarity = (similarity_kitti_synlidar + similarity_kitti_poss + similarity_synlidar_poss) / 3

        # self.sim_avg_deque_4.append(average_similarity.item())
        # self.sim_avg_deque_16.append(average_similarity.item())
        # self.sim_avg_deque_64.append(average_similarity.item())
        # if self.local_step > 64:
        #     self.writer.add_scalar('sim_avg_4', statistics.mean(self.sim_avg_deque_4), self.local_step)
        #     self.writer.add_scalar('sim_avg_16', statistics.mean(self.sim_avg_deque_16), self.local_step)
        #     self.writer.add_scalar('sim_avg_64', statistics.mean(self.sim_avg_deque_64), self.local_step)

        # if average_similarity.item() < 0:
        #     pass
        # else:
        #     self.n += 1
        #     self.sim_mean = (self.sim_mean * (self.n - 1) + average_similarity.item()) / self.n
        # beta = 0.8
        # self.sim_mean = beta * self.sim_mean + (1 - beta) * similarity_synlidar_poss.item()
        self.writer.add_scalar('ks_similarity', similarity_kitti_synlidar.item(), self.local_step)
        self.writer.add_scalar('kp_similarity', similarity_kitti_poss.item(), self.local_step)
        self.writer.add_scalar('sp_similarity', similarity_synlidar_poss.item(), self.local_step)
        self.writer.add_scalar('avg_similarity', average_similarity.item(), self.local_step)
        # self.writer.add_scalar('similarity mean', self.sim_mean, self.local_step)


        # losses = torch.cat((loss_kitti, loss_synlidar, loss_poss), 1)
        losses = torch.stack([loss_kitti, loss_synlidar, loss_poss], dim=0)


        # if self.local_step > 10 and statistics.variance(self.miou_deque) > 0.015:
        if self.local_step > 10 and average_similarity > 0.1:
            miou_array = np.array(self.miou_deque)
            miou_max_index = np.argmax(miou_array)
            weight = self.weight_deque[miou_max_index]
            perturb_factors = (1.2, 0.8)
            perturb = np.random.choice(perturb_factors)
            weight = torch.tensor(weight * perturb)
            weight = torch.nn.functional.softmax(weight, dim=-1).cuda()
            self.weight_deque.append(weight)
        else:
            weight = torch.nn.functional.softmax(torch.randn(3), dim=-1).cuda()
            self.weight_deque.append(weight)
        total_loss = torch.sum(losses * weight)
        # total_loss = 0.2*loss_poss + 0.3*loss_kitti + 0.5*loss_synlidar



        if total_loss.requires_grad:

            self.writer.add_scalar('Kitti Loss', loss_kitti.item(), self.local_step)
            self.writer.add_scalar('Synlidar Loss', loss_synlidar.item(), self.local_step)
            self.writer.add_scalar('Poss Loss', loss_poss.item(), self.local_step)

            self.summary.add_scalar('Total Loss', total_loss.item())

            self.optimizer.zero_grad()
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            return {'outputs_kitti': outputs_kitti, 'targets_kitti': targets_kitti,
                'outputs_synlidar': outputs_synlidar, 'targets_synlidar': targets_synlidar,
                'outputs_poss': outputs_poss, 'targets_poss': targets_poss}


    def run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:
        output_dict = self._run_step(feed_dict)
        return output_dict  

    def run_step_inference(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:
        _inputs = {}
        for key, value in feed_dict.items():
            if 'name' not in key and 'ids' not in key:
                _inputs[key] = value.cuda()

        inputs_1 = _inputs['lidar']
        targets_1 = feed_dict['targets'].F.long().cuda(non_blocking=True)
        
        with amp.autocast(enabled=self.amp_enabled):
            outputs_1, feat_1 = self.model(inputs_1)
            # loss = None
            if outputs_1.requires_grad:
                loss_1 = self.criterion(outputs_1, targets_1)

        invs = feed_dict['inverse_map']
        all_labels = feed_dict['targets_mapped']
        _outputs = []
        _targets = []
        for idx in range(invs.C[:, -1].max() + 1):
            cur_scene_pts = (inputs_1.C[:, -1] == idx).cpu().numpy()
            cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
            cur_label = (all_labels.C[:, -1] == idx).cpu().numpy()
            outputs_mapped = outputs_1[cur_scene_pts][cur_inv].argmax(1)
            targets_mapped = all_labels.F[cur_label]
            _outputs.append(outputs_mapped)
            _targets.append(targets_mapped)
        outputs = torch.cat(_outputs, 0)
        targets = torch.cat(_targets, 0)
        output_dict = {'outputs': outputs, 'targets': targets}
        return output_dict

    def _after_epoch(self) -> None:
        self.model.eval()

    def _state_dict(self) -> Dict[str, Any]:
        state_dict = {}
        state_dict['model'] = self.model.state_dict()
        state_dict['scaler'] = self.scaler.state_dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['scheduler'] = self.scheduler.state_dict()
        return state_dict

    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict['model'])
        self.scaler.load_state_dict(state_dict.pop('scaler'))
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])

    def _load_previous_checkpoint(self, checkpoint_path: str) -> None:
        state_dict = io.load(checkpoint_path, map_location='cpu')
        self.load_state_dict(state_dict)

    def train_with_defaults(self,
                            dataflow_kitti: DataLoader,
                            dataflow_synlidar: DataLoader,
                            dataflow_poss: DataLoader,
                            *,
                            num_epochs: int = 9999999,
                            callbacks: Optional[List[Callback]] = None
                            ) -> None:
        if callbacks is None:
            callbacks = []
        callbacks += [
            MetaInfoSaver(),
            ConsoleWriter(),
            TFEventWriter(),
            JSONLWriter(),
            ProgressBar(),
            EstimatedTimeLeft()
        ]
        self.train(dataflow_kitti=dataflow_kitti,
                   dataflow_synlidar=dataflow_synlidar,
                   dataflow_poss=dataflow_poss,
                   num_epochs=num_epochs,
                   callbacks=callbacks)

    def train(self,
              dataflow_kitti: DataLoader,
              dataflow_synlidar: DataLoader,
              dataflow_poss: DataLoader,
              *,
              num_epochs: int = 9999999,
              callbacks: Optional[List[Callback]] = None) -> None:
        self.dataflow_kitti = dataflow_kitti
        self.dataflow_synlidar = dataflow_synlidar
        self.dataflow_poss = dataflow_poss
        self.steps_per_epoch = max(len(self.dataflow_kitti), len(self.dataflow_synlidar), len(self.dataflow_poss))
        self.num_epochs = num_epochs

        if callbacks is None:
            callbacks = []
        self.callbacks = Callbacks(callbacks)
        self.summary = Summary()

        try:
            self.callbacks.set_trainer(self)
            self.summary.set_trainer(self)

            self.epoch_num = 0
            self.global_step = 0

            train_time = time.perf_counter()
            self.before_train()

            while self.epoch_num < self.num_epochs:
                self.epoch_num += 1
                self.local_step = 0

                logger.info('Epoch {}/{} started.'.format(
                    self.epoch_num, self.num_epochs))
                epoch_time = time.perf_counter()
                self.before_epoch()

                # Convert self.dataflow_kitti, self.dataflow_synlidar, and self.dataflow_poss to iterators
                dataflow_kitti_iter = iter(self.dataflow_kitti)
                dataflow_synlidar_iter = iter(self.dataflow_synlidar)
                dataflow_poss_iter = iter(self.dataflow_poss)

                # mIoU = MeanIoU(name=f'iou/test_', num_classes=11, ignore_label=255)
                # mIoU.before_epoch()

                for _ in range(max(len(self.dataflow_kitti), len(self.dataflow_synlidar), len(self.dataflow_poss))):
                    self.global_step += 1      

                    # Kitti data
                    self.local_step += 1
                    try:
                        feed_dict_kitti = next(dataflow_kitti_iter)
                    except StopIteration:
                        # Handle the end of the iterator or resample if needed
                        dataflow_kitti_iter = iter(self.dataflow_kitti)
                        feed_dict_kitti = next(dataflow_kitti_iter)

                    # self.before_step(feed_dict_kitti)
                    # output_dict_kitti = self.run_step(feed_dict_kitti)
                    # self.after_step(output_dict_kitti)
                    # self.trigger_step()

                    # Synlidar data  
                    # self.local_step += 1
                    try:
                        feed_dict_synlidar = next(dataflow_synlidar_iter)
                    except StopIteration:
                        # Handle the end of the iterator or resample if needed
                        dataflow_synlidar_iter = iter(self.dataflow_synlidar)
                        feed_dict_synlidar = next(dataflow_synlidar_iter)

                    # self.before_step(feed_dict_synlidar)
                    # output_synlidar = self.run_step(feed_dict_synlidar)
                    # self.after_step(output_synlidar)
                    # self.trigger_step()

                    # SemanticPOSS data
                    # self.local_step += 1
                    try:
                        feed_dict_poss = next(dataflow_poss_iter)
                    except StopIteration:
                        # Handle the end of the iterator or resample if needed
                        dataflow_poss_iter = iter(self.dataflow_poss)
                        feed_dict_poss = next(dataflow_poss_iter)
                    
                    feed_dict = {}
                    feed_dict['kitti'] = feed_dict_kitti
                    feed_dict['synlidar'] = feed_dict_synlidar
                    feed_dict['poss'] = feed_dict_poss

                    self.before_step(feed_dict)
                    output_dict = self.run_step(feed_dict)
                    self.after_step(output_dict)
                    self.trigger_step()
                    self.miou = evaluate(output_dict, self.model)
                    self.miou_deque.append(self.miou)
                    # miou = torch.tensor(miou)
                    # print(miou)
                    self.writer.add_scalar('miou', self.miou, self.local_step)
                    # self.summary.add_scalar("mIoU_train", miou.item())
                    # self.summary.add_scalar('Total Loss', total_loss.item())

                    


                    # self.writer.add_scalar('miou', step_miou, self.local_step)

                epoch_number = self.epoch_num

                self.after_epoch()
                logger.info('Training finished in {}.'.format(humanize.naturaldelta(time.perf_counter() - epoch_time)))

                self.trigger_epoch()
                logger.info('Epoch finished in {}.'.format(humanize.naturaldelta(time.perf_counter() - epoch_time)))

            logger.success('{} epochs of training finished in {}.'.format(self.num_epochs, humanize.naturaldelta(time.perf_counter() - train_time)))
        except StopTraining as e:
            logger.info('Training was stopped by {}.'.format(str(e)))
        finally:
            self.after_train()

    def save_model(self, path):
        assert '.pt' in path, "Checkpoint save path is wrong"
        state_dict = dict()
        state_dict['model'] = self.model.state_dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['scheduler'] = self.scheduler.state_dict()
        torch.save(state_dict, path)


def evaluate(val_loader, model):
    mIoU = MeanIoU(name=f'iou/test_', num_classes=11, ignore_label=255)
    mIoU.before_epoch()

    # with torch.no_grad():
        # for feed_dict in tqdm.tqdm(val_loader, ncols=0):
        #     _inputs = dict()
        #     for key, value in feed_dict.items():
        #         if not 'name' in key:
        #             _inputs[key] = value.cuda()
        #     inputs = _inputs['lidar']
        #     # targets = feed_dict['targets'].F.long().cuda(non_blocking=True)
        #     outputs = model(inputs)

        #     invs = feed_dict['inverse_map']
        #     all_labels = feed_dict['targets_mapped']
        #     _outputs = []
        #     _targets = []
        #     for idx in range(invs.C[:, -1].max() + 1):
        #         cur_scene_pts = (inputs.C[:, -1] == idx).cpu().numpy()
        #         cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
        #         cur_label = (all_labels.C[:, -1] == idx).cpu().numpy()
        #         outputs_mapped = outputs[cur_scene_pts][cur_inv].argmax(1)
        #         targets_mapped = all_labels.F[cur_label]
        #         _outputs.append(outputs_mapped)
        #         _targets.append(targets_mapped)
        #     outputs = torch.cat(_outputs, 0)
        #     targets = torch.cat(_targets, 0)
        #     assert not outputs.requires_grad, "produced grad, wrong"
        #     output_dict = {'outputs': outputs, 'targets': targets}
    mIoU.after_step(val_loader)
    mIoU.trigger_step()
    mIoU.after_epoch()
    miou = mIoU.step_miou
    return miou

            