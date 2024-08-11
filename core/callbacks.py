from typing import Any, Dict

import numpy as np
import torch
from torchpack import distributed as dist
from torchpack.callbacks.callback import Callback

__all__ = ['MeanIoU']


class MeanIoU(Callback):

    def __init__(self,
                 num_classes: int,
                 ignore_label: int,
                 *,
                 output_tensor: str = 'outputs',
                 target_tensor: str = 'targets',
                 name: str = 'iou') -> None:
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.name = name
        self.output_tensor = output_tensor
        self.target_tensor = target_tensor
        self.miou = None
        self.step_miou = None

    def _before_epoch(self) -> None:
        self.total_seen = np.zeros(self.num_classes)
        self.total_correct = np.zeros(self.num_classes)
        self.total_positive = np.zeros(self.num_classes)

        self.step_seen = np.zeros(self.num_classes)
        self.step_correct = np.zeros(self.num_classes)
        self.step_positive = np.zeros(self.num_classes)

    def _after_step(self, output_dict: Dict[str, Any]) -> None:
        if len(output_dict) < 3:
            outputs = output_dict[self.output_tensor]
            targets = output_dict[self.target_tensor]
            outputs = outputs[targets != self.ignore_label]
            targets = targets[targets != self.ignore_label]

            for i in range(self.num_classes):
                self.step_seen[i] = torch.sum(targets == i).item()
                # print('this is after_step', str(self.step_seen[i]))
                self.step_correct[i] = torch.sum((targets == i)
                                                & (outputs == targets)).item()
                self.step_positive[i] = torch.sum(outputs == i).item()

                self.total_seen[i] += self.step_seen[i]
                self.total_correct[i] += self.step_correct[i]
                self.total_positive[i] += self.step_positive[i]
        else:
            outputs_kitti = output_dict['outputs_kitti']
            targets_kitti = output_dict['targets_kitti']
            outputs_kitti = outputs_kitti[targets_kitti != self.ignore_label]
            targets_kitti = targets_kitti[targets_kitti != self.ignore_label]
            outputs_synlidar = output_dict['outputs_synlidar']
            targets_synlidar = output_dict['targets_synlidar']
            outputs_synlidar = outputs_synlidar[targets_synlidar != self.ignore_label]
            targets_synlidar = targets_synlidar[targets_synlidar != self.ignore_label]
            outputs_poss = output_dict['outputs_poss']
            targets_poss = output_dict['targets_poss']
            outputs_poss = outputs_poss[targets_poss != self.ignore_label]
            targets_poss = targets_poss[targets_poss != self.ignore_label]

            # for i in range(self.num_classes):
            #     self.step_seen[i] = torch.sum(targets_kitti == i, targets_synlidar == i, targets_poss == i).item()
            #     # print('this is after_step', str(self.step_seen[i]))
            #     self.step_correct[i] = torch.sum((targets_kitti == i) & (outputs_kitti == targets_kitti), (targets_synlidar == i) & (outputs_synlidar == targets_synlidar), (targets_poss == i) & (outputs_poss == targets_poss)).item()
            #     self.step_positive[i] = torch.sum(outputs_kitti == i, outputs_synlidar == i, outputs_poss == i).item()

            #     self.total_seen[i] += self.step_seen[i]
            #     self.total_correct[i] += self.step_correct[i]
            #     self.total_positive[i] += step_positive[i]

            for i in range(self.num_classes):
                self.step_seen[i] = torch.sum(targets_kitti == i).item() + torch.sum(targets_synlidar == i).item() + torch.sum(targets_poss == i).item()
                self.step_correct[i] = (torch.sum((targets_kitti == i) & (outputs_kitti == targets_kitti)).item() +
                                        torch.sum((targets_synlidar == i) & (outputs_synlidar == targets_synlidar)).item() +
                                        torch.sum((targets_poss == i) & (outputs_poss == targets_poss)).item())
                self.step_positive[i] = (torch.sum(outputs_kitti == i).item() +
                                        torch.sum(outputs_synlidar == i).item() +
                                        torch.sum(outputs_poss == i).item())

                self.total_seen[i] += self.step_seen[i]
                self.total_correct[i] += self.step_correct[i]
                self.total_positive[i] += self.step_positive[i]

    
    def _trigger_step(self) -> None:
        ious = []

        for i in range(self.num_classes):
            if self.step_seen[i] == 0:
                ious.append(1)
            else:
                cur_iou = self.step_correct[i] / (self.step_seen[i]
                                                   + self.step_positive[i]
                                                   - self.step_correct[i])
                ious.append(cur_iou)

        self.step_miou = np.mean(ious)
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'summary'):
            self.trainer.summary.add_scalar('Train mIoU', self.step_miou)
        # if hasattr(self, 'trainer') and hasattr(self.trainer, 'summary'):
        #     self.trainer.writer.add_scalar('miou', self.step_miou, self.trainer.local_step)
        # print(self.step_miou)
        # return step_miou

    def _after_epoch(self) -> None:
        for i in range(self.num_classes):
            self.total_seen[i] = dist.allreduce(self.total_seen[i],
                                                reduction='sum')
            self.total_correct[i] = dist.allreduce(self.total_correct[i],
                                                   reduction='sum')
            self.total_positive[i] = dist.allreduce(self.total_positive[i],
                                                    reduction='sum')

        ious = []

        for i in range(self.num_classes):
            if self.total_seen[i] == 0:
                ious.append(1)
            else:
                cur_iou = self.total_correct[i] / (self.total_seen[i]
                                                   + self.total_positive[i]
                                                   - self.total_correct[i])
                ious.append(cur_iou)

        self.miou = np.mean(ious)
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'summary'):
            self.trainer.summary.add_scalar(self.name, self.miou * 100)
        # else:
        #     # print(ious)
        #     print(self.miou)

    # def get_step_miou(self):
    #     print('this is miou module', str(self.step_miou))
    #     return self.step_miou
