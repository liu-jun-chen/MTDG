import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import copy
import numpy as np
from collections import OrderedDict
try:
    from backpack import backpack, extend
    from backpack.extensions import BatchGrad
except:
    backpack = None

from core import builder

__all__ = ['HGP']

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, model, criterion):
        super(Algorithm, self).__init__()
        self.model = model
        self.criterion = criterion

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

class HGP(Algorithm):
    "Domain Generanization through Hessian Gradient Alignment-HGP"

    def __init__(self, model, criterion):
        super(HGP, self).__init__(model, criterion)
        self.model = model
        self.criterion = criterion

        # self.register_buffer("update_count", torch.tensor([0]))
        # self.bce_extended = nn.CrossEntropyLoss()
        self.penalty_alpha = 0.5
        self.penalty_beta = 0.5
    #     self._init_optimizer()

    # def _init_optimizer(self):
    #     self.optimizer = torch.optim.Adam(
    #         self.network.parameters(),
    #         lr=self.hparams["lr"],
    #         weight_decay=self.hparams["weight_decay"],
    #     )

    def update(self, inputs_lidar, targets):
        
        envs = []
        dataset_names = ['kitti', 'synlidar', 'poss']
        for name in dataset_names:
            # features = self.featurizer(x)
            # logits = self.classifier(features)
            output, feat = self.model(inputs_lidar[name])
            env = {}
            env['nll'] = self.criterion(output, targets[name])
            env['sadg'], env['grad'] = self.compute_sadg_penalty(output, targets[name])
            envs.append(env)
        
        train_nll = torch.stack([env['nll'] for env in envs]).mean()

        mean_grad = autograd.grad(train_nll, self.model.classifier.parameters(),create_graph=True, retain_graph=True)
        flatten_mean_grad = self._flatten_grad(mean_grad)
        norm_of_mean_grad=flatten_mean_grad.pow(2).sum().sqrt()
        norm_of_mean_grad = norm_of_mean_grad+ 1e-16
        grad_of_norm_of_mean_grad = autograd.grad(norm_of_mean_grad, self.classifier.parameters(), create_graph=True,retain_graph=True)
        flatten_grad_of_norm_of_mean_grad = self._flatten_grad(grad_of_norm_of_mean_grad)
        mean_hessian_grad= torch.mul(norm_of_mean_grad,flatten_grad_of_norm_of_mean_grad) 

        loss = train_nll.clone()
        
       
        sadg_penalty_list = []
        all_flatten_grads = [self._flatten_grad(env['grad']) for env in envs]

        
        grads_of_norm_of_grad = [autograd.grad(env['sadg'], self.model.classifier.parameters(), create_graph=True,retain_graph=True) for env in envs]
        all_flatten_grads_of_norm_of_grad = [self._flatten_grad(grad_of_norm_of_grad) for grad_of_norm_of_grad in grads_of_norm_of_grad]

        hessian_grad = [torch.mul(envs[k]['sadg'],f_grad) for k, f_grad in enumerate(all_flatten_grads_of_norm_of_grad)]
        
        if len(envs) > 0:
            for i in range(len(all_flatten_grads)):
                sadg_penalty_list.append(self.penalty_alpha *(hessian_grad[i] - mean_hessian_grad.detach()).pow(2).sum() + self.penalty_beta * (all_flatten_grads[i] - flatten_mean_grad.detach()).pow(2).sum() )

            N = len(sadg_penalty_list)
            sadg_penalty = torch.stack(sadg_penalty_list).sum()/len(envs)
        else:
             sadg_penalty = torch.stack([self.penalty_alpha * torch.flatten(hessian_grad[0]).pow(2).sum(),self.penalty_beta* envs[0]['sadg']]).sum()


        loss += sadg_penalty

        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        # self.update_count += 1
        # return {'loss': loss.item(), 'nll': train_nll.item(), 'penalty': sadg_penalty.item()}
        return loss

    def compute_sadg_penalty(self, logits, y):
        gradient_norm=[]
        numels=[]
        loss = self.criterion(logits, y)
        grads = autograd.grad(loss, self.model.classifier.parameters(), create_graph=True,retain_graph=True)
        for grad in grads:
            grad = grad + 1e-16
            gradient_norm.append(torch.norm(grad, p=2))
            numels.append(torch.numel(grad))
        gradient_loss = torch.norm(torch.stack(gradient_norm), p=2)
        return  gradient_loss, grads
        
    def _flatten_grad(self, grads):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad