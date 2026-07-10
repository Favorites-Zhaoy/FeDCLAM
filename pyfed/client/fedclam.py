import torch
from pyfed.utils.clam_utils import calculate_of, calculate_vlr
from .base import BaseClient
from pyfed.manager.helper.build_loss import build_loss
from pyfed.loss.loss import *


class FedCLAMClient(BaseClient):
    def __init__(self, config, site, server_model, partition=None):
        super(FedCLAMClient, self).__init__(config, site, server_model, partition)
        self.init_val_loss = None
        self.init_val_acc = None
        self.trained_val_loss = None
        self.trained_val_acc = None
        self.train_loss = None
        self.val_loss_ratio = None
        self.of_penalty = None
        self.alpha = config.ALPHA
        self.beta = config.BETA
        self.fim_warmup = config.FIM_WARMUP
        self.fim_rampup = config.FIM_RAMPUP

    def train(self, server_model=None):

        # FIM Loss warm-up
        if self.fim_rampup:
            curr_l_fim = self.round / self.config.TRAIN_ROUNDS * 0.1
            self.config.L_FIM = curr_l_fim
            self.loss_fn = build_loss(self.config)
        else:
            if self.round < self.fim_warmup:
                self.loss_fn = JointLoss() if "fundus" not in self.config.TRAIN_LOSS else DiceLossFundus()
            else:
                self.loss_fn = build_loss(self.config)

        
        self.init_val_loss, self.init_val_acc = self.val()
        self.model.to(self.device)
        self.model.train()
        loss_all = 0

        metric_sum = 0.0
        sample_count = 0

        for step, (image, label) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            image, label = image.to(self.device), label.to(self.device)
            output = self.model(image)
            loss = (
                self.loss_fn(output, label)
                if not 'FIM' in self.loss_fn._get_name()
                else 
                self.loss_fn(output, label, image)
            )
            loss_all += loss.item()
            metric_sum, sample_count = self._metric_update(output, label, metric_sum, sample_count)
            loss.backward()
            self.optimizer.step()
            self.curr_iter += 1
        
        loss = loss_all / len(self.train_loader)
        self.train_loss = loss
        self.trained_val_loss, self.trained_val_acc = self.val()
        self.val_loss_ratio = calculate_vlr(self.init_val_loss, self.trained_val_loss, self.beta)
        self.of_penalty = calculate_of(self.train_loss, self.trained_val_loss, self.alpha)

        acc = metric_sum / max(sample_count, 1)
        self.round += 1

        self.model.to('cpu')
        return loss, acc
    

    def client_to_server(self):
        return {'model': self.model,
                'optimizer': self.optimizer,
                'data_len': len(self.train_loader),
                'init_val_loss': self.init_val_loss,
                'init_val_acc': self.init_val_acc,
                'trained_val_loss': self.trained_val_loss,
                'trained_val_acc': self.trained_val_acc,
                'train_loss': self.train_loss,
                'val_loss_ratio': self.val_loss_ratio,
                'of_penalty': self.of_penalty,
                }
