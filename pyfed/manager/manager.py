import os
import shutil
import csv
import json
import random

import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter

from pyfed.manager.comm import Comm
from pyfed.manager.helper.build_model import (
    build_model,
    build_client
)
from pyfed.utils.log import print_log


class Manager(object):
    def __init__(self, config):
        self.config = config
        self.best_acc = 0
        self.best_epoch = 0
        self.start_round = 0

        self._setup()

    def _setup(self):
        self.server_model = build_model(self.config)

        pytorch_total_params = sum(p.numel() for p in self.server_model.parameters())
        total_params_size = abs(pytorch_total_params * 4. / (1024 ** 2.))
        print(f'Network: {self.config.NETWORK}, Params size (MB): {total_params_size}')

        self._build_clients()
        self.comm = Comm(self.server_model, self.config)
        self.writer = SummaryWriter(log_dir=self.config.DIR_LOG)
        if getattr(self.config, 'TRAIN_RESUME', False):
            self._resume()

    def _resume(self):
        path = os.path.join(self.config.DIR_CKPT, 'model_latest.pth')
        if not os.path.isfile(path):
            print_log(f'Resume requested but no checkpoint exists at {path}; starting fresh')
            return
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        self.server_model.load_state_dict(checkpoint['server'])
        self.best_epoch = checkpoint.get('best_epoch', 0)
        self.best_acc = checkpoint.get('best_acc', 0)
        self.start_round = checkpoint.get('round', -1) + 1
        for idx, client in enumerate(self.inner_clients):
            key = f'optim_{idx}'
            if key in checkpoint:
                client.optimizer.load_state_dict(checkpoint[key])
                for state in client.optimizer.state.values():
                    for state_key, state_value in state.items():
                        if torch.is_tensor(state_value):
                            state[state_key] = state_value.to(client.device)
            client.server_to_client(self.server_model)
            client.round = self.start_round
        self._load_comm_state(checkpoint.get('comm_state', {}))
        rng = checkpoint.get('rng_state', {})
        if 'python' in rng: random.setstate(rng['python'])
        if 'numpy' in rng: np.random.set_state(rng['numpy'])
        if 'torch' in rng: torch.set_rng_state(rng['torch'])
        if 'cuda' in rng and torch.cuda.is_available(): torch.cuda.set_rng_state_all(rng['cuda'])
        print_log(f'Resumed from round {self.start_round}')

    def _comm_state(self):
        state = {}
        comm = self.comm.comm_fn
        for name in ('of_penalties', 'val_loss_ratios', 'local_train_losses',
                     'local_init_val_losses', 'local_trained_val_losses',
                     'client_weight_speeds', 'u_dis', 'u_data'):
            if hasattr(comm, name): state[name] = getattr(comm, name)
        if hasattr(comm, 'v'): state['v'] = comm.v.state_dict()
        return state

    def _load_comm_state(self, state):
        comm = self.comm.comm_fn
        for name, value in state.items():
            if name == 'v' and hasattr(comm, 'v'):
                comm.v.load_state_dict(value)
            elif hasattr(comm, name):
                setattr(comm, name, value)

    def _append_long_metrics(self, round_idx, metrics):
        fields = ['dataset', 'method', 'protocol', 'round', 'site', 'split',
                  'loss', 'dice', 'avg_dice', 'site_std', 'metric', 'value']
        rows = []
        mappings = [
            ('train', 'train_loss_', 'train_acc_'),
            ('server_val', 'server_val_loss_on_', 'server_val_acc_on_'),
            ('server_test', 'server_test_loss_on_', 'server_test_acc_on_'),
            ('local_val', 'l_val_loss_', 'l_val_acc_'),
            ('local_test', 'l_test_loss_', 'l_test_acc_'),
        ]
        for split, loss_prefix, dice_prefix in mappings:
            dice_values = [metrics.get(dice_prefix + site, np.nan) for site in self.config.INNER_SITES]
            finite = np.asarray([x for x in dice_values if np.isfinite(x)], dtype=float)
            avg = float(np.mean(finite)) if finite.size else np.nan
            std = float(np.std(finite)) if finite.size else np.nan
            for site, dice in zip(self.config.INNER_SITES, dice_values):
                loss = metrics.get(loss_prefix + site, np.nan)
                if np.isfinite(loss) or np.isfinite(dice):
                    rows.append(dict(dataset=self.config.DATASET,
                                     method=getattr(self.config, 'METHOD', self.config.COMM_TYPE),
                                     protocol=getattr(self.config, 'PROTOCOL', 'official'),
                                     round=round_idx, site=site, split=split,
                                     loss=loss, dice=dice, avg_dice=avg, site_std=std,
                                     metric='', value=''))
        for prefix in ('init_val_loss_', 'trained_val_loss_', 'init_val_acc_',
                       'trained_val_acc_', 'val_loss_ratio_', 'of_penalty_'):
            for site in self.config.INNER_SITES:
                key = prefix + site
                if key in metrics:
                    rows.append(dict(dataset=self.config.DATASET,
                                     method=getattr(self.config, 'METHOD', self.config.COMM_TYPE),
                                     protocol=getattr(self.config, 'PROTOCOL', 'official'),
                                     round=round_idx, site=site, split='fedclam',
                                     loss='', dice='', avg_dice='', site_std='',
                                     metric=prefix.rstrip('_'), value=metrics[key]))
        for split, loss_prefix, dice_prefix in [
            ('outer_val', 'g_outer_val_loss_', 'g_outer_val_acc_'),
            ('outer_test', 'g_outer_test_loss_', 'g_outer_test_acc_'),
            ('inner_val', 'g_inner_val_loss_', 'g_inner_val_acc_'),
            ('inner_test', 'g_inner_test_loss_', 'g_inner_test_acc_'),
        ]:
            sites = self.config.OUTER_SITES if split.startswith('outer') else self.config.INNER_SITES
            dice_values = [metrics.get(dice_prefix + site, np.nan) for site in sites]
            finite = np.asarray([x for x in dice_values if np.isfinite(x)], dtype=float)
            avg = float(np.mean(finite)) if finite.size else np.nan
            std = float(np.std(finite)) if finite.size else np.nan
            for site, dice in zip(sites, dice_values):
                loss = metrics.get(loss_prefix + site, np.nan)
                if np.isfinite(loss) or np.isfinite(dice):
                    rows.append(dict(dataset=self.config.DATASET,
                                     method=getattr(self.config, 'METHOD', self.config.COMM_TYPE),
                                     protocol=getattr(self.config, 'PROTOCOL', 'official'),
                                     round=round_idx, site=site, split=split,
                                     loss=loss, dice=dice, avg_dice=avg, site_std=std,
                                     metric='', value=''))
        path = os.path.join(self.config.DIR_LOG, 'metrics_long.csv')
        write_header = not os.path.isfile(path) or os.path.getsize(path) == 0
        with open(path, 'a', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if write_header: writer.writeheader()
            writer.writerows(rows)

    def _aggregate_clients(self, iter_round, metrics):
        if self.config.COMM_TYPE == 'FedCLAM':
            clients = []
            for client in self.inner_clients:
                client_info = client.client_to_server()
                clients.append(client_info)
                metrics['init_val_loss_' + client.name] = client_info['init_val_loss']
                metrics['init_val_acc_' + client.name] = client_info['init_val_acc']
                metrics['trained_val_loss_' + client.name] = client_info['trained_val_loss']
                metrics['trained_val_acc_' + client.name] = client_info['trained_val_acc']
                metrics['val_loss_ratio_' + client.name] = client_info['val_loss_ratio']
                metrics['of_penalty_' + client.name] = client_info['of_penalty']
            return self.comm(clients, self.client_weights, self.server_model, iter_round)
        if self.config.COMM_TYPE == 'FedEvi':
            clients = [client.client_to_server() for client in self.inner_clients]
            return self.comm(clients, self.client_weights, self.server_model, iter_round)
        client_models = [client.client_to_server()['model'] for client in self.inner_clients]
        return self.comm(client_models, self.client_weights, self.server_model, iter_round)


    def _build_clients(self):
        client_class = build_client(self.config)
        print_log(f'Client type: {self.config.CLIENT}')

        if self.config.TRAIN_MODE == 'centralized':
            self.central = client_class(self.config, self.config.INNER_SITES, self.server_model)
        else:
            self.inner_clients = [client_class(self.config, site, self.server_model)
                                  for site in self.config.INNER_SITES]

            train_nums = [len(client.train_loader.dataset)  for client in self.inner_clients]
            self.client_weights = [num * 1.0 / sum(train_nums) for num in train_nums]
            print(self.client_weights)
            if len(self.config.OUTER_SITES) > 0:
                self.outer_clients = [client_class(self.config, site, self.server_model)
                                      for site in self.config.OUTER_SITES]

    def train(self):
        metrics = {}
        best_server_val_acc, best_server_val_round = 0, 0
        best_person_val_acc, best_person_val_round = 0, 0
        best_server_test_acc, best_server_test_round = 0, 0

        for iter_round in range(self.start_round, self.config.TRAIN_ROUNDS):
            for iter_local in range(self.config.TRAIN_EPOCH_PER_ROUND):
                print_log(f"============ Train epoch: {iter_local + iter_round * self.config.TRAIN_EPOCH_PER_ROUND}/{self.config.TRAIN_ROUNDS} ============")

                for ci, client in enumerate(self.inner_clients):
                    train_loss, train_acc = client.train(server_model=self.server_model)

                    print_log(f'site-{client.name:<10s}| train loss: {train_loss:.4f} | train acc: {train_acc:.4f}')
                    metrics['train_loss_' + client.name] = train_loss
                    self.writer.add_scalar(f'Train_Loss/{client.name}', train_loss, iter_round)
                    metrics['train_acc_' + client.name] = train_acc
                    self.writer.add_scalar(f'Train_Acc/{client.name}', train_acc, iter_round)

            # client to server
            if self.config.COMM_TYPE == 'FedCLAM':
                # clients = [client.client_to_server() for client in self.inner_clients]
                clients = []
                for client in self.inner_clients:
                    client_info = client.client_to_server()
                    clients.append(client_info)
                    self.writer.add_scalar(f'Init_Val_Loss/{client.name}', client_info['init_val_loss'], iter_round)
                    metrics['init_val_loss_' + client.name] = client_info['init_val_loss']
                    metrics['init_val_acc_' + client.name] = client_info['init_val_acc']
                    self.writer.add_scalar(f'Trained_Val_Loss/{client.name}', client_info['trained_val_loss'], iter_round)
                    metrics['trained_val_loss_' + client.name] = client_info['trained_val_loss']
                    metrics['trained_val_acc_' + client.name] = client_info['trained_val_acc']
                    self.writer.add_scalar(f'Val_Loss_Ratio/{client.name}', client_info['val_loss_ratio'], iter_round)
                    metrics['val_loss_ratio_' + client.name] = client_info['val_loss_ratio']
                    self.writer.add_scalar(f'OF_Penalty/{client.name}', client_info['of_penalty'], iter_round)
                    metrics['of_penalty_' + client.name] = client_info['of_penalty']
                client_weights = self.client_weights
                self.server_model = self.comm(clients, client_weights, self.server_model, iter_round)
            elif self.config.COMM_TYPE == 'FedEvi':
                clients = [client.client_to_server() for client in self.inner_clients]
                client_weights = self.client_weights
                self.server_model = self.comm(clients, client_weights, self.server_model, iter_round)
            else:
                client_models = [client.client_to_server()['model'] for client in self.inner_clients]
                client_weights = self.client_weights
                self.server_model = self.comm(client_models, client_weights, self.server_model, iter_round)

            # run global validation and testing
            print_log('============== Server Model Validation ==============')
            val_accs, test_accs = [], []
            for ci, client in enumerate(self.inner_clients):
                val_loss, val_acc = client.val(self.server_model)
                if getattr(self.config, 'PROTOCOL', 'official') == 'strict':
                    test_loss, test_acc = np.nan, np.nan
                else:
                    test_loss, test_acc = client.test(self.server_model)
                val_accs.append(val_acc)
                test_accs.append(test_acc)

                metrics['server_val_loss_on_' + client.name] = val_loss
                metrics['server_val_acc_on_' + client.name] = val_acc
                metrics['server_test_loss_on_' + client.name] = test_loss
                metrics['server_test_acc_on_' + client.name] = test_acc

                print_log('site-{:<10s}| val loss: {:.4f} | val acc: {:.4f} | test acc: {:.4f}'.format(
                    client.name, val_loss, val_acc, test_acc))
                
                self.writer.add_scalar(f'Server_Val_Loss/{client.name}', val_loss, iter_round)
                self.writer.add_scalar(f'Server_Val_Acc/{client.name}', val_acc, iter_round)
                if np.isfinite(test_acc):
                    self.writer.add_scalar(f'Server_Test_Loss/{client.name}', test_loss, iter_round)
                    self.writer.add_scalar(f'Server_Test_Acc/{client.name}', test_acc, iter_round)

            metrics['server_val_acc_avg'] = np.mean(val_accs)
            metrics['server_test_acc_avg'] = np.nanmean(test_accs) if np.any(np.isfinite(test_accs)) else np.nan
            self.writer.add_scalar('Server_Val_Acc/Avg', np.mean(val_accs), iter_round)
            if np.isfinite(metrics['server_test_acc_avg']):
                self.writer.add_scalar('Server_Test_Acc/Avg', metrics['server_test_acc_avg'], iter_round)
            print_log('avg val acc: {:.4f} | avg test acc: {:.4f}'.format(np.mean(val_accs), np.mean(test_accs)))

            # for FedCLAM, we already do local validation in client.train
            if self.config.COMM_TYPE != 'FedCLAM':
                print_log('============== Local Validation ==============')
                val_accs, test_accs = [], []
                for ci, client in enumerate(self.inner_clients):
                    val_loss, val_acc = client.val()
                    if getattr(self.config, 'PROTOCOL', 'official') == 'strict':
                        test_loss, test_acc = np.nan, np.nan
                    else:
                        test_loss, test_acc = client.test()
                    val_accs.append(val_acc)
                    test_accs.append(test_acc)

                    metrics['l_val_loss_' + client.name] = val_loss
                    metrics['l_val_acc_' + client.name] = val_acc
                    metrics['l_test_loss_' + client.name] = test_loss
                    metrics['l_test_acc_' + client.name] = test_acc

                    print_log('site-{:<10s}| val loss: {:.4f} | val acc: {:.4f} | test acc: {:.4f}'.format(
                        client.name, val_loss, val_acc, test_acc))

                metrics['l_val_acc_avg'] = np.mean(val_accs)
                metrics['l_test_acc_avg'] = np.mean(test_accs)
                print_log('avg val acc: {:.4f} | avg test acc: {:.4f}'.format(np.mean(val_accs), np.mean(test_accs)))
                if metrics['l_val_acc_avg'] > best_person_val_acc:
                    best_person_val_acc = metrics['l_val_acc_avg']
                    best_person_val_round = iter_round
            if metrics['server_val_acc_avg'] > best_server_val_acc:
                best_server_val_acc = metrics['server_val_acc_avg']
                best_server_val_round = iter_round

            if np.isfinite(metrics['server_test_acc_avg']) and metrics['server_test_acc_avg'] > best_server_test_acc:
                best_server_test_acc = metrics['server_test_acc_avg']
                best_server_test_round = iter_round

            print_log('============== Summary ==============')
            print_log(f'best server val round: {best_server_val_round} | best server val acc: {best_server_val_acc:.4f}')
            print_log(f'best server test round: {best_server_test_round} | best server test acc: {best_server_test_acc:.4f}')
            # print_log('best person val round: {} | best person val acc: {:.4f}'.format(best_person_val_round, best_person_val_acc))

            # server to client
            for ci, client in enumerate(self.inner_clients):
                client.server_to_client(self.server_model)
            
            np.save(os.path.join(self.config.DIR_LOG, f'metrics_{iter_round}.npy'), metrics)
            self._append_long_metrics(iter_round, metrics)
            self.save(iter_round, np.mean(val_accs))

        if getattr(self.config, 'PROTOCOL', 'official') == 'strict':
            self._strict_final_test()

    def _strict_final_test(self):
        checkpoint = torch.load(os.path.join(self.config.DIR_CKPT, 'model_best.pth'),
                                map_location='cpu', weights_only=False)
        self.server_model.load_state_dict(checkpoint['server'])
        results = {}
        for client in self.inner_clients:
            loss, dice = client.test(self.server_model)
            results[client.name] = {'loss': loss, 'dice': dice}
        values = [item['dice'] for item in results.values()]
        results['summary'] = {'avg_dice': float(np.mean(values)),
                              'site_std': float(np.std(values)),
                              'selected_round': int(checkpoint['best_epoch'])}
        with open(os.path.join(self.config.DIR_LOG, 'strict_final_test.json'), 'w', encoding='utf-8') as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)

    def train_inner_outer(self):
        metrics = {}
        for iter_round in range(self.start_round, self.config.TRAIN_ROUNDS):
            # train each client for one round
            for iter_local in range(self.config.TRAIN_EPOCH_PER_ROUND):
                print_log("============ Train epoch: {}/{} ============".format(
                    iter_local + iter_round * self.config.TRAIN_EPOCH_PER_ROUND, self.config.TRAIN_ROUNDS))

                for ci, client in enumerate(self.inner_clients):
                    train_loss, train_acc = client.train(server_model=self.server_model)

                    print_log('site-{:<10s}| train loss: {:.4f} | train acc: {:.4f}'.format(
                        client.name, train_loss, train_acc))

                    metrics['train_loss_' + client.name] = train_loss
                    metrics['train_acc_' + client.name] = train_acc

            # client to server
            self.server_model = self._aggregate_clients(iter_round, metrics)

            # run global validation and testing
            print_log('============== {} =============='.format('Inner Testing'))
            val_accs, test_accs = [], []
            for ci, client in enumerate(self.inner_clients):
                val_loss, val_acc = client.val(self.server_model)
                test_loss, test_acc = client.test(self.server_model)
                val_accs.append(val_acc)
                test_accs.append(test_acc)

                metrics['g_inner_val_loss_' + client.name] = val_loss
                metrics['g_inner_val_acc_' + client.name] = val_acc
                metrics['g_inner_test_loss_' + client.name] = test_loss
                metrics['g_inner_test_acc_' + client.name] = test_acc

                print_log('site-{:<10s}| val loss: {:.4f} | val acc: {:.4f} | test acc: {:.4f}'.format(
                    client.name, val_loss, val_acc, test_acc))

            metrics['g_inner_val_acc_avg'] = np.mean(val_accs)
            metrics['g_inner_test_acc_avg'] = np.mean(test_accs)
            print_log(
                '[inner] avg val acc: {:.4f} | avg test acc: {:.4f}'.format(np.mean(val_accs), np.mean(test_accs)))

            print_log('============== {} =============='.format('Outer Testing'))
            val_accs, test_accs = [], []
            for ci, client in enumerate(self.outer_clients):
                val_loss, val_acc = client.val(self.server_model)
                test_loss, test_acc = client.test(self.server_model)
                val_accs.append(val_acc)
                test_accs.append(test_acc)

                metrics['g_outer_val_loss_' + client.name] = val_loss
                metrics['g_outer_val_acc_' + client.name] = val_acc
                metrics['g_outer_test_loss_' + client.name] = test_loss
                metrics['g_outer_test_acc_' + client.name] = test_acc

                print_log('site-{:<10s}| val loss: {:.4f} | val acc: {:.4f} | test acc: {:.4f}'.format(
                    client.name, val_loss, val_acc, test_acc))

            metrics['g_outer_val_acc_avg'] = np.mean(val_accs)
            metrics['g_outer_test_acc_avg'] = np.mean(test_accs)
            print_log(
                '[outer] avg val acc: {:.4f} | avg test acc: {:.4f}'.format(np.mean(val_accs), np.mean(test_accs)))

            # server to client
            for ci, client in enumerate(self.inner_clients):
                client.server_to_client(self.server_model)

            np.save(os.path.join(self.config.DIR_LOG, f'metrics_{iter_round}.npy'), metrics)
            self._append_long_metrics(iter_round, metrics)
            self.save(iter_round, np.mean(val_accs))

            # wandb.log(metrics)

    def train_centralized(self):
        metrics = {}
        for iter_round in range(0, self.config.TRAIN_ROUNDS):
            # train each client for one round
            for iter_local in range(self.config.TRAIN_EPOCH_PER_ROUND):
                print_log("============ Train epoch: {}/{} ============".format(
                    iter_local + iter_round * self.config.TRAIN_EPOCH_PER_ROUND, self.config.TRAIN_ROUNDS))

                train_loss, train_acc = self.central.train()

                print_log('central| train loss: {:.4f} | train acc: {:.4f}'.format(train_loss, train_acc))

                metrics['train_loss_central'] = train_loss
                metrics['train_acc_central'] = train_acc

            # run global validation and testing
            print_log('============== {} =============='.format('Global Validation'))
            val_loss_list, val_acc_list = self.central.val()
            test_loss_list, test_acc_list = self.central.test()

            for site_idx, site in enumerate(self.config.INNER_SITES):
                metrics['g_val_loss_' + site] = val_loss_list[site_idx]
                metrics['g_val_acc_' + site] = val_acc_list[site_idx]
                metrics['g_test_acc_' + site] = test_acc_list[site_idx]

                print_log('site-{:<10s}| val loss: {:.4f} | val acc: {:.4f} | test acc: {:.4f}'.format(
                    site, val_loss_list[site_idx], val_acc_list[site_idx], test_acc_list[site_idx]))

            metrics['l_val_acc_avg'] = np.mean(val_acc_list)
            metrics['l_test_acc_avg'] = np.mean(test_acc_list)
            print_log(
                'avg val acc: {:.4f} | avg test acc: {:.4f}'.format(np.mean(val_acc_list), np.mean(test_acc_list)))

            self.save(iter_round, np.mean(val_acc_list))

            # wandb.log(metrics)

    def train_individual(self):
        metrics = {}
        for iter_round in range(0, self.config.TRAIN_ROUNDS):
            # train each client for one round
            for iter_local in range(self.config.TRAIN_EPOCH_PER_ROUND):
                print_log("============ Train epoch: {}/{} ============".format(
                    iter_local + iter_round * self.config.TRAIN_EPOCH_PER_ROUND, self.config.TRAIN_ROUNDS))

                for ci, client in enumerate(self.inner_clients):
                    train_loss, train_acc = client.train()

                    print_log('site-{:<10s}| train loss: {:.4f} | train acc: {:.4f}'.format(
                        client.name, train_loss, train_acc))

                    metrics['train_loss_' + client.name] = train_loss
                    metrics['train_acc_' + client.name] = train_acc

            # run global validation and testing
            print_log('============== {} =============='.format('Global Validation'))
            val_accs, test_accs = [], []
            for ci, client in enumerate(self.inner_clients):
                val_loss, val_acc = client.val()
                test_loss, test_acc = client.test()
                val_accs.append(val_acc)
                test_accs.append(test_acc)

                metrics['l_val_loss_' + client.name] = val_loss
                metrics['l_val_acc_' + client.name] = val_acc
                metrics['l_test_acc_' + client.name] = test_acc

                print_log('site-{:<10s}| val loss: {:.4f} | val acc: {:.4f} | test acc: {:.4f}'.format(
                    client.name, val_loss, val_acc, test_acc))

            metrics['l_val_acc_avg'] = np.mean(val_accs)
            metrics['l_test_acc_avg'] = np.mean(test_accs)
            print_log('avg val acc: {:.4f} | avg test acc: {:.4f}'.format(np.mean(val_accs), np.mean(test_accs)))

            self.save(iter_round, np.mean(val_accs))

            # wandb.log(metrics)

    def save(self, iter_round, val_acc):
        better = False
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.best_epoch = iter_round
            better = True

        # save server status
        save_dicts = {'server': self.server_model.state_dict(),
                      'best_epoch': self.best_epoch,
                      'best_acc': self.best_acc,
                      'round': iter_round}
        save_dicts['comm_state'] = self._comm_state()
        save_dicts['rng_state'] = {
            'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

        if self.config.TRAIN_MODE == 'centralized':
            save_dicts['optim'] = self.central.client_to_server()['optimizer'].state_dict()
        else:
            # save local status
            for ci, client in enumerate(self.inner_clients):
                save_dicts[f'optim_{ci}'] = client.client_to_server()['optimizer'].state_dict()

        latest_path = os.path.join(self.config.DIR_CKPT, 'model_latest.pth')
        torch.save(save_dicts, latest_path)
        if better:
            shutil.copyfile(latest_path, os.path.join(self.config.DIR_CKPT, 'model_best.pth'))


    def finish(self):
        self.writer.close()
        # in self.config.DIR_LOG we have one npy file for each round. We now merge them into one file and delete the others
        metrics = {}
        for iter_round in range(self.config.TRAIN_ROUNDS):
            round_metrics_path = os.path.join(self.config.DIR_LOG, f'metrics_{iter_round}.npy')
            with open(round_metrics_path, 'rb') as round_metrics_file:
                metrics[iter_round] = np.load(round_metrics_file, allow_pickle=True).item()
            if os.environ.get('FEDCLAM_CLEAN_ROUND_METRICS', '0') == '1':
                try:
                    os.remove(round_metrics_path)
                except PermissionError:
                    print_log(f'Warning: could not remove temporary metrics file: {round_metrics_path}')
        np.save(os.path.join(self.config.DIR_LOG, 'metrics.npy'), metrics)

        print_log('Finished training')
        if self.config.COMM_TYPE == 'FedCLAM':
            try:
                import matplotlib.pyplot as plt
            except ImportError as exc:
                print_log(f'Warning: skip FedCLAM diagnostic plots because matplotlib dependencies are unavailable: {exc}')
                return
            vlr = self.comm.comm_fn.val_loss_ratios.cpu().numpy().T
            of = self.comm.comm_fn.of_penalties.cpu().numpy().T
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            for client_idx in range(vlr.shape[1]):
                ax[0].plot(np.arange(len(vlr)), vlr[:, client_idx], label=f"Client {client_idx}")
                ax[1].plot(np.arange(len(of)), of[:, client_idx], label=f"Client {client_idx}")
            ax[0].set_title("VLR")
            ax[1].set_title("OF")
            plt.legend()
            plt.savefig(os.path.join(self.config.DIR_LOG, "vlr_of.png"))





