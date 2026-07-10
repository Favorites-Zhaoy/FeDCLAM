import argparse
import importlib
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from pyfed.manager.manager import Manager
from pathlib import Path


def cli_main(config):
    manager = Manager(config)

    print('mode: ', config.TRAIN_MODE)

    if config.TRAIN_MODE == 'federated':
        print('train')
        manager.train()
        manager.finish()
    elif config.TRAIN_MODE == 'individual':
        manager.train_individual()
    elif config.TRAIN_MODE == 'centralized':
        manager.train_centralized()
    elif config.TRAIN_MODE == 'innerouter':
        manager.train_inner_outer()
        manager.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default='fedavg')
    parser.add_argument('--inner_sites', nargs='+', default=[])
    parser.add_argument('--outer_sites', nargs='+', default=[])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--trial', default=0)
    parser.add_argument('--train_ratio', default=None)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Root directory of the preprocessed dataset')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Root directory used for experiment outputs')
    parser.add_argument('--protocol', choices=['official', 'strict'], default='official',
                        help='Evaluation protocol and data split to use')
    parser.add_argument('--train_mode',
                        choices=['federated', 'individual', 'centralized', 'innerouter'],
                        default=None,
                        help='Override config.TRAIN_MODE')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from the latest complete checkpoint')
    parser.add_argument("--fl_rounds", type=int, default=None, help="Number of FL rounds")
    parser.add_argument("--local_epochs", type=int, default=None, help="Number of local epochs")
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay')
    parser.add_argument('--alpha', type=float, default=None, help='FedCLAM dampening exponent')
    parser.add_argument('--beta', type=float, default=None, help='FedCLAM sigmoid steepness')
    parser.add_argument('--fim_method', type=str, choices=['mse', 'w2'], default=None, help='FIM loss calculation method')
    parser.add_argument('--fim_warmup', type=int, default=10, help='Rounds before FIM is enabled')
    parser.add_argument('--fim_rampup', type=str, choices=["Yes", "No"], default="No", help='Alternative, ramp-up FIM weight over time')
    parser.add_argument('--zero_init', type=str, choices=["Yes", "No"], default=False, help='Zero initialization of speed vectors')
    parser.add_argument("--norm", type=str, choices=["bn", "in", None], default="in", help="Normalization type")
    parser.add_argument("--loss", type=str, default=None, help="loss function")
    parser.add_argument("--l_fim", type=float, default=None, help="lambda FIM weight")
    parser.add_argument('--run_name', help='A short display name for this run, which is how you will '
                                                          'identify this run in the UI.')
    parser.add_argument('--run_notes', required=True, help='A longer description of the run, like a -m commit message '
                                                           'in git.')

    args = parser.parse_args()

    config_cls = getattr(importlib.import_module(args.config), 'Config')
    # e.g. config.prostate_mri.fedavg
    config = config_cls(exp_name=f'{args.exp_name}_{args.run_notes}')
    config.METHOD = args.config.rsplit('.', 1)[-1]
    # e.g. fedavg_trial0, this is the name of the experiment

    if args.dataset is not None:
        config.DATASET = args.dataset
    if args.seed is not None:
        config.SEED = args.seed
    if args.fl_rounds is not None:
        config.TRAIN_ROUNDS = args.fl_rounds
    if args.local_epochs is not None:
        config.TRAIN_EPOCH_PER_ROUND = args.local_epochs
    if args.norm is not None:
        config.NETWORK_PARAMS['norm'] = args.norm
    if args.lr is not None:
        config.TRAIN_LR = args.lr
    if args.batch_size is not None:
        config.TRAIN_BATCHSIZE = args.batch_size
    if args.weight_decay is not None:
        config.TRAIN_WEIGHT_DECAY = args.weight_decay
    if args.alpha is not None:
        config.ALPHA = args.alpha
    if args.beta is not None:
        config.BETA = args.beta
    if args.zero_init is not None:
        if args.zero_init == "Yes":
            config.ZERO_INIT = True
        else:
            config.ZERO_INIT = False
    if args.fim_warmup is not None:
        config.FIM_WARMUP = args.fim_warmup
    if args.fim_rampup is not None:
        config.FIM_RAMPUP = args.fim_rampup
        if args.fim_rampup == "Yes":
            config.FIM_RAMPUP = True
        else:
            config.FIM_RAMPUP = False
    if args.fim_method is not None:
        config.FIM_METHOD = args.fim_method
    if args.loss is not None:
        config.TRAIN_LOSS = args.loss
    if args.train_mode is not None:
        config.TRAIN_MODE = args.train_mode
    if args.l_fim is not None:
        config.L_FIM = args.l_fim
    if args.run_name is not None:
        config.RUN_NAME = args.run_name

    config.PROTOCOL = args.protocol
    if args.data_dir is not None:
        config.DIR_DATA = str(Path(args.data_dir).resolve())
    if args.output_dir is not None:
        config.DIR_SAVE = str((Path(args.output_dir).resolve() /
                               config.DATASET / args.protocol /
                               config.EXP_NAME))
        config.DIR_CKPT = str(Path(config.DIR_SAVE) / 'ckpt')
        config.DIR_LOG = str(Path(config.DIR_SAVE) / 'log')
    Path(config.DIR_CKPT).mkdir(parents=True, exist_ok=True)
    Path(config.DIR_LOG).mkdir(parents=True, exist_ok=True)
    config.TRAIN_RESUME = args.resume
    

    print('seed:', config.SEED)
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True

    print('NETWORK_PARAMS :', config.NETWORK_PARAMS)

    if len(args.inner_sites) > 0:
        config.INNER_SITES = args.inner_sites
    if len(args.outer_sites) > 0:
        config.OUTER_SITES = args.outer_sites
    if args.train_ratio is not None:
        config.TRAIN_RATIO = float(args.train_ratio)

    print(args)
    cli_main(config)
