from monai.losses.dice import DiceFocalLoss
import torch.nn as nn
from pyfed.utils.metric import Metric
from pyfed.loss.loss import *


def _deep_supervision_enabled(config):
    if getattr(config, 'DEEP_SUPERVISION', False):
        return True
    network_params = getattr(config, 'NETWORK_PARAMS', {})
    return bool(network_params.get('deep_supervision', False))


def _attach_loss_metadata(loss_fn, uses_image=False):
    loss_fn.uses_image = uses_image
    return loss_fn


def _maybe_wrap_deep_supervision(config, loss_fn, uses_image=False):
    if not _deep_supervision_enabled(config):
        return _attach_loss_metadata(loss_fn, uses_image=uses_image)
    weights = getattr(config, 'DEEP_SUPERVISION_WEIGHTS', [1.0, 0.5, 0.25])
    return DeepSupervisionLoss(loss_fn, weights=weights, uses_image=uses_image)


def build_loss(config):
    uses_image = False
    if config.TRAIN_LOSS == 'diceloss':
        loss_fn = DiceLoss()
    elif config.TRAIN_LOSS == 'dicece' or config.TRAIN_LOSS == 'joint':
        loss_fn = JointLoss()
    elif config.TRAIN_LOSS == 'tversky':
        loss_fn = TverskyLoss()
    elif config.TRAIN_LOSS == 'tverskyce':
        loss_fn = TverskyCELoss()
    elif config.TRAIN_LOSS == 'FIM':
        loss_fn = FIMLoss()
        uses_image = True
    elif config.TRAIN_LOSS == 'diceFIM':
        loss_fn = DiceFIMLoss(lambda_fim=config.L_FIM)
        uses_image = True
    elif config.TRAIN_LOSS == 'tverskyFIM':
        loss_fn = TverskyFIMLoss()
        uses_image = True
    elif config.TRAIN_LOSS == 'diceFIMce':
        if config.FIM_METHOD is not None:
            loss_fn = DiceFIMCELoss(lambda_fim=config.L_FIM, method=config.FIM_METHOD)
        else:
            loss_fn = DiceFIMCELoss(lambda_fim=config.L_FIM)
        uses_image = True
    elif config.TRAIN_LOSS == 'tverskyFIMce':
        loss_fn = TverskyFIMCELoss()
        uses_image = True
    elif config.TRAIN_LOSS == 'Hausdorff':
        loss_fn = HausdorffLoss()
    elif config.TRAIN_LOSS == 'ce':
        loss_fn = nn.CrossEntropyLoss()
    elif config.TRAIN_LOSS == 'dicefocal':
        loss_fn = DiceFocalLoss(include_background=True, to_onehot_y=False, softmax=True)
    elif config.TRAIN_LOSS == 'edl_dice':
        loss_fn = EDL_Dice_Loss()
    elif config.TRAIN_LOSS == 'edl_dicefundus':
        loss_fn = EDL_Dice_LossFundus()
    elif config.TRAIN_LOSS == 'dicefundus':
        loss_fn = DiceLossFundus()
    elif config.TRAIN_LOSS == 'diceFIMfundus':
        if config.FIM_METHOD is not None:
            loss_fn = DiceFIMLossFundus(lambda_fim=config.L_FIM, fim_method=config.FIM_METHOD)
        else:
            loss_fn = DiceFIMLossFundus(lambda_fim=config.L_FIM)
        uses_image = True
    else:
        raise ValueError('Unknown loss function: {}'.format(config.TRAIN_LOSS))

    return _maybe_wrap_deep_supervision(config, loss_fn, uses_image=uses_image)


def build_warmup_loss(config):
    if 'fundus' in config.TRAIN_LOSS:
        loss_fn = DiceLossFundus()
    else:
        loss_fn = JointLoss()
    return _maybe_wrap_deep_supervision(config, loss_fn, uses_image=False)


def build_metric(config):
    return Metric(config.METRIC)