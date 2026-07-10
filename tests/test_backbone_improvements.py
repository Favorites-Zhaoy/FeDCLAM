import unittest

import torch

from pyfed.client.base import BaseClient
from pyfed.loss.loss import DeepSupervisionLoss, DiceFIMLoss, JointLoss
from pyfed.manager.helper.build_loss import build_loss, build_warmup_loss
from pyfed.manager.helper.build_model import build_model
from pyfed.network.resatt_unet import ResAttUNet


class BackboneImprovementTests(unittest.TestCase):
    def test_resattunet_deep_supervision_shapes(self):
        model = ResAttUNet(init_features=8, attention='gate', deep_supervision=True, norm='in')
        outputs = model(torch.randn(2, 3, 64, 64))
        self.assertEqual(len(outputs), 3)
        self.assertEqual(tuple(outputs[0].shape), (2, 2, 64, 64))
        self.assertEqual(tuple(outputs[1].shape[-2:]), (32, 32))
        self.assertEqual(tuple(outputs[2].shape[-2:]), (16, 16))

    def test_deep_supervision_loss_is_finite(self):
        model = ResAttUNet(init_features=8, attention='gate', deep_supervision=True, norm='in')
        image = torch.randn(2, 3, 64, 64)
        label = torch.randint(0, 2, (2, 1, 64, 64)).float()
        outputs = model(image)
        joint_loss = DeepSupervisionLoss(JointLoss())(outputs, label)
        fim_loss = DeepSupervisionLoss(DiceFIMLoss(lambda_fim=0.01), uses_image=True)(outputs, label, image)
        self.assertTrue(torch.isfinite(joint_loss))
        self.assertTrue(torch.isfinite(fim_loss))

    def test_builders_enable_final_model(self):
        class Config:
            NETWORK = 'resattunet'
            NETWORK_PARAMS = {
                'init_features': 8,
                'attention': 'gate',
                'deep_supervision': True,
                'norm': 'in',
            }
            DEEP_SUPERVISION = True
            DEEP_SUPERVISION_WEIGHTS = [1.0, 0.5, 0.25]
            TRAIN_LOSS = 'diceFIMce'
            L_FIM = 0.01
            FIM_METHOD = 'w2'

        model = build_model(Config)
        loss = build_loss(Config)
        warmup_loss = build_warmup_loss(Config)
        self.assertIsInstance(model, ResAttUNet)
        self.assertIsInstance(loss, DeepSupervisionLoss)
        self.assertTrue(loss.uses_image)
        self.assertIsInstance(warmup_loss, DeepSupervisionLoss)
        self.assertFalse(warmup_loss.uses_image)

    def test_fim_warmup_without_deep_supervision_does_not_pass_image_to_joint_loss(self):
        class Config:
            NETWORK_PARAMS = {'deep_supervision': False}
            DEEP_SUPERVISION = False
            TRAIN_LOSS = 'diceFIM'
            L_FIM = 0.01
            FIM_METHOD = 'w2'

        image = torch.randn(2, 3, 16, 16)
        label = torch.randint(0, 2, (2, 1, 16, 16)).float()
        output = torch.randn(2, 2, 16, 16)

        client = BaseClient.__new__(BaseClient)
        client.config = Config
        client.loss_fn = build_warmup_loss(Config)
        self.assertFalse(client._loss_uses_image())
        self.assertTrue(torch.isfinite(client._compute_loss(output, label, image)))

        client.loss_fn = build_loss(Config)
        self.assertTrue(client._loss_uses_image())
        self.assertTrue(torch.isfinite(client._compute_loss(output, label, image)))

if __name__ == '__main__':
    unittest.main()