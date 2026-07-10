import unittest

import numpy as np
import torch

from pyfed.loss.loss import DiceFIMCELoss, DiceFIMLossFundus
from pyfed.utils.metric import dice_func, dice_func_fundus
from tools.prepare_datasets import normalize_fundus_mask, split_indices


class ReproductionTests(unittest.TestCase):
    def test_fundus_mask_normalization(self):
        source = np.array([[255, 128, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(normalize_fundus_mask(source), [[0, 1, 2]])

    def test_split_is_deterministic_and_disjoint(self):
        first = split_indices(159, 0)
        second = split_indices(159, 0)
        self.assertEqual(first, second)
        sets = [set(first[name]) for name in ("train", "valid", "test")]
        self.assertFalse(sets[0] & sets[1])
        self.assertFalse(sets[0] & sets[2])
        self.assertFalse(sets[1] & sets[2])
        self.assertEqual(len(set.union(*sets)), 159)

    def test_dice_metrics_are_one_for_perfect_predictions(self):
        label = torch.tensor([[[[0, 1], [1, 0]]]], dtype=torch.float32)
        logits = torch.cat([(1 - label) * 20, label * 20], dim=1)
        self.assertAlmostEqual(dice_func(logits, label), 1.0, places=5)
        fundus_label = torch.cat([label, label], dim=1)
        fundus_logits = torch.where(fundus_label > 0, 20.0, -20.0)
        self.assertAlmostEqual(dice_func_fundus(fundus_logits, fundus_label), 1.0, places=5)

    def test_fim_losses_are_finite(self):
        image = torch.rand(2, 3, 32, 32)
        prostate_target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        fundus_target = torch.randint(0, 2, (2, 2, 32, 32)).float()
        prediction = torch.randn(2, 2, 32, 32)
        self.assertTrue(torch.isfinite(DiceFIMCELoss(lambda_fim=0.01)(prediction, prostate_target, image)))
        self.assertTrue(torch.isfinite(DiceFIMLossFundus(lambda_fim=0.01)(prediction, fundus_target, image)))


if __name__ == "__main__":
    unittest.main()
