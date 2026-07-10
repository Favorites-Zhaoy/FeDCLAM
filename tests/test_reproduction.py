import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from pyfed.dataset.dataset import Fundus, Nuclei
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

    def test_fundus_valid_split_does_not_fall_back_to_test(self):
        with TemporaryDirectory() as tmp:
            site_dir = Path(tmp) / "siteA"
            self._write_image_mask_split(site_dir, "train", count=10, fundus=True)
            self._write_image_mask_split(site_dir, "test", count=3, fundus=True)

            train_set = Fundus("siteA", tmp, train_ratio=0.6, split="train")
            valid_set = Fundus("siteA", tmp, train_ratio=0.6, split="valid")
            test_set = Fundus("siteA", tmp, train_ratio=0.6, split="test")

            self.assertTrue(set(train_set.sample_paths).isdisjoint(valid_set.sample_paths))
            self.assertTrue(set(valid_set.sample_paths).isdisjoint(test_set.sample_paths))
            self.assertEqual(len(train_set), 6)
            self.assertEqual(len(valid_set), 4)

    def test_nuclei_valid_split_does_not_fall_back_to_test(self):
        with TemporaryDirectory() as tmp:
            site_dir = Path(tmp) / "siteB"
            self._write_image_mask_split(site_dir, "train", count=10, fundus=False)
            self._write_image_mask_split(site_dir, "test", count=3, fundus=False)

            train_set = Nuclei("siteB", tmp, train_ratio=0.7, split="train")
            valid_set = Nuclei("siteB", tmp, train_ratio=0.7, split="valid")
            test_set = Nuclei("siteB", tmp, train_ratio=0.7, split="test")

            self.assertTrue(set(train_set.sample_paths).isdisjoint(valid_set.sample_paths))
            self.assertTrue(set(valid_set.sample_paths).isdisjoint(test_set.sample_paths))
            self.assertEqual(len(train_set), 7)
            self.assertEqual(len(valid_set), 3)

    @staticmethod
    def _write_image_mask_split(site_dir, split, count, fundus):
        image_dir = site_dir / split / "image"
        mask_dir = site_dir / split / "mask"
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(count):
            image = np.full((8, 8, 3), idx, dtype=np.uint8)
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[2:5, 2:5] = 2 if fundus else 1
            np.save(image_dir / f"{idx:03d}.npy", image)
            np.save(mask_dir / f"{idx:03d}.npy", mask)


if __name__ == "__main__":
    unittest.main()
