"""Prepare the two datasets used by FedCLAM and emit auditable split manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk


FUNDUS_SITES = {
    "Drishti-GS": ("Drishti_GS", 101),
    "RIM-ONE": ("RIM-ONE-r3", 159),
    "REFUGE_t": ("REFUGE_Zeiss", 400),
    "REFUGE_v": ("REFUGE_Canon", 400),
}
FUNDUS_DOMAINS = {
    "Drishti-GS": "Domain1", "RIM-ONE": "Domain2",
    "REFUGE_t": "Domain3", "REFUGE_v": "Domain4",
}
PROSTATE_SITES = ["BIDMC", "HK", "I2CVB", "BMC", "RUNMC", "UCL"]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_files(path: Path) -> list[Path]:
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def split_indices(count: int, seed: int) -> dict[str, list[int]]:
    order = np.random.default_rng(seed).permutation(count).tolist()
    test_n = int(np.floor(0.2 * count))
    valid_n = int(np.floor(0.1 * count))
    return {
        "test": sorted(order[:test_n]),
        "valid": sorted(order[test_n:test_n + valid_n]),
        "train": sorted(order[test_n + valid_n:]),
    }


def select_masks(images: list[Path], masks: list[Path]) -> list[Path]:
    by_stem: dict[str, list[Path]] = defaultdict(list)
    for mask in masks:
        by_stem[mask.stem.lower()].append(mask)
    exact = [by_stem.get(image.stem.lower(), []) for image in images]
    if all(items for items in exact):
        return [items[0] for items in exact]
    if len(masks) == len(images) * 6:
        return masks[::6]
    if len(masks) == len(images):
        return masks
    raise ValueError(f"Cannot pair {len(images)} images with {len(masks)} masks")


def normalize_fundus_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[..., 0]
    values = np.unique(mask)
    if set(values.tolist()).issubset({0, 1, 2}):
        return mask.astype(np.uint8)
    result = np.zeros(mask.shape, dtype=np.uint8)
    result[mask < 64] = 2
    result[(mask >= 64) & (mask < 192)] = 1
    result[mask >= 192] = 0
    if not set(np.unique(result).tolist()).issubset({0, 1, 2}):
        raise ValueError("Fundus mask normalization produced invalid labels")
    return result


def prepare_fundus(source: Path, output: Path, seed: int) -> None:
    rows: list[dict[str, object]] = []
    for target_site, (source_site, expected) in FUNDUS_SITES.items():
        domain_root = source / FUNDUS_DOMAINS[target_site]
        released_layout = domain_root.is_dir()
        if released_layout:
            pairs = []
            for original_split in ("train", "test"):
                split_images = image_files(domain_root / original_split / "image")
                split_masks = image_files(domain_root / original_split / "mask")
                for image_path, mask_path in zip(split_images, select_masks(split_images, split_masks)):
                    pairs.append((original_split, image_path, mask_path))
            images = [item[1] for item in pairs]
            paired_masks = [item[2] for item in pairs]
        else:
            site_root = source / source_site
            images = image_files(site_root / "image")
            masks = image_files(site_root / "mask")
            paired_masks = select_masks(images, masks)
            pairs = [("unsplit", image, mask) for image, mask in zip(images, paired_masks)]
        if len(images) != expected:
            raise ValueError(f"{source_site}: expected {expected} images, found {len(images)}")
        if released_layout:
            train_indices = [idx for idx, item in enumerate(pairs) if item[0] == "train"]
            shuffled = np.random.default_rng(seed).permutation(train_indices).tolist()
            valid_n = min(max(1, int(np.floor(0.1 * expected))), max(1, len(train_indices) - 1))
            strict_valid = set(shuffled[:valid_n])
            official_membership = {idx: item[0] for idx, item in enumerate(pairs)}
            strict_membership = {idx: ("test" if item[0] == "test" else
                                       "valid" if idx in strict_valid else "train")
                                 for idx, item in enumerate(pairs)}
        else:
            splits = split_indices(len(images), seed)
            strict_membership = {idx: split for split, indices in splits.items() for idx in indices}
            official_membership = {idx: ("train" if split != "test" else "test")
                                   for idx, split in strict_membership.items()}
        for idx, (_, image_path, mask_path) in enumerate(pairs):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if image is None or mask is None:
                raise ValueError(f"Unreadable pair: {image_path}, {mask_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (384, 384), interpolation=cv2.INTER_CUBIC)
            mask = cv2.resize(normalize_fundus_mask(mask), (384, 384), interpolation=cv2.INTER_NEAREST)
            stem = f"{idx:04d}_{image_path.stem}"
            for protocol in ("official", "strict"):
                destination_split = (official_membership if protocol == "official"
                                     else strict_membership)[idx]
                base = output / protocol / target_site / destination_split
                (base / "image").mkdir(parents=True, exist_ok=True)
                (base / "mask").mkdir(parents=True, exist_ok=True)
                np.save(base / "image" / f"{stem}.npy", image)
                np.save(base / "mask" / f"{stem}.npy", mask)
            rows.append({
                "site": target_site,
                "index": idx,
                "official_split": official_membership[idx],
                "strict_split": strict_membership[idx],
                "image": str(image_path.resolve()),
                "mask": str(mask_path.resolve()),
                "image_sha256": sha256(image_path),
                "mask_sha256": sha256(mask_path),
            })
    output.mkdir(parents=True, exist_ok=True)
    with (output / "fundus_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def prostate_volume(image_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path)))
    label = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))
    label = (label > 0).astype(np.uint8)
    high = np.quantile(image, 0.99)
    low = np.min(image)
    image = np.clip(image, low, high)
    denom = max(float(high - low), 1e-8)
    image = ((image - low) / denom * 255).astype(np.uint8)
    images, labels = [], []
    for idx in range(1, label.shape[0] - 1):
        if not np.any(label[idx]):
            continue
        images.append(np.transpose(image[idx - 1:idx + 2], (1, 2, 0)))
        labels.append(label[idx])
    return np.asarray(images), np.asarray(labels)


def save_prostate_split(path: Path, images: list[np.ndarray], labels: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_array = np.concatenate(images, axis=0) if images else np.empty((0, 384, 384, 3), np.uint8)
    label_array = np.concatenate(labels, axis=0).astype(np.int_) if labels else np.empty((0, 384, 384), np.int_)
    np.save(path, {"images": image_array, "labels": label_array}, allow_pickle=True)


def prepare_prostate(source: Path, output: Path, seed: int) -> None:
    manifest: list[dict[str, object]] = []
    for site in PROSTATE_SITES:
        site_root = source / site
        mask_paths = sorted(p for p in site_root.glob("*.nii.gz") if "segmentation" in p.name.lower())
        volumes: list[tuple[Path, Path, np.ndarray, np.ndarray]] = []
        for mask_path in mask_paths:
            case_id = mask_path.name[:6]
            image_path = site_root / f"{case_id}.nii.gz"
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            images, labels = prostate_volume(image_path, mask_path)
            volumes.append((image_path, mask_path, images, labels))

        all_images = np.concatenate([v[2] for v in volumes], axis=0)
        all_labels = np.concatenate([v[3] for v in volumes], axis=0)
        train_n = int(max(0.6 * len(all_labels), 32))
        valid_n = int(0.2 * len(all_labels))
        official = {
            "train": (all_images[:train_n], all_labels[:train_n]),
            "valid": (all_images[train_n:train_n + valid_n], all_labels[train_n:train_n + valid_n]),
            "test": (all_images[-valid_n:], all_labels[-valid_n:]),
        }
        for split, (images, labels) in official.items():
            target = output / "official" / f"{site}_data" / f"{split}.npy"
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target, {"images": images, "labels": labels.astype(np.int_)}, allow_pickle=True)

        order = np.random.default_rng(seed).permutation(len(volumes)).tolist()
        test_n = max(1, int(np.ceil(0.2 * len(volumes))))
        valid_n = max(1, int(np.ceil(0.2 * len(volumes))))
        strict_ids = {
            "test": order[:test_n],
            "valid": order[test_n:test_n + valid_n],
            "train": order[test_n + valid_n:],
        }
        for split, ids in strict_ids.items():
            save_prostate_split(
                output / "strict" / f"{site}_data" / f"{split}.npy",
                [volumes[i][2] for i in ids],
                [volumes[i][3] for i in ids],
            )
            for i in ids:
                manifest.append({
                    "site": site,
                    "case": volumes[i][0].name,
                    "strict_split": split,
                    "image": str(volumes[i][0].resolve()),
                    "mask": str(volumes[i][1].resolve()),
                    "foreground_slices": int(len(volumes[i][3])),
                    "image_sha256": sha256(volumes[i][0]),
                    "mask_sha256": sha256(volumes[i][1]),
                })
    output.mkdir(parents=True, exist_ok=True)
    with (output / "prostate_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["fundus", "prostate", "all"], default="all")
    parser.add_argument("--fundus-source", type=Path)
    parser.add_argument("--prostate-source", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.dataset in {"fundus", "all"}:
        if args.fundus_source is None:
            parser.error("--fundus-source is required")
        prepare_fundus(args.fundus_source, args.output_root / "fundus", args.seed)
    if args.dataset in {"prostate", "all"}:
        if args.prostate_source is None:
            parser.error("--prostate-source is required")
        prepare_prostate(args.prostate_source, args.output_root / "prostate", args.seed)


if __name__ == "__main__":
    main()
