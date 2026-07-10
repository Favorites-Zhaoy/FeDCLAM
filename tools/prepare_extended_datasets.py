"""Prepare extended FedCLAM generalization datasets.

This script writes the same npy protocol consumed by the FedCLAM loaders:

  fundus/{official,strict}/{site}/{split}/image/*.npy
  fundus/{official,strict}/{site}/{split}/mask/*.npy
  nuclei/{official,strict}/{site}/{split}/image/*.npy
  nuclei/{official,strict}/{site}/{split}/mask/*.npy

G1020 and ORIGA are only prepared when real OD/OC ground-truth masks are
provided. Paths containing "predict" are rejected to avoid accidentally using
model-generated masks as ground truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.io as sio
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_SUFFIXES = IMAGE_SUFFIXES | {".npy"}
TARGET_SIZE = (384, 384)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_files(path: Path, suffixes=IMAGE_SUFFIXES) -> list[Path]:
    if path is None or not path.exists():
        return []
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)


def key_candidates(path: Path) -> set[str]:
    stem = path.stem.lower()
    keys = {stem}
    for token in ("_mask", "-mask", "_gt", "-gt", "_disc", "-disc", "_disk", "-disk", "_cup", "-cup"):
        if stem.endswith(token):
            keys.add(stem[: -len(token)])
    if stem.isdigit():
        keys.add(stem.zfill(3))
    return keys


def index_by_key(paths: list[Path]) -> dict[str, Path]:
    result = {}
    collisions = defaultdict(list)
    for path in paths:
        for key in key_candidates(path):
            if key in result and result[key] != path:
                collisions[key].append(path)
            else:
                result[key] = path
    if collisions:
        sample = ", ".join(sorted(collisions.keys())[:5])
        raise ValueError(f"Ambiguous mask keys: {sample}")
    return result


def reject_predicted(paths: list[Path]) -> None:
    bad = [str(path) for path in paths if "predict" in str(path).lower()]
    if bad:
        raise ValueError("Predicted masks are not allowed as ground truth: " + bad[0])


def read_rgb(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image = image.resize(TARGET_SIZE, Image.Resampling.BICUBIC)
    return np.asarray(image)


def read_mask(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        mask = np.load(path)
    else:
        mask = np.asarray(Image.open(path))
    if mask is None:
        raise ValueError(f"Unreadable mask: {path}")
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask


def normalize_fundus_combined(mask: np.ndarray) -> np.ndarray:
    values = set(np.unique(mask).astype(int).tolist())
    if values.issubset({0, 1, 2}):
        return mask.astype(np.uint8)
    result = np.zeros(mask.shape, dtype=np.uint8)
    result[mask < 64] = 2
    result[(mask >= 64) & (mask < 192)] = 1
    result[mask >= 192] = 0
    return result


def combine_disc_cup(disc: np.ndarray, cup: np.ndarray) -> np.ndarray:
    if disc.ndim == 3:
        disc = disc[..., 0]
    if cup.ndim == 3:
        cup = cup[..., 0]
    result = np.zeros(disc.shape, dtype=np.uint8)
    result[disc > 0] = 1
    result[cup > 0] = 2
    return result


def resize_mask(mask: np.ndarray) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8))
    return np.asarray(image.resize(TARGET_SIZE, Image.Resampling.NEAREST))


def save_npy_pair(base: Path, site: str, split: str, stem: str, image: np.ndarray, mask: np.ndarray) -> None:
    out = base / site / split
    (out / "image").mkdir(parents=True, exist_ok=True)
    (out / "mask").mkdir(parents=True, exist_ok=True)
    np.save(out / "image" / f"{stem}.npy", image)
    np.save(out / "mask" / f"{stem}.npy", mask)


def clear_prepared_site(output_root: Path, task: str, site: str) -> None:
    for protocol in ("official", "strict"):
        path = output_root / task / protocol / site
        if path.exists():
            shutil.rmtree(path)


def strict_membership(count: int, train_indices: list[int], test_indices: list[int], seed: int) -> dict[int, str]:
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(train_indices).tolist()
    valid_n = max(1, int(np.floor(0.1 * len(train_indices)))) if len(train_indices) > 1 else 0
    valid = set(shuffled[:valid_n])
    membership = {idx: "valid" if idx in valid else "train" for idx in train_indices}
    membership.update({idx: "test" for idx in test_indices})
    assert len(membership) == count
    return membership


def copy_existing_fundus(existing_root: Path | None, output_root: Path) -> None:
    if existing_root is None:
        return
    for protocol in ("official", "strict"):
        src = existing_root / protocol
        dst = output_root / "fundus" / protocol
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)


def collect_mask_paths(combined: Path | None, disc: Path | None, cup: Path | None) -> tuple[dict[str, Path], dict[str, Path], dict[str, Path]]:
    combined_paths = image_files(combined, MASK_SUFFIXES)
    disc_paths = image_files(disc, MASK_SUFFIXES)
    cup_paths = image_files(cup, MASK_SUFFIXES)
    reject_predicted(combined_paths + disc_paths + cup_paths)
    return index_by_key(combined_paths), index_by_key(disc_paths), index_by_key(cup_paths)


def fundus_mask_for(image_path: Path, combined: dict[str, Path], disc: dict[str, Path], cup: dict[str, Path]) -> tuple[np.ndarray, str]:
    keys = key_candidates(image_path)
    for key in keys:
        if key in combined:
            mask = normalize_fundus_combined(read_mask(combined[key]))
            return mask, str(combined[key])
    for key in keys:
        if key in disc and key in cup:
            mask = combine_disc_cup(read_mask(disc[key]), read_mask(cup[key]))
            return mask, f"{disc[key]}|{cup[key]}"
    raise FileNotFoundError(f"No OD/OC ground-truth mask found for {image_path}")


def save_fundus_site(
    output_root: Path,
    site: str,
    samples: list[tuple[str, Path, np.ndarray, str]],
    seed: int,
    rows: list[dict[str, object]],
) -> None:
    clear_prepared_site(output_root, "fundus", site)
    train_indices = [idx for idx, item in enumerate(samples) if item[0] == "train"]
    test_indices = [idx for idx, item in enumerate(samples) if item[0] == "test"]
    valid_indices = [idx for idx, item in enumerate(samples) if item[0] == "valid"]
    if valid_indices:
        strict = {idx: "train" for idx in train_indices}
        strict.update({idx: "valid" for idx in valid_indices})
        strict.update({idx: "test" for idx in test_indices})
    else:
        strict = strict_membership(len(samples), train_indices, test_indices, seed)
    official = {
        idx: ("test" if item[0] == "test" else "train")
        for idx, item in enumerate(samples)
    }
    for idx, (_, image_path, mask, mask_source) in enumerate(samples):
        image = read_rgb(image_path)
        mask = resize_mask(mask)
        stem = f"{idx:04d}_{image_path.stem}"
        for protocol, membership in (("official", official), ("strict", strict)):
            save_npy_pair(output_root / "fundus" / protocol, site, membership[idx], stem, image, mask)
        rows.append({
            "task": "fundus",
            "site": site,
            "index": idx,
            "official_split": official[idx],
            "strict_split": strict[idx],
            "image": str(image_path.resolve()),
            "mask": mask_source,
            "image_sha256": sha256(image_path),
            "mask_sha256": "" if "|" in mask_source else sha256(Path(mask_source)),
        })


def prepare_refuge2(root: Path, output_root: Path, seed: int, rows: list[dict[str, object]]) -> None:
    site_root = root / "REFUGE2" / "REFUGE2"
    samples = []
    for split, normalized in (("train", "train"), ("val", "valid"), ("test", "test")):
        images = image_files(site_root / split / "images")
        masks = index_by_key(image_files(site_root / split / "mask", MASK_SUFFIXES))
        for image_path in images:
            mask_path = next((masks[key] for key in key_candidates(image_path) if key in masks), None)
            if mask_path is None:
                raise FileNotFoundError(f"No REFUGE2 mask for {image_path}")
            samples.append((normalized, image_path, normalize_fundus_combined(read_mask(mask_path)), str(mask_path)))
    save_fundus_site(output_root, "REFUGE2", samples, seed, rows)


def prepare_g1020_or_origa(
    root: Path,
    output_root: Path,
    site: str,
    image_root: Path,
    combined_root: Path | None,
    disc_root: Path | None,
    cup_root: Path | None,
    seed: int,
    rows: list[dict[str, object]],
) -> bool:
    combined, disc, cup = collect_mask_paths(combined_root, disc_root, cup_root)
    if not combined and not (disc and cup):
        rows.append({"task": "fundus", "site": site, "status": "skipped_missing_gt"})
        return False
    samples = []
    if site == "G1020":
        for split in ("train", "test"):
            for image_path in image_files(image_root / split):
                mask, source = fundus_mask_for(image_path, combined, disc, cup)
                samples.append((split, image_path, mask, source))
    else:
        for split in ("train", "test"):
            for image_path in image_files(image_root / split):
                mask, source = fundus_mask_for(image_path, combined, disc, cup)
                samples.append((split, image_path, mask, source))
    expected = {"G1020": 1020, "ORIGA": 650}[site]
    if len(samples) != expected:
        raise ValueError(f"{site}: expected {expected} image/mask pairs, found {len(samples)}")
    save_fundus_site(output_root, site, samples, seed, rows)
    return True


def load_mat_mask(data: bytes) -> np.ndarray:
    mat = sio.loadmat(io.BytesIO(data))
    if "inst_map" in mat:
        return mat["inst_map"]
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        if isinstance(value, np.ndarray) and value.ndim == 2:
            return value
    raise ValueError("No 2D mask found in .mat file")


def read_zip_image(zip_file: zipfile.ZipFile, name: str) -> np.ndarray:
    with zip_file.open(name) as handle:
        image = Image.open(handle).convert("RGB")
        image = image.resize(TARGET_SIZE, Image.Resampling.BICUBIC)
        image = np.asarray(image)
    return image


def zip_entries_by_stem(zip_file: zipfile.ZipFile, contains: str, suffix: str) -> dict[str, str]:
    result = {}
    for entry in zip_file.namelist():
        if "__MACOSX/" in entry or Path(entry).name.startswith("._"):
            continue
        if contains in entry and entry.lower().endswith(suffix):
            result[Path(entry).stem.lower()] = entry
    return result


def save_nuclei_samples(output_root: Path, site: str, samples: list[tuple[str, str, np.ndarray, np.ndarray]], seed: int, rows: list[dict[str, object]]) -> None:
    clear_prepared_site(output_root, "nuclei", site)
    train_indices = [idx for idx, item in enumerate(samples) if item[0] == "train"]
    test_indices = [idx for idx, item in enumerate(samples) if item[0] == "test"]
    valid_indices = [idx for idx, item in enumerate(samples) if item[0] == "valid"]
    if train_indices and test_indices and not valid_indices and len(train_indices) > 1:
        shuffled = np.random.default_rng(seed).permutation(train_indices).tolist()
        valid_n = max(1, int(np.floor(0.1 * len(train_indices))))
        valid_indices = shuffled[:valid_n]
        train_indices = [idx for idx in train_indices if idx not in set(valid_indices)]
    if not train_indices and not test_indices:
        order = np.random.default_rng(seed).permutation(len(samples)).tolist()
        test_n = max(1, int(np.floor(0.2 * len(samples))))
        valid_n = max(1, int(np.floor(0.1 * len(samples))))
        test_indices = order[:test_n]
        valid_indices = order[test_n:test_n + valid_n]
        train_indices = order[test_n + valid_n:]
    strict = {idx: "train" for idx in train_indices}
    strict.update({idx: "valid" for idx in valid_indices})
    strict.update({idx: "test" for idx in test_indices})
    official = {idx: ("test" if strict[idx] == "test" else "train") for idx in range(len(samples))}
    for idx, (_, source, image, mask) in enumerate(samples):
        mask = resize_mask((mask > 0).astype(np.uint8))
        stem = f"{idx:04d}_{Path(source).stem}"
        for protocol, membership in (("official", official), ("strict", strict)):
            save_npy_pair(output_root / "nuclei" / protocol, site, membership[idx], stem, image, mask)
        rows.append({
            "task": "nuclei",
            "site": site,
            "index": idx,
            "official_split": official[idx],
            "strict_split": strict[idx],
            "source": source,
        })


def prepare_consep(root: Path, output_root: Path, seed: int, rows: list[dict[str, object]]) -> None:
    path = root / "CoNSeP" / "OpenDataLab___CoNSeP" / "raw" / "consep.zip"
    with zipfile.ZipFile(path) as zf:
        samples = []
        for split in ("Train", "Test"):
            images = zip_entries_by_stem(zf, f"CoNSeP/{split}/Images/", ".png")
            labels = zip_entries_by_stem(zf, f"CoNSeP/{split}/Labels/", ".mat")
            for stem, image_entry in sorted(images.items()):
                if stem not in labels:
                    raise FileNotFoundError(f"No CoNSeP label for {image_entry}")
                image = read_zip_image(zf, image_entry)
                mask = load_mat_mask(zf.read(labels[stem]))
                samples.append(("train" if split == "Train" else "test", image_entry, image, mask))
    save_nuclei_samples(output_root, "CoNSeP", samples, seed, rows)


def prepare_cpm17(root: Path, output_root: Path, seed: int, rows: list[dict[str, object]]) -> None:
    path = next((root / "Cpm17").glob("cpm17*.zip"))
    with zipfile.ZipFile(path) as zf:
        samples = []
        for split in ("train", "test"):
            images = zip_entries_by_stem(zf, f"cpm17/{split}/Images/", ".png")
            labels = zip_entries_by_stem(zf, f"cpm17/{split}/Labels/", ".mat")
            for stem, image_entry in sorted(images.items()):
                if stem not in labels:
                    raise FileNotFoundError(f"No CPM17 label for {image_entry}")
                image = read_zip_image(zf, image_entry)
                mask = load_mat_mask(zf.read(labels[stem]))
                samples.append((split, image_entry, image, mask))
    save_nuclei_samples(output_root, "CPM17", samples, seed, rows)


def prepare_tnbc(root: Path, output_root: Path, seed: int, rows: list[dict[str, object]]) -> None:
    base = root / "TNBC" / "TNBC_dataset"
    images = image_files(base, IMAGE_SUFFIXES)
    slide_images = [path for path in images if path.parent.name.startswith("Slide_")]
    masks = index_by_key([path for path in images if path.parent.name.startswith("GT_")])
    samples = []
    for image_path in sorted(slide_images):
        mask_path = next((masks[key] for key in key_candidates(image_path) if key in masks), None)
        if mask_path is None:
            continue
        samples.append(("", str(image_path), read_rgb(image_path), read_mask(mask_path)))
    save_nuclei_samples(output_root, "TNBC", samples, seed, rows)


def prepare_cryonuseg(root: Path, output_root: Path, seed: int, rows: list[dict[str, object]]) -> None:
    image_root = root / "CryoNuSeg" / "tissue images"
    mask_root = root / "CryoNuSeg" / "Annotator 1 (biologist)" / "label masks"
    masks = index_by_key(image_files(mask_root, MASK_SUFFIXES))
    samples = []
    for image_path in image_files(image_root):
        mask_path = next((masks[key] for key in key_candidates(image_path) if key in masks), None)
        if mask_path is None:
            raise FileNotFoundError(f"No CryoNuSeg label mask for {image_path}")
        samples.append(("", str(image_path), read_rgb(image_path), read_mask(mask_path)))
    save_nuclei_samples(output_root, "CryoNuSeg", samples, seed, rows)


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["fundus", "nuclei", "all"], default="all")
    parser.add_argument("--extended-root", type=Path, default=Path(r"D:\MLPractice\ExtendedDatasets"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--existing-fundus-root", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--require-complete-fundus", action="store_true")
    parser.add_argument("--g1020-mask-root", type=Path)
    parser.add_argument("--g1020-disc-mask-root", type=Path)
    parser.add_argument("--g1020-cup-mask-root", type=Path)
    parser.add_argument("--origa-mask-root", type=Path)
    parser.add_argument("--origa-disc-mask-root", type=Path)
    parser.add_argument("--origa-cup-mask-root", type=Path)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    prepared = {}
    if args.dataset in {"fundus", "all"}:
        copy_existing_fundus(args.existing_fundus_root, args.output_root)
        prepare_refuge2(args.extended_root, args.output_root, args.seed, rows)
        prepared["G1020"] = prepare_g1020_or_origa(
            args.extended_root, args.output_root, "G1020",
            args.extended_root / "G1020" / "Images_splitted",
            args.g1020_mask_root, args.g1020_disc_mask_root, args.g1020_cup_mask_root,
            args.seed, rows,
        )
        prepared["ORIGA"] = prepare_g1020_or_origa(
            args.extended_root, args.output_root, "ORIGA",
            args.extended_root / "ORIGA" / "ORIGA",
            args.origa_mask_root, args.origa_disc_mask_root, args.origa_cup_mask_root,
            args.seed, rows,
        )
        if args.require_complete_fundus and not all(prepared.values()):
            missing = ", ".join(site for site, ok in prepared.items() if not ok)
            raise SystemExit(f"Missing real OD/OC ground truth for: {missing}")
    if args.dataset in {"nuclei", "all"}:
        prepare_consep(args.extended_root, args.output_root, args.seed, rows)
        prepare_cpm17(args.extended_root, args.output_root, args.seed, rows)
        prepare_tnbc(args.extended_root, args.output_root, args.seed, rows)
        prepare_cryonuseg(args.extended_root, args.output_root, args.seed, rows)
    write_manifest(args.output_root / "extended_manifest.csv", rows)
    summary = {
        "output_root": str(args.output_root.resolve()),
        "prepared_fundus_gt": prepared,
    }
    (args.output_root / "extended_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
