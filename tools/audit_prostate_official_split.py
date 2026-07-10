from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import SimpleITK as sitk


SITES = ["BIDMC", "HK", "I2CVB", "BMC", "RUNMC", "UCL"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def foreground_slices(mask_path: Path) -> list[tuple[int, int, float]]:
    label = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))
    label = (label > 0).astype(np.uint8)
    rows = []
    for slice_idx in range(1, label.shape[0] - 1):
        mask = label[slice_idx]
        if not np.any(mask):
            continue
        fg_pixels = int(mask.sum())
        rows.append((slice_idx, fg_pixels, float(fg_pixels / mask.size)))
    return rows


def split_for_index(index: int, total: int) -> tuple[str, int]:
    train_n = int(max(0.6 * total, 32))
    valid_n = int(0.2 * total)
    test_n = int(0.2 * total)
    if index < train_n:
        return "train", index
    if index < train_n + valid_n:
        return "valid", index - train_n
    if index >= total - test_n:
        return "test", index - (total - test_n)
    return "unused", index - (train_n + valid_n)


def build_rows(source: Path, processed: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for site in SITES:
        site_root = source / site
        mask_paths = sorted(p for p in site_root.glob("*.nii.gz") if "segmentation" in p.name.lower())
        entries: list[dict[str, object]] = []
        for volume_index, mask_path in enumerate(mask_paths):
            case_id = mask_path.name[:6]
            image_path = site_root / f"{case_id}.nii.gz"
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            image_hash = sha256(image_path)
            mask_hash = sha256(mask_path)
            for case_slice_index, (slice_index, fg_pixels, fg_ratio) in enumerate(foreground_slices(mask_path)):
                entries.append({
                    "site": site,
                    "volume_index": volume_index,
                    "case_id": case_id,
                    "case_slice_index": case_slice_index,
                    "slice_index": slice_index,
                    "fg_pixels": fg_pixels,
                    "fg_ratio": fg_ratio,
                    "source_image": str(image_path.resolve()),
                    "source_mask": str(mask_path.resolve()),
                    "image_sha256": image_hash,
                    "mask_sha256": mask_hash,
                })
        split_counts = {"train": 0, "valid": 0, "test": 0, "unused": 0}
        total = len(entries)
        expected = {}
        for split in ["train", "valid", "test"]:
            payload = np.load(processed / f"{site}_data" / f"{split}.npy", allow_pickle=True).item()
            expected[split] = int(len(payload["labels"]))
        for global_index, row in enumerate(entries):
            split, split_index = split_for_index(global_index, total)
            row = dict(row)
            row.update({
                "global_slice_index": global_index,
                "split": split,
                "split_index": split_index,
            })
            split_counts[split] += 1
            rows.append(row)
        observed = {key: split_counts[key] for key in ["train", "valid", "test"]}
        if observed != expected:
            raise AssertionError(f"{site}: manifest split counts {observed} != processed counts {expected}")
    return rows


def write_summary(rows: list[dict[str, object]], output_dir: Path) -> None:
    lines = [
        "# Prostate official split manifest audit",
        "",
        "| site | split | slices | cases | mean fg ratio |",
        "|---|---|---:|---:|---:|",
    ]
    for site in SITES:
        for split in ["train", "valid", "test", "unused"]:
            selected = [r for r in rows if r["site"] == site and r["split"] == split]
            cases = {r["case_id"] for r in selected}
            fg = np.asarray([float(r["fg_ratio"]) for r in selected], dtype=float)
            lines.append(f"| {site} | {split} | {len(selected)} | {len(cases)} | {float(fg.mean()) if fg.size else float('nan'):.4f} |")
    (output_dir / "prostate_official_split_manifest.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path(r"D:\MLPractice\datasets\Prostate MRI"))
    parser.add_argument("--processed-dir", type=Path, default=Path(r"D:\MLPractice\datasets\processed\prostate\official"))
    parser.add_argument("--output-dir", type=Path, default=Path(r"D:\MLPractice\FedCLAM\outputs\prostate\analysis"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args.source_dir, args.processed_dir)
    csv_path = args.output_dir / "prostate_official_split_manifest.csv"
    fieldnames = [
        "site", "split", "split_index", "global_slice_index", "volume_index", "case_id",
        "case_slice_index", "slice_index", "fg_pixels", "fg_ratio", "source_image",
        "source_mask", "image_sha256", "mask_sha256",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_summary(rows, args.output_dir)
    print(csv_path)


if __name__ == "__main__":
    main()
