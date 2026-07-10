import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np

from prepare_datasets import normalize_fundus_mask


SITES_TO_COPY = ["Drishti-GS", "REFUGE_t", "REFUGE_v"]


def copy_existing_sites(source_official: Path, output_official: Path) -> None:
    for site in SITES_TO_COPY:
        src = source_official / site
        dst = output_official / site
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def crop_to_mask_half(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    h, w = mask.shape
    xs = np.where(mask > 0)[1]
    if len(xs) == 0:
        raise ValueError("RIM-ONE mask has no foreground")
    center_x = float(xs.mean())
    half = w // 2
    if center_x < half:
        crop = (0, half)
        side = "left"
    else:
        crop = (half, w)
        side = "right"
    x0, x1 = crop
    return image[:, x0:x1], mask[:, x0:x1], side


def regenerate_rimone(manifest_path: Path, output_official: Path) -> None:
    if (output_official / "RIM-ONE").exists():
        shutil.rmtree(output_official / "RIM-ONE")

    rows_out = []
    with manifest_path.open("r", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["site"] == "RIM-ONE"]

    for row in rows:
        image_path = Path(row["image"])
        mask_path = Path(row["mask"])
        split = row["official_split"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None:
            raise ValueError(f"Unreadable pair: {image_path}, {mask_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = normalize_fundus_mask(mask)
        image_crop, mask_crop, side = crop_to_mask_half(image, mask)
        image_resized = cv2.resize(image_crop, (384, 384), interpolation=cv2.INTER_CUBIC)
        mask_resized = cv2.resize(mask_crop, (384, 384), interpolation=cv2.INTER_NEAREST)

        stem = f"{int(row['index']):04d}_{image_path.stem}"
        base = output_official / "RIM-ONE" / split
        (base / "image").mkdir(parents=True, exist_ok=True)
        (base / "mask").mkdir(parents=True, exist_ok=True)
        np.save(base / "image" / f"{stem}.npy", image_resized)
        np.save(base / "mask" / f"{stem}.npy", mask_resized)
        rows_out.append({
            "site": "RIM-ONE",
            "index": row["index"],
            "official_split": split,
            "source_image": str(image_path),
            "source_mask": str(mask_path),
            "crop_side": side,
            "source_shape": "x".join(map(str, image.shape)),
            "output_shape": "x".join(map(str, image_resized.shape)),
        })

    with (output_official.parent / "rimone_crop_diagnostic_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-official", type=Path, default=Path(r"D:\MLPractice\datasets\processed\fundus\official"))
    parser.add_argument("--source-manifest", type=Path, default=Path(r"D:\MLPractice\datasets\processed\fundus\fundus_manifest.csv"))
    parser.add_argument("--output-root", type=Path, default=Path(r"D:\MLPractice\datasets\processed\fundus_rimone_crop_diagnostic"))
    args = parser.parse_args()

    output_official = args.output_root / "official"
    output_official.mkdir(parents=True, exist_ok=True)
    copy_existing_sites(args.source_official, output_official)
    regenerate_rimone(args.source_manifest, output_official)
    print(f"Wrote diagnostic dataset to {output_official}")


if __name__ == "__main__":
    main()
