"""Validate prepared extended FedCLAM datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


EXPECTED_FUNDUS_COUNTS = {
    "REFUGE2": 1200,
    "G1020": 1020,
    "ORIGA": 650,
}

EXPECTED_NUCLEI_SITES = {"CoNSeP", "CPM17", "CryoNuSeg", "TNBC"}


def npy_files(path: Path) -> list[Path]:
    return sorted(path.glob("*.npy"))


def validate_site(site_root: Path, task: str) -> dict[str, object]:
    splits = {}
    total = 0
    for split in ("train", "valid", "test"):
        image_dir = site_root / split / "image"
        mask_dir = site_root / split / "mask"
        images = npy_files(image_dir) if image_dir.exists() else []
        masks = npy_files(mask_dir) if mask_dir.exists() else []
        if len(images) != len(masks):
            raise ValueError(f"{site_root.name}/{split}: {len(images)} images but {len(masks)} masks")
        if images:
            image = np.load(images[0])
            mask = np.load(masks[0])
            if image.shape != (384, 384, 3):
                raise ValueError(f"{images[0]} has shape {image.shape}, expected (384, 384, 3)")
            values = set(np.unique(mask).astype(int).tolist())
            if task == "fundus" and not values.issubset({0, 1, 2}):
                raise ValueError(f"{masks[0]} has invalid fundus labels: {sorted(values)}")
            if task == "nuclei" and not values.issubset({0, 1}):
                raise ValueError(f"{masks[0]} has invalid nuclei labels: {sorted(values)}")
        splits[split] = len(images)
        total += len(images)
    return {"splits": splits, "total": total}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", choices=["official", "strict"], default="strict")
    parser.add_argument("--require-complete-fundus", action="store_true")
    args = parser.parse_args()

    report = {"root": str(args.root.resolve()), "protocol": args.protocol, "fundus": {}, "nuclei": {}}
    fundus_root = args.root / "fundus" / args.protocol
    if fundus_root.exists():
        for site_root in sorted(path for path in fundus_root.iterdir() if path.is_dir()):
            site_report = validate_site(site_root, "fundus")
            expected = EXPECTED_FUNDUS_COUNTS.get(site_root.name)
            if expected is not None and site_report["total"] != expected:
                raise ValueError(f"{site_root.name}: expected {expected} samples, found {site_report['total']}")
            report["fundus"][site_root.name] = site_report
        if args.require_complete_fundus:
            missing = sorted(set(EXPECTED_FUNDUS_COUNTS) - set(report["fundus"]))
            if missing:
                raise ValueError("Missing fundus sites: " + ", ".join(missing))

    nuclei_root = args.root / "nuclei" / args.protocol
    if nuclei_root.exists():
        for site_root in sorted(path for path in nuclei_root.iterdir() if path.is_dir()):
            report["nuclei"][site_root.name] = validate_site(site_root, "nuclei")
        missing = sorted(EXPECTED_NUCLEI_SITES - set(report["nuclei"]))
        if missing:
            raise ValueError("Missing nuclei sites: " + ", ".join(missing))

    path = args.root / f"validation_report_{args.protocol}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
