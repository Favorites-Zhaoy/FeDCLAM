from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


FUNDUS_COUNTS = {"Drishti-GS": 101, "RIM-ONE": 159, "REFUGE_t": 400, "REFUGE_v": 400}
PROSTATE_SITES = ["BIDMC", "HK", "I2CVB", "BMC", "RUNMC", "UCL"]


def validate_fundus(root: Path) -> dict:
    report = {}
    for protocol in ("official", "strict"):
        report[protocol] = {}
        for site, expected in FUNDUS_COUNTS.items():
            split_names = ("train", "test") if protocol == "official" else ("train", "valid", "test")
            seen = set()
            counts = {}
            for split in split_names:
                images = sorted((root / protocol / site / split / "image").glob("*.npy"))
                masks = sorted((root / protocol / site / split / "mask").glob("*.npy"))
                if [x.name for x in images] != [x.name for x in masks]:
                    raise AssertionError(f"Pair mismatch: {protocol}/{site}/{split}")
                names = {x.name for x in images}
                if seen & names:
                    raise AssertionError(f"Split overlap: {protocol}/{site}/{split}")
                seen |= names
                counts[split] = len(images)
                for mask_path in masks[:3]:
                    if not set(np.unique(np.load(mask_path)).tolist()).issubset({0, 1, 2}):
                        raise AssertionError(f"Invalid labels in {mask_path}")
            if len(seen) != expected:
                raise AssertionError(f"{site}: expected {expected}, found {len(seen)}")
            report[protocol][site] = counts
    return report


def validate_prostate(root: Path) -> dict:
    report = {}
    for protocol in ("official", "strict"):
        report[protocol] = {}
        for site in PROSTATE_SITES:
            counts = {}
            for split in ("train", "valid", "test"):
                path = root / protocol / f"{site}_data" / f"{split}.npy"
                payload = np.load(path, allow_pickle=True).item()
                images, labels = payload["images"], payload["labels"]
                if len(images) != len(labels) or images.shape[-1] != 3:
                    raise AssertionError(f"Invalid arrays in {path}")
                if not set(np.unique(labels).tolist()).issubset({0, 1}):
                    raise AssertionError(f"Invalid labels in {path}")
                counts[split] = len(images)
            report[protocol][site] = counts
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "fundus": validate_fundus(args.root / "fundus"),
        "prostate": validate_prostate(args.root / "prostate"),
    }
    destination = args.root / "validation_report.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
