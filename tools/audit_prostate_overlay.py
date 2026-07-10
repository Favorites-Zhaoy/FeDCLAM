import argparse
from pathlib import Path

import cv2
import numpy as np


def sample_indices(n: int, k: int, seed: int) -> list[int]:
    if n <= k:
        return list(range(n))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(n, size=k, replace=False).tolist())


def overlay(image: np.ndarray, label: np.ndarray) -> np.ndarray:
    middle = image[..., 1]
    base = cv2.cvtColor(middle.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    mask = (label > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    color = base.copy()
    color[mask.astype(bool)] = (0.55 * color[mask.astype(bool)] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)
    out = cv2.addWeighted(base, 0.45, color, 0.55, 0)
    cv2.drawContours(out, contours, -1, (255, 0, 0), 2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(r"D:\MLPractice\datasets\processed\prostate\official"))
    parser.add_argument("--output-dir", type=Path, default=Path(r"D:\MLPractice\FedCLAM\outputs\analysis\prostate_overlay_audit"))
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for site_dir in sorted(args.data_dir.glob("*_data")):
        site = site_dir.name.replace("_data", "")
        for split in ["train", "valid", "test"]:
            data = np.load(site_dir / f"{split}.npy", allow_pickle=True).item()
            images, labels = data["images"], data["labels"]
            target = args.output_dir / site / split
            target.mkdir(parents=True, exist_ok=True)
            for idx in sample_indices(len(labels), args.samples, args.seed):
                out = overlay(images[idx], labels[idx])
                cv2.imwrite(str(target / f"{idx:04d}_overlay.png"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            rows.append({
                "site": site,
                "split": split,
                "n": int(len(labels)),
                "image_shape": "x".join(map(str, images.shape)),
                "label_shape": "x".join(map(str, labels.shape)),
                "fg_ratio_mean": float(labels.mean()) if labels.size else float("nan"),
            })

    lines = ["# Prostate overlay audit", "", "| site | split | n | fg ratio |", "|---|---|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['site']} | {row['split']} | {row['n']} | {row['fg_ratio_mean']:.4f} |")
    (args.output_dir / "prostate_overlay_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote overlays to {args.output_dir}")


if __name__ == "__main__":
    main()
