from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyfed.network.unet import UNet


SITES = ["BIDMC", "HK", "I2CVB", "BMC", "RUNMC", "UCL"]


def read_manifest(path: Path, split: str) -> dict[str, list[dict[str, str]]]:
    rows_by_site: dict[str, list[dict[str, str]]] = {site: [] for site in SITES}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["split"] == split:
                rows_by_site[row["site"]].append(row)
    for site, rows in rows_by_site.items():
        rows.sort(key=lambda r: int(r["split_index"]))
    return rows_by_site


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = UNet(norm="in")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("server", checkpoint)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def update_stats(stats: dict[str, float], pred: np.ndarray, label: np.ndarray) -> None:
    pred_fg = pred == 1
    label_fg = label == 1
    pred_bg = pred == 0
    label_bg = label == 0
    stats["fg_inter"] += float(np.logical_and(pred_fg, label_fg).sum())
    stats["fg_pred"] += float(pred_fg.sum())
    stats["fg_label"] += float(label_fg.sum())
    stats["bg_inter"] += float(np.logical_and(pred_bg, label_bg).sum())
    stats["bg_pred"] += float(pred_bg.sum())
    stats["bg_label"] += float(label_bg.sum())
    stats["slices"] += 1
    stats["fg_pixels"] += float(label_fg.sum())
    stats["pixels"] += float(label.size)


def dice(inter: float, pred: float, label: float) -> float:
    return (2.0 * inter) / (pred + label + 1e-5)


def finalize(stats: dict[str, float]) -> dict[str, float]:
    fg = dice(stats["fg_inter"], stats["fg_pred"], stats["fg_label"])
    bg = dice(stats["bg_inter"], stats["bg_pred"], stats["bg_label"])
    return {
        "foreground_dice": fg,
        "background_dice": bg,
        "mean_bg_fg_dice": (fg + bg) / 2.0,
        "slices": int(stats["slices"]),
        "fg_ratio": stats["fg_pixels"] / max(stats["pixels"], 1.0),
    }


def evaluate_site(model: torch.nn.Module, device: torch.device, data_dir: Path, site: str,
                  manifest_rows: list[dict[str, str]], batch_size: int) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    payload = np.load(data_dir / f"{site}_data" / "test.npy", allow_pickle=True).item()
    images = payload["images"]
    labels = payload["labels"].astype(np.int64)
    if len(labels) != len(manifest_rows):
        raise AssertionError(f"{site}: test labels {len(labels)} != manifest rows {len(manifest_rows)}")

    site_stats = defaultdict(float)
    case_stats: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    with torch.no_grad():
        for start in range(0, len(labels), batch_size):
            end = min(start + batch_size, len(labels))
            batch_images = torch.from_numpy(images[start:end].transpose(0, 3, 1, 2)).float() / 255.0
            logits = model(batch_images.to(device))
            pred = torch.argmax(torch.softmax(logits, dim=1), dim=1).cpu().numpy().astype(np.int64)
            for offset, pred_slice in enumerate(pred):
                idx = start + offset
                label = labels[idx]
                row = manifest_rows[idx]
                case_id = row["case_id"]
                update_stats(site_stats, pred_slice, label)
                update_stats(case_stats[case_id], pred_slice, label)
    return finalize(site_stats), {case: finalize(stats) for case, stats in case_stats.items()}


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(r"D:\MLPractice\datasets\processed\prostate\official"))
    parser.add_argument("--manifest", type=Path, default=Path(r"D:\MLPractice\FedCLAM\outputs\prostate\analysis\prostate_official_split_manifest.csv"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(r"D:\MLPractice\FedCLAM\outputs\prostate\analysis\paperparams_case_audit"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}. Run audit_prostate_official_split.py first.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith("cpu") or torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    manifest = read_manifest(args.manifest, "test")

    site_rows: list[dict[str, object]] = []
    case_rows: list[dict[str, object]] = []
    for site in SITES:
        site_result, case_result = evaluate_site(model, device, args.data_dir, site, manifest[site], args.batch_size)
        site_rows.append({"site": site, **site_result})
        for case_id, result in sorted(case_result.items()):
            case_rows.append({"site": site, "case_id": case_id, **result})

    write_csv(
        args.output_dir / "prostate_per_site_dice.csv",
        site_rows,
        ["site", "foreground_dice", "background_dice", "mean_bg_fg_dice", "slices", "fg_ratio"],
    )
    write_csv(
        args.output_dir / "prostate_per_case_dice.csv",
        case_rows,
        ["site", "case_id", "foreground_dice", "background_dice", "mean_bg_fg_dice", "slices", "fg_ratio"],
    )

    site_mean = float(np.mean([row["mean_bg_fg_dice"] for row in site_rows]))
    site_fg_mean = float(np.mean([row["foreground_dice"] for row in site_rows]))
    worst_cases = sorted(case_rows, key=lambda r: float(r["foreground_dice"]))[:12]
    lines = [
        "# Prostate checkpoint case audit",
        "",
        f"Checkpoint: `{args.checkpoint}`",
        f"Mean site foreground Dice: {100 * site_fg_mean:.2f}%",
        f"Mean site background+foreground Dice: {100 * site_mean:.2f}%",
        "",
        "## Per-site Dice",
        "",
        "| site | fg Dice | bg+fg Dice | slices | fg ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in site_rows:
        lines.append(
            f"| {row['site']} | {100 * float(row['foreground_dice']):.2f} | "
            f"{100 * float(row['mean_bg_fg_dice']):.2f} | {row['slices']} | {float(row['fg_ratio']):.4f} |"
        )
    lines += [
        "",
        "## Worst foreground-Dice cases",
        "",
        "| site | case | fg Dice | bg+fg Dice | slices | fg ratio |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in worst_cases:
        lines.append(
            f"| {row['site']} | {row['case_id']} | {100 * float(row['foreground_dice']):.2f} | "
            f"{100 * float(row['mean_bg_fg_dice']):.2f} | {row['slices']} | {float(row['fg_ratio']):.4f} |"
        )
    (args.output_dir / "prostate_case_failure_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.output_dir)


if __name__ == "__main__":
    main()
