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
PAPER_SITE_DICE = {
    "BIDMC": 92.85,
    "HK": 94.61,
    "I2CVB": 94.77,
    "BMC": 95.71,
    "RUNMC": 94.27,
    "UCL": 90.21,
}


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
    model.load_state_dict(checkpoint.get("server", checkpoint))
    model.to(device)
    model.eval()
    return model


def dice_from_counts(inter: float, pred: float, label: float) -> float:
    return (2.0 * inter) / (pred + label + 1e-5)


def slice_dices(pred: np.ndarray, label: np.ndarray) -> tuple[float, float, float]:
    fg_pred = pred == 1
    fg_label = label == 1
    bg_pred = pred == 0
    bg_label = label == 0
    fg = dice_from_counts(
        float(np.logical_and(fg_pred, fg_label).sum()),
        float(fg_pred.sum()),
        float(fg_label.sum()),
    )
    bg = dice_from_counts(
        float(np.logical_and(bg_pred, bg_label).sum()),
        float(bg_pred.sum()),
        float(bg_label.sum()),
    )
    return fg, bg, (fg + bg) / 2.0


def add_counts(stats: dict[str, float], pred: np.ndarray, label: np.ndarray) -> None:
    fg_pred = pred == 1
    fg_label = label == 1
    bg_pred = pred == 0
    bg_label = label == 0
    stats["fg_inter"] += float(np.logical_and(fg_pred, fg_label).sum())
    stats["fg_pred"] += float(fg_pred.sum())
    stats["fg_label"] += float(fg_label.sum())
    stats["bg_inter"] += float(np.logical_and(bg_pred, bg_label).sum())
    stats["bg_pred"] += float(bg_pred.sum())
    stats["bg_label"] += float(bg_label.sum())
    stats["pixels"] += float(label.size)
    stats["fg_pixels"] += float(fg_label.sum())


def pooled_from_counts(stats: dict[str, float]) -> tuple[float, float, float]:
    fg = dice_from_counts(stats["fg_inter"], stats["fg_pred"], stats["fg_label"])
    bg = dice_from_counts(stats["bg_inter"], stats["bg_pred"], stats["bg_label"])
    return fg, bg, (fg + bg) / 2.0


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def evaluate(
    model: torch.nn.Module,
    device: torch.device,
    data_dir: Path,
    manifest: dict[str, list[dict[str, str]]],
    batch_size: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    site_rows: list[dict[str, object]] = []
    case_rows: list[dict[str, object]] = []
    slice_rows: list[dict[str, object]] = []

    with torch.no_grad():
        for site in SITES:
            payload = np.load(data_dir / f"{site}_data" / "test.npy", allow_pickle=True).item()
            images = payload["images"]
            labels = payload["labels"].astype(np.int64)
            rows = manifest[site]
            if len(labels) != len(rows):
                raise AssertionError(f"{site}: test labels {len(labels)} != manifest rows {len(rows)}")

            site_counts: dict[str, float] = defaultdict(float)
            case_counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
            site_slice_fg: list[float] = []
            site_slice_bg: list[float] = []
            site_slice_mean: list[float] = []
            case_slice_fg: dict[str, list[float]] = defaultdict(list)
            case_slice_bg: dict[str, list[float]] = defaultdict(list)
            case_slice_mean: dict[str, list[float]] = defaultdict(list)

            for start in range(0, len(labels), batch_size):
                end = min(start + batch_size, len(labels))
                batch = torch.from_numpy(images[start:end].transpose(0, 3, 1, 2)).float() / 255.0
                logits = model(batch.to(device))
                pred = torch.argmax(torch.softmax(logits, dim=1), dim=1).cpu().numpy().astype(np.int64)
                for offset, pred_slice in enumerate(pred):
                    idx = start + offset
                    label = labels[idx]
                    row = rows[idx]
                    case_id = row["case_id"]
                    fg, bg, bgfg = slice_dices(pred_slice, label)
                    add_counts(site_counts, pred_slice, label)
                    add_counts(case_counts[case_id], pred_slice, label)
                    site_slice_fg.append(fg)
                    site_slice_bg.append(bg)
                    site_slice_mean.append(bgfg)
                    case_slice_fg[case_id].append(fg)
                    case_slice_bg[case_id].append(bg)
                    case_slice_mean[case_id].append(bgfg)
                    slice_rows.append(
                        {
                            "site": site,
                            "case_id": case_id,
                            "split_index": row["split_index"],
                            "foreground_dice": fg,
                            "background_dice": bg,
                            "mean_bg_fg_dice": bgfg,
                            "fg_ratio": float((label == 1).sum() / label.size),
                        }
                    )

            pooled_fg, pooled_bg, pooled_bgfg = pooled_from_counts(site_counts)
            site_rows.append(
                {
                    "site": site,
                    "slices": len(labels),
                    "cases": len(case_counts),
                    "slice_avg_foreground_dice": mean(site_slice_fg),
                    "slice_avg_background_dice": mean(site_slice_bg),
                    "slice_avg_mean_bg_fg_dice": mean(site_slice_mean),
                    "pooled_foreground_dice": pooled_fg,
                    "pooled_background_dice": pooled_bg,
                    "pooled_mean_bg_fg_dice": pooled_bgfg,
                    "paper_dice": PAPER_SITE_DICE[site] / 100.0,
                }
            )

            for case_id in sorted(case_counts):
                pooled_fg, pooled_bg, pooled_bgfg = pooled_from_counts(case_counts[case_id])
                case_rows.append(
                    {
                        "site": site,
                        "case_id": case_id,
                        "slices": len(case_slice_mean[case_id]),
                        "slice_avg_foreground_dice": mean(case_slice_fg[case_id]),
                        "slice_avg_background_dice": mean(case_slice_bg[case_id]),
                        "slice_avg_mean_bg_fg_dice": mean(case_slice_mean[case_id]),
                        "pooled_foreground_dice": pooled_fg,
                        "pooled_background_dice": pooled_bg,
                        "pooled_mean_bg_fg_dice": pooled_bgfg,
                    }
                )

    return site_rows, case_rows, slice_rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_summary(rows: list[dict[str, object]], name: str, value: float, paper: float) -> None:
    rows.append(
        {
            "metric": name,
            "value": value,
            "paper_avg": paper,
            "delta_vs_paper": value - paper,
        }
    )


def build_summary(site_rows: list[dict[str, object]], case_rows: list[dict[str, object]], slice_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    paper_avg = mean([PAPER_SITE_DICE[site] / 100.0 for site in SITES])
    summary: list[dict[str, object]] = []

    add_summary(summary, "site_macro__slice_avg__mean_bg_fg", mean([float(r["slice_avg_mean_bg_fg_dice"]) for r in site_rows]), paper_avg)
    add_summary(summary, "site_macro__slice_avg__foreground_only", mean([float(r["slice_avg_foreground_dice"]) for r in site_rows]), paper_avg)
    add_summary(summary, "site_macro__pooled__mean_bg_fg", mean([float(r["pooled_mean_bg_fg_dice"]) for r in site_rows]), paper_avg)
    add_summary(summary, "site_macro__pooled__foreground_only", mean([float(r["pooled_foreground_dice"]) for r in site_rows]), paper_avg)

    add_summary(summary, "slice_micro__slice_avg__mean_bg_fg", mean([float(r["mean_bg_fg_dice"]) for r in slice_rows]), paper_avg)
    add_summary(summary, "slice_micro__slice_avg__foreground_only", mean([float(r["foreground_dice"]) for r in slice_rows]), paper_avg)
    add_summary(summary, "case_macro__slice_avg__mean_bg_fg", mean([float(r["slice_avg_mean_bg_fg_dice"]) for r in case_rows]), paper_avg)
    add_summary(summary, "case_macro__pooled__mean_bg_fg", mean([float(r["pooled_mean_bg_fg_dice"]) for r in case_rows]), paper_avg)

    global_counts: dict[str, float] = defaultdict(float)
    for row in slice_rows:
        pass
    # Reconstruct global pooled counts from site pooled dice is impossible; use site rows only for macro summaries.
    return summary


def write_markdown(path: Path, checkpoint: Path, summary: list[dict[str, object]], site_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Prostate metric aggregation audit",
        "",
        f"Checkpoint: `{checkpoint}`",
        "",
        "## Aggregation summary",
        "",
        "| metric | value | paper avg | delta |",
        "|---|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['metric']} | {100 * float(row['value']):.2f} | "
            f"{100 * float(row['paper_avg']):.2f} | {100 * float(row['delta_vs_paper']):+.2f} |"
        )
    lines += [
        "",
        "## Per-site comparison",
        "",
        "| site | paper | slice avg bg+fg | pooled bg+fg | slice fg | pooled fg |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in site_rows:
        lines.append(
            f"| {row['site']} | {100 * float(row['paper_dice']):.2f} | "
            f"{100 * float(row['slice_avg_mean_bg_fg_dice']):.2f} | "
            f"{100 * float(row['pooled_mean_bg_fg_dice']):.2f} | "
            f"{100 * float(row['slice_avg_foreground_dice']):.2f} | "
            f"{100 * float(row['pooled_foreground_dice']):.2f} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(r"D:\MLPractice\datasets\processed\prostate\official"))
    parser.add_argument("--manifest", type=Path, default=Path(r"D:\MLPractice\FedCLAM\outputs\prostate\analysis\prostate_official_split_manifest.csv"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(r"D:\MLPractice\FedCLAM\outputs\prostate\analysis\metric_aggregation_audit"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device.startswith("cpu") or torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    manifest = read_manifest(args.manifest, "test")
    site_rows, case_rows, slice_rows = evaluate(model, device, args.data_dir, manifest, args.batch_size)
    summary = build_summary(site_rows, case_rows, slice_rows)

    write_csv(
        args.output_dir / "prostate_metric_site.csv",
        site_rows,
        [
            "site", "slices", "cases",
            "slice_avg_foreground_dice", "slice_avg_background_dice", "slice_avg_mean_bg_fg_dice",
            "pooled_foreground_dice", "pooled_background_dice", "pooled_mean_bg_fg_dice",
            "paper_dice",
        ],
    )
    write_csv(
        args.output_dir / "prostate_metric_case.csv",
        case_rows,
        [
            "site", "case_id", "slices",
            "slice_avg_foreground_dice", "slice_avg_background_dice", "slice_avg_mean_bg_fg_dice",
            "pooled_foreground_dice", "pooled_background_dice", "pooled_mean_bg_fg_dice",
        ],
    )
    write_csv(
        args.output_dir / "prostate_metric_slice.csv",
        slice_rows,
        ["site", "case_id", "split_index", "foreground_dice", "background_dice", "mean_bg_fg_dice", "fg_ratio"],
    )
    write_csv(
        args.output_dir / "prostate_metric_summary.csv",
        summary,
        ["metric", "value", "paper_avg", "delta_vs_paper"],
    )
    write_markdown(args.output_dir / "prostate_metric_aggregation_audit.md", args.checkpoint, summary, site_rows)
    print(args.output_dir)


if __name__ == "__main__":
    main()
