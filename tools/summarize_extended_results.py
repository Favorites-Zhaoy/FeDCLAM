"""Summarize extended FedCLAM generalization runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def float_or_nan(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def summarize_metrics(path: Path) -> list[dict[str, object]]:
    rows = read_csv(path)
    by_split = {}
    for row in rows:
        split = row.get("split", "")
        dice = float_or_nan(row.get("dice", ""))
        if not np.isfinite(dice):
            continue
        key = (split, int(float(row.get("round", 0))))
        by_split.setdefault(key, []).append(dice)
    summaries = []
    for (split, round_idx), values in sorted(by_split.items()):
        values = np.asarray(values, dtype=float)
        summaries.append({
            "run": str(path.parent.parent),
            "split": split,
            "round": round_idx,
            "avg_dice": float(np.mean(values)),
            "site_std": float(np.std(values)),
            "worst_site_dice": float(np.min(values)),
            "site_count": int(values.size),
        })
    return summaries


def best_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    best = {}
    for row in rows:
        key = (row["run"], row["split"])
        if key not in best or row["avg_dice"] > best[key]["avg_dice"]:
            best[key] = row
    return sorted(best.values(), key=lambda item: (item["run"], item["split"]))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["run", "split", "round", "avg_dice", "site_std", "worst_site_dice", "site_count"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "| Run | Split | Round | Avg Dice | Site Std | Worst Site Dice | Sites |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run']} | {row['split']} | {row['round']} | "
            f"{row['avg_dice']:.4f} | {row['site_std']:.4f} | "
            f"{row['worst_site_dice']:.4f} | {row['site_count']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_strict_final(path: Path) -> list[dict[str, object]]:
    rows = []
    for final_path in path.rglob("strict_final_test.json"):
        data = json.loads(final_path.read_text(encoding="utf-8"))
        summary = data.get("summary", {})
        if summary:
            rows.append({
                "run": str(final_path.parent.parent),
                "split": "strict_final_test",
                "round": summary.get("selected_round", ""),
                "avg_dice": summary.get("avg_dice", float("nan")),
                "site_std": summary.get("site_std", float("nan")),
                "worst_site_dice": min(
                    item["dice"] for key, item in data.items()
                    if key != "summary" and isinstance(item, dict) and "dice" in item
                ),
                "site_count": len([key for key in data.keys() if key != "summary"]),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary-dir", type=Path, default=Path("outputs/analysis"))
    args = parser.parse_args()

    rows = []
    for metrics_path in args.output_root.rglob("metrics_long.csv"):
        rows.extend(summarize_metrics(metrics_path))
    rows.extend(collect_strict_final(args.output_root))
    rows = best_rows(rows)
    write_csv(args.summary_dir / "extended_summary.csv", rows)
    write_markdown(args.summary_dir / "extended_summary.md", rows)
    print(f"Wrote {len(rows)} summary rows to {args.summary_dir}")


if __name__ == "__main__":
    main()
