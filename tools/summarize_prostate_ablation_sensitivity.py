"""Summarize Prostate component ablation and K/lambda_FIM sensitivity runs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PAPER_ABLATION = {
    "None": 92.77,
    "FIM": 93.07,
    "CLAM": 93.50,
    "FIM + CLAM": 93.74,
}


@dataclass(frozen=True)
class RunSpec:
    group: str
    label: str
    path: Path
    paper: float | None = None
    note: str = ""


def load_summary(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "status": "missing",
            "round_count": 0,
            "best_val_round": "",
            "best_val": "",
            "best_val_test": "",
            "best_test_round": "",
            "best_test": "",
            "last_round": "",
            "last_test": "",
        }

    metrics = np.load(path, allow_pickle=True).item()
    rows: list[tuple[int, float, float]] = []
    for round_id, values in metrics.items():
        if not isinstance(values, dict):
            continue
        if "server_test_acc_avg" not in values:
            continue
        rows.append(
            (
                int(round_id),
                float(values.get("server_val_acc_avg", np.nan)) * 100.0,
                float(values.get("server_test_acc_avg", np.nan)) * 100.0,
            )
        )

    if not rows:
        return {
            "status": "empty",
            "round_count": 0,
            "best_val_round": "",
            "best_val": "",
            "best_val_test": "",
            "best_test_round": "",
            "best_test": "",
            "last_round": "",
            "last_test": "",
        }

    best_val = max(rows, key=lambda item: item[1])
    best_test = max(rows, key=lambda item: item[2])
    last = max(rows, key=lambda item: item[0])
    return {
        "status": "complete",
        "round_count": len(rows),
        "best_val_round": best_val[0],
        "best_val": best_val[1],
        "best_val_test": best_val[2],
        "best_test_round": best_test[0],
        "best_test": best_test[2],
        "last_round": last[0],
        "last_test": last[2],
    }


def fmt(value: object) -> str:
    if value == "" or value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def build_specs(output_root: Path) -> list[RunSpec]:
    base = output_root / "prostate" / "official"
    clean = base / "fedclam_paperparams_clean2_seed0" / "log" / "metrics.npy"

    specs = [
        RunSpec("ablation", "None", base / "ablation" / "ablation_none_paper_seed0" / "log" / "metrics.npy", PAPER_ABLATION["None"]),
        RunSpec("ablation", "FIM", base / "ablation" / "ablation_fim_paper_seed0" / "log" / "metrics.npy", PAPER_ABLATION["FIM"]),
        RunSpec("ablation", "CLAM", base / "ablation" / "ablation_clam_paper_seed0" / "log" / "metrics.npy", PAPER_ABLATION["CLAM"]),
        RunSpec("ablation", "FIM + CLAM", clean, PAPER_ABLATION["FIM + CLAM"], "reuses paperparams clean2 base"),
    ]

    for k in ["1", "2", "5", "10", "20"]:
        path = clean if k == "1" else base / "sensitivity" / "prostate_paper_base" / "k" / f"sensitivity_k_{k}_paper_seed0" / "log" / "metrics.npy"
        note = "reuses paperparams clean2 base" if k == "1" else "Prostate-only extension of paper Fundus grid"
        specs.append(RunSpec("sensitivity_k", f"k={k}", path, None, note))

    for lambda_fim in ["1e-4", "1e-3", "1e-2", "1e-1", "1"]:
        path = clean if lambda_fim == "1e-2" else base / "sensitivity" / "prostate_paper_base" / "fim" / f"sensitivity_lfim_{lambda_fim}_paper_seed0" / "log" / "metrics.npy"
        note = "reuses paperparams clean2 base" if lambda_fim == "1e-2" else "Prostate-only extension of paper Fundus grid"
        specs.append(RunSpec("sensitivity_lfim", f"lambda_FIM={lambda_fim}", path, None, note))

    return specs


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def table(group: str, title: str) -> list[str]:
        selected = [row for row in rows if row["group"] == group]
        lines = [
            f"## {title}",
            "",
            "| Setting | Status | Paper | Best-val test | Best-test | Last test | Delta vs paper | Note |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for row in selected:
            lines.append(
                "| {label} | {status} | {paper} | {best_val_test} | {best_test} | {last_test} | {delta_vs_paper} | {note} |".format(
                    **{key: fmt(value) for key, value in row.items()}
                )
            )
        return lines + [""]

    lines = [
        "# Prostate Ablation And Sensitivity Summary",
        "",
        "This report uses `fedclam_paperparams_clean2_seed0` as the shared paper-parameter base for `FIM + CLAM`, `k=1`, and `lambda_FIM=1e-2`.",
        "The K and lambda_FIM grids follow the paper's Fundus sensitivity grid, but are run on Prostate as an additional reproduction diagnostic.",
        "",
    ]
    lines.extend(table("ablation", "Component Ablation"))
    lines.extend(table("sensitivity_k", "K Sensitivity"))
    lines.extend(table("sensitivity_lfim", "Lambda_FIM Sensitivity"))
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--analysis-dir", type=Path, default=None)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    analysis_dir = args.analysis_dir or output_root / "prostate" / "analysis"

    rows: list[dict[str, object]] = []
    for spec in build_specs(output_root):
        summary = load_summary(spec.path)
        paper = spec.paper
        best_val_test = summary["best_val_test"]
        delta = ""
        if paper is not None and isinstance(best_val_test, float):
            delta = best_val_test - paper
        rows.append(
            {
                "group": spec.group,
                "label": spec.label,
                "status": summary["status"],
                "round_count": summary["round_count"],
                "paper": paper if paper is not None else "",
                "best_val_round": summary["best_val_round"],
                "best_val": summary["best_val"],
                "best_val_test": summary["best_val_test"],
                "best_test_round": summary["best_test_round"],
                "best_test": summary["best_test"],
                "last_round": summary["last_round"],
                "last_test": summary["last_test"],
                "delta_vs_paper": delta,
                "note": spec.note,
                "metrics": str(spec.path),
            }
        )

    csv_path = analysis_dir / "prostate_ablation_sensitivity_summary.csv"
    md_path = analysis_dir / "prostate_ablation_sensitivity_summary.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
