from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SITES = {
    "fundus": ["Drishti-GS", "RIM-ONE", "REFUGE_t", "REFUGE_v"],
    "prostate": ["BIDMC", "HK", "I2CVB", "BMC", "RUNMC", "UCL"],
}
DISPLAY = {
    "fedavg": "FedAvg", "fedavgm": "FedAvgM", "fedevi": "FedEvi",
    "fedfa": "FedFA", "fedprox": "FedProx", "fedsam": "FedSAM",
    "feddyn": "FedDyn", "fedharmo": "HarmoFL", "fedclam": "FedCLAM",
    "fedclam_released": "FedCLAM (released script)",
}
PAPER = {
    "fundus": {
        "fedavg": [74.53, 90.83, 90.80, 82.71], "fedavgm": [81.58, 90.52, 90.55, 80.81],
        "fedevi": [85.48, 91.07, 92.20, 82.97], "fedfa": [79.56, 91.77, 91.69, 82.10],
        "fedprox": [84.24, 91.62, 92.21, 82.71], "fedsam": [81.44, 91.42, 90.69, 80.64],
        "feddyn": [85.78, 90.28, 90.87, 83.05], "fedharmo": [80.55, 91.52, 91.13, 80.82],
        "fedclam": [88.14, 90.30, 91.66, 85.17],
    },
    "prostate": {
        "fedavg": [91.46, 93.06, 92.61, 95.40, 93.84, 90.24],
        "fedavgm": [89.08, 93.46, 94.31, 96.00, 94.18, 89.44],
        "fedevi": [92.02, 93.36, 93.45, 95.71, 93.60, 90.66],
        "fedfa": [91.42, 93.29, 93.26, 95.28, 93.87, 90.74],
        "fedprox": [91.35, 93.57, 92.07, 95.36, 93.70, 90.20],
        "fedsam": [92.44, 93.23, 93.02, 95.57, 94.30, 90.41],
        "feddyn": [91.90, 93.44, 92.98, 95.69, 93.82, 89.97],
        "fedharmo": [92.61, 93.87, 94.60, 95.96, 94.23, 90.40],
        "fedclam": [92.85, 94.61, 94.77, 95.71, 94.27, 90.21],
    },
}


def run_identity(metrics_path: Path, output_root: Path):
    relative = metrics_path.relative_to(output_root)
    dataset, protocol, experiment = relative.parts[:3]
    method = experiment.split("_paper_seed0")[0].split("_smoke_seed0")[0]
    return dataset, protocol, method


def select_round(metrics, criterion: str):
    rounds = sorted(metrics)
    if criterion == "last": return rounds[-1]
    key = "server_val_acc_avg" if criterion == "best_val" else "server_test_acc_avg"
    candidates = [(r, metrics[r].get(key, np.nan)) for r in rounds]
    candidates = [(r, v) for r, v in candidates if np.isfinite(v)]
    return max(candidates, key=lambda item: item[1])[0]


def classification(delta: float) -> str:
    delta = abs(delta)
    if delta <= 1: return "接近"
    if delta <= 3: return "偏差"
    return "显著偏差"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    analysis_dir = args.output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    rows, curves = [], {}
    for path in args.output_root.glob("*/*/*/log/metrics.npy"):
        dataset, protocol, method = run_identity(path, args.output_root)
        metrics = np.load(path, allow_pickle=True).item()
        curves[(dataset, protocol, method)] = metrics
        criteria = ("last", "best_val", "oracle") if protocol == "official" else ()
        for criterion in criteria:
            round_idx = select_round(metrics, criterion)
            values = [100 * metrics[round_idx].get(f"server_test_acc_on_{site}", np.nan)
                      for site in SITES[dataset]]
            finite = np.asarray([v for v in values if np.isfinite(v)])
            paper_values = PAPER.get(dataset, {}).get(method.replace("_released", ""))
            paper_avg = float(np.mean(paper_values)) if paper_values else np.nan
            avg = float(np.mean(finite)) if finite.size else np.nan
            row = {"dataset": dataset, "protocol": protocol, "method": method,
                   "selection": criterion, "round": round_idx}
            row.update({site: value for site, value in zip(SITES[dataset], values)})
            row.update({"average": avg, "site_std": float(np.std(finite)) if finite.size else np.nan,
                        "paper_average": paper_avg, "delta": avg - paper_avg,
                        "classification": classification(avg - paper_avg) if np.isfinite(paper_avg) else "N/A"})
            rows.append(row)
    for path in args.output_root.glob("*/*/*/log/strict_final_test.json"):
        dataset, protocol, method = run_identity(path.with_name("metrics.npy"), args.output_root)
        result = json.loads(path.read_text(encoding="utf-8"))
        values = [100 * result[site]["dice"] for site in SITES[dataset]]
        row = {"dataset": dataset, "protocol": protocol, "method": method,
               "selection": "best_val", "round": result["summary"]["selected_round"]}
        row.update({site: value for site, value in zip(SITES[dataset], values)})
        row.update({"average": float(np.mean(values)), "site_std": float(np.std(values)),
                    "paper_average": np.nan, "delta": np.nan, "classification": "严格口径"})
        rows.append(row)

    all_sites = SITES["fundus"] + SITES["prostate"]
    fieldnames = ["dataset", "protocol", "method", "selection", "round"] + all_sites + [
        "average", "site_std", "paper_average", "delta", "classification"]
    with (analysis_dir / "reproduction_table.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)

    for dataset in SITES:
        fig, ax = plt.subplots(figsize=(10, 6))
        for (data_name, protocol, method), metrics in curves.items():
            if data_name != dataset or protocol != "official" or "smoke" in method: continue
            rounds = sorted(metrics)
            values = [100 * metrics[r].get("server_test_acc_avg", np.nan) for r in rounds]
            ax.plot(rounds, values, label=DISPLAY.get(method, method), linewidth=1)
        ax.set(xlabel="Round", ylabel="Average test Dice (%)", title=f"{dataset.title()} convergence")
        ax.grid(alpha=.2); ax.legend(ncol=2, fontsize=8); fig.tight_layout()
        fig.savefig(analysis_dir / f"{dataset}_test_dice.png", dpi=180); plt.close(fig)

    lines = ["# FedCLAM 复现报告", "", "## 实验口径", "",
             "官方口径忠实保留发布代码的逐轮测试行为；严格口径隔离验证集与测试集，并只在最佳验证检查点上测试一次。",
             "代码可以确认 Fundus 的官方验证/测试集合重合，但现有证据不能证明作者主观进行了 cherry-pick。", "",
             "## 论文结果对比", ""]
    for dataset in SITES:
        lines += [f"### {dataset.title()}", "", "| 方法 | 选择口径 | 复现平均 Dice | 论文平均 Dice | 差值 | 判断 |", "|---|---:|---:|---:|---:|---|"]
        for row in rows:
            if row["dataset"] != dataset: continue
            paper = "-" if not np.isfinite(row["paper_average"]) else f'{row["paper_average"]:.2f}'
            delta = "-" if not np.isfinite(row["delta"]) else f'{row["delta"]:+.2f}'
            lines.append(f'| {DISPLAY.get(row["method"], row["method"])} | {row["protocol"]}/{row["selection"]} | {row["average"]:.2f} | {paper} | {delta} | {row["classification"]} |')
        lines.append("")
    lines += ["## 已知复现限制", "",
              "- FedCLAM 仓库未发布原始 Fundus 划分清单，本复现固定 seed=0 并保存完整清单。",
              "- 论文写明 Prostate λFIM=0.01，而发布脚本使用 0.1 且 warm-up=100；两种配置分别报告。",
              "- 本机运行环境和依赖指纹见 `outputs/environment.json`。", ""]
    (analysis_dir / "reproduction_report_zh.md").write_text("\n".join(lines), encoding="utf-8")
    print(analysis_dir)


if __name__ == "__main__":
    main()
