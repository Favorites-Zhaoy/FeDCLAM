from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs"
ANALYSIS_DIR = OUTPUT_ROOT / "analysis"

SITES = {
    "fundus": ["Drishti-GS", "RIM-ONE", "REFUGE_t", "REFUGE_v"],
    "prostate": ["BIDMC", "HK", "I2CVB", "BMC", "RUNMC", "UCL"],
}

CN_NAME = {
    "fundus": "Fundus 眼底数据集",
    "prostate": "Prostate MRI 前列腺数据集",
}

PLOT_NAME = {
    "fundus": "Fundus",
    "prostate": "Prostate MRI",
}

PAPER_FEDCLAM = {
    "fundus": [88.14, 90.30, 91.66, 85.17],
    "prostate": [92.85, 94.61, 94.77, 95.71, 94.27, 90.21],
}


def load_metrics(dataset: str) -> dict:
    path = OUTPUT_ROOT / dataset / "official" / "fedclam_paper_seed0" / "log" / "metrics.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=True).item()


def select_round(metrics: dict, key: str) -> int:
    candidates = [(r, float(v.get(key, np.nan))) for r, v in metrics.items()]
    candidates = [(r, v) for r, v in candidates if np.isfinite(v)]
    return max(candidates, key=lambda item: item[1])[0]


def summarize_round(dataset: str, metrics: dict, round_idx: int) -> dict:
    sites = SITES[dataset]
    values = metrics[round_idx]
    test = [100 * float(values[f"server_test_acc_on_{s}"]) for s in sites]
    val = [100 * float(values[f"server_val_acc_on_{s}"]) for s in sites]
    train_loss = [float(values[f"train_loss_{s}"]) for s in sites]
    train_dice = [100 * float(values[f"train_acc_{s}"]) for s in sites]
    test_loss = [float(values[f"server_test_loss_on_{s}"]) for s in sites]
    vlr = [float(values.get(f"val_loss_ratio_{s}", np.nan)) for s in sites]
    of = [float(values.get(f"of_penalty_{s}", np.nan)) for s in sites]
    return {
        "round": round_idx,
        "test": test,
        "val": val,
        "test_avg": float(np.mean(test)),
        "test_std": float(np.std(test)),
        "val_avg": float(np.mean(val)),
        "train_loss_avg": float(np.mean(train_loss)),
        "train_dice_avg": float(np.mean(train_dice)),
        "test_loss_avg": float(np.mean(test_loss)),
        "vlr_avg": float(np.nanmean(vlr)),
        "of_avg": float(np.nanmean(of)),
    }


def classify(delta: float) -> str:
    ad = abs(delta)
    if ad <= 1:
        return "接近（≤1 个百分点）"
    if ad <= 3:
        return "偏差（1–3 个百分点）"
    return "显著偏差（>3 个百分点）"


def fmt_list(values: list[float]) -> str:
    return " / ".join(f"{v:.2f}" for v in values)


def md_rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def plot_curves(dataset: str, metrics: dict) -> dict[str, Path]:
    sites = SITES[dataset]
    rounds = sorted(metrics)
    paths = {}

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rounds, [100 * metrics[r]["server_test_acc_avg"] for r in rounds], label="Average test Dice")
    ax.plot(rounds, [100 * metrics[r]["server_val_acc_avg"] for r in rounds], label="Average validation Dice")
    ax.set(title=f"{PLOT_NAME[dataset]} FedCLAM average Dice", xlabel="Round", ylabel="Dice (%)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = ANALYSIS_DIR / f"{dataset}_fedclam_avg_dice.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths["avg_dice"] = path

    fig, ax = plt.subplots(figsize=(10, 5))
    for site in sites:
        ax.plot(rounds, [100 * metrics[r][f"server_test_acc_on_{site}"] for r in rounds], label=site, linewidth=1)
    ax.set(title=f"{PLOT_NAME[dataset]} FedCLAM per-client Dice", xlabel="Round", ylabel="Dice (%)")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    path = ANALYSIS_DIR / f"{dataset}_fedclam_site_dice.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths["site_dice"] = path

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rounds, [np.mean([metrics[r][f"train_loss_{s}"] for s in sites]) for r in rounds], label="Average train loss")
    ax.plot(rounds, [np.mean([metrics[r][f"server_test_loss_on_{s}"] for s in sites]) for r in rounds], label="Average test loss")
    ax.set(title=f"{PLOT_NAME[dataset]} FedCLAM loss", xlabel="Round", ylabel="Loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = ANALYSIS_DIR / f"{dataset}_fedclam_loss.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths["loss"] = path

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rounds, [np.nanmean([metrics[r].get(f"val_loss_ratio_{s}", np.nan) for s in sites]) for r in rounds], label="Mean VLR/beta")
    ax.plot(rounds, [np.nanmean([metrics[r].get(f"of_penalty_{s}", np.nan) for s in sites]) for r in rounds], label="Mean OF/tau")
    ax.set(title=f"{PLOT_NAME[dataset]} FedCLAM factors", xlabel="Round", ylabel="Factor")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = ANALYSIS_DIR / f"{dataset}_fedclam_factors.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths["factors"] = path
    return paths


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    env_path = OUTPUT_ROOT / "environment.json"
    validation_path = Path("D:/MLPractice/datasets/processed/validation_report.json")
    env = json.loads(env_path.read_text(encoding="utf-8")) if env_path.exists() else {}
    validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else {}

    summaries = {}
    curve_paths = {}
    rows = []
    for dataset in ["fundus", "prostate"]:
        metrics = load_metrics(dataset)
        summaries[dataset] = {
            "best_val": summarize_round(dataset, metrics, select_round(metrics, "server_val_acc_avg")),
            "best_test": summarize_round(dataset, metrics, select_round(metrics, "server_test_acc_avg")),
            "last": summarize_round(dataset, metrics, max(metrics)),
        }
        curve_paths[dataset] = plot_curves(dataset, metrics)
        paper_avg = float(np.mean(PAPER_FEDCLAM[dataset]))
        for key, label in [("best_val", "best_val"), ("best_test", "best_test_oracle"), ("last", "last")]:
            s = summaries[dataset][key]
            delta = s["test_avg"] - paper_avg
            rows.append(
                {
                    "dataset": dataset,
                    "selection": label,
                    "round": s["round"],
                    "site_test_dice": fmt_list(s["test"]),
                    "avg_test_dice": f"{s['test_avg']:.4f}",
                    "site_std": f"{s['test_std']:.4f}",
                    "paper_avg": f"{paper_avg:.4f}",
                    "delta": f"{delta:+.4f}",
                    "classification": classify(delta),
                }
            )

    csv_path = ANALYSIS_DIR / "fedclam_official_paper_summary.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# FedCLAM 官方主实验复现结果总结",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. 实验范围",
        "",
        "本报告只总结本轮已完成的 FedCLAM 论文主实验，不包含 FedAvg、FedSAM、FedDyn 等 baseline。",
        "",
        "- 数据集：Fundus 眼底分割、Prostate MRI 前列腺分割",
        "- 方法：FedCLAM",
        "- 协议：official 口径，即沿用官方代码的数据使用方式",
        "- 随机种子：seed=0",
        "- 网络与主要训练设置：U-Net + InstanceNorm；Fundus 200 轮，Prostate 500 轮；每轮 1 个本地 epoch",
        "- FedCLAM 参数：α=1，β=1，λFIM=0.01，warm-up=10",
        "",
        "说明：official 口径下，当前官方代码对 Fundus 的 validation/test 使用同一测试集合。因此报告中同时列出 best validation 轮、best test oracle 轮和最后一轮；但应把 best test oracle 视作分析项，而不是严格无泄露选模结果。",
        "",
        "## 2. 运行环境与数据概况",
        "",
    ]
    if env:
        versions = env.get("versions", {})
        lines += [
            f"- Python：{env.get('python', env.get('python_version', '见 environment.json'))}",
            f"- PyTorch：{versions.get('torch', env.get('torch', env.get('pytorch', '见 environment.json')))}",
            f"- CUDA 可用：{env.get('cuda_available', '见 environment.json')}",
            f"- GPU：{env.get('gpu', env.get('gpu_name', '见 environment.json'))}",
        ]
    lines.append(f"- 环境记录：`{md_rel(env_path)}`" if env_path.exists() else "- 环境记录：未找到 `outputs/environment.json`")
    lines.append(f"- 数据验证报告：`{validation_path}`" if validation_path.exists() else "- 数据验证报告：未找到")
    lines.append("")

    if validation:
        lines += ["数据划分数量如下：", "", "| 数据集 | 客户端 | train | valid | test |", "|---|---:|---:|---:|---:|"]
        for dataset in ["fundus", "prostate"]:
            for site, counts in validation.get(dataset, {}).get("official", {}).items():
                lines.append(f"| {dataset} | {site} | {counts.get('train', '-')} | {counts.get('valid', '-')} | {counts.get('test', '-')} |")
        lines.append("")

    lines += [
        "## 3. 论文结果对比总表",
        "",
        "单位均为 Dice (%)。差异分级：≤1 个百分点为“接近”，1–3 为“偏差”，>3 为“显著偏差”。",
        "",
        "| 数据集 | 选择口径 | 轮次 | 复现平均 Dice | 论文平均 Dice | 差值 | 客户端标准差 | 判断 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for dataset in ["fundus", "prostate"]:
        paper_avg = float(np.mean(PAPER_FEDCLAM[dataset]))
        for key, label in [("best_val", "最佳验证轮"), ("best_test", "最佳测试轮（oracle）"), ("last", "最后一轮")]:
            s = summaries[dataset][key]
            delta = s["test_avg"] - paper_avg
            lines.append(f"| {CN_NAME[dataset]} | {label} | {s['round']} | {s['test_avg']:.2f} | {paper_avg:.2f} | {delta:+.2f} | {s['test_std']:.2f} | {classify(delta)} |")
    lines.append("")

    for dataset in ["fundus", "prostate"]:
        sites = SITES[dataset]
        paper = PAPER_FEDCLAM[dataset]
        paper_avg = float(np.mean(paper))
        best = summaries[dataset]["best_val"]
        oracle = summaries[dataset]["best_test"]
        last = summaries[dataset]["last"]
        lines += [
            f"## 4. {CN_NAME[dataset]} 详细结果",
            "",
            f"- 论文 FedCLAM 逐客户端 Dice：{fmt_list(paper)}；平均 {paper_avg:.2f}",
            f"- 最佳验证轮：round {best['round']}，平均 test Dice {best['test_avg']:.2f}，较论文 {best['test_avg'] - paper_avg:+.2f}，{classify(best['test_avg'] - paper_avg)}",
            f"- 最佳测试轮（oracle）：round {oracle['round']}，平均 test Dice {oracle['test_avg']:.2f}，较论文 {oracle['test_avg'] - paper_avg:+.2f}，{classify(oracle['test_avg'] - paper_avg)}",
            f"- 最后一轮：round {last['round']}，平均 test Dice {last['test_avg']:.2f}，较论文 {last['test_avg'] - paper_avg:+.2f}，{classify(last['test_avg'] - paper_avg)}",
            "",
            "### 逐客户端 Dice 对比",
            "",
            "| 客户端 | 论文 FedCLAM | 复现 best-val | 差值 | 复现 oracle | 复现 last |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for i, site in enumerate(sites):
            lines.append(f"| {site} | {paper[i]:.2f} | {best['test'][i]:.2f} | {best['test'][i] - paper[i]:+.2f} | {oracle['test'][i]:.2f} | {last['test'][i]:.2f} |")
        lines += [
            "",
            "### 训练过程摘要",
            "",
            "| 指标 | 最佳验证轮 | 最后一轮 |",
            "|---|---:|---:|",
            f"| 平均 train loss | {best['train_loss_avg']:.4f} | {last['train_loss_avg']:.4f} |",
            f"| 平均 train Dice (%) | {best['train_dice_avg']:.2f} | {last['train_dice_avg']:.2f} |",
            f"| 平均 test loss | {best['test_loss_avg']:.4f} | {last['test_loss_avg']:.4f} |",
            f"| 平均 val Dice (%) | {best['val_avg']:.2f} | {last['val_avg']:.2f} |",
            f"| 平均 VLR/β | {best['vlr_avg']:.4f} | {last['vlr_avg']:.4f} |",
            f"| 平均 OF/τ | {best['of_avg']:.4f} | {last['of_avg']:.4f} |",
            "",
            "### 曲线图",
            "",
            f"- 平均 Dice 曲线：`{md_rel(curve_paths[dataset]['avg_dice'])}`",
            f"- 客户端 Dice 曲线：`{md_rel(curve_paths[dataset]['site_dice'])}`",
            f"- Loss 曲线：`{md_rel(curve_paths[dataset]['loss'])}`",
            f"- FedCLAM 调制因子曲线：`{md_rel(curve_paths[dataset]['factors'])}`",
            "",
        ]

    fundus_delta = summaries["fundus"]["best_val"]["test_avg"] - float(np.mean(PAPER_FEDCLAM["fundus"]))
    prostate_delta = summaries["prostate"]["best_val"]["test_avg"] - float(np.mean(PAPER_FEDCLAM["prostate"]))
    lines += [
        "## 5. 结果解读",
        "",
        f"Fundus 的 best-val 平均 Dice 为 {summaries['fundus']['best_val']['test_avg']:.2f}%，论文平均为 {np.mean(PAPER_FEDCLAM['fundus']):.2f}%，差值 {fundus_delta:+.2f} 个百分点，按预设规则属于 1–3 个百分点的“偏差”范围。逐客户端看，Drishti-GS、REFUGE_t 与论文几乎一致，REFUGE_v 高于论文，RIM-ONE 明显偏低；总体平均的主要差异来自 RIM-ONE。",
        "",
        f"Prostate 的 best-val 平均 Dice 为 {summaries['prostate']['best_val']['test_avg']:.2f}%，论文平均为 {np.mean(PAPER_FEDCLAM['prostate']):.2f}%，差值 {prostate_delta:+.2f} 个百分点，属于显著偏差。该结果在训练早期达到最好值后长期停留在约 49% Dice，说明当前 Prostate official 复现链路虽然能够完整训练和记录指标，但结果没有复现论文表 2 的 90%+ Dice 水平。",
        "",
        "需要强调：这里报告的是“当前代码、当前数据预处理和 official 口径下的可观测结果”。它能确认 Fundus official 口径存在 validation/test 重合风险，也能确认 Prostate 当前结果与论文差异巨大；但不能据此推断作者主观 cherry-pick 或动机。",
        "",
        "## 6. 建议的下一步排查",
        "",
        "1. 优先排查 Prostate 标签语义和 Dice 计算：当前 Prostate Dice 约 0.49，形态上像是二分类前景/背景或标签通道处理存在系统性不一致。",
        "2. 对照官方发布脚本参数再跑一次 Prostate：发布脚本疑似使用 `λFIM=0.1, warm-up=100`，而论文文字为 `λFIM=0.01, warm-up=10`。",
        "3. 对 Prostate 运行单中心/集中式小实验，确认同一预处理数据在 U-Net 上是否能达到合理 Dice；如果集中式也停在 0.49，问题大概率在数据/标签/损失定义。",
        "4. 若要写最终课程/论文复现报告，建议补跑 strict 无泄露 FedCLAM，和 official 结果分开呈现。",
        "",
        "## 7. 输出文件",
        "",
        f"- 汇总 CSV：`{md_rel(csv_path)}`",
        "- Fundus metrics：`outputs/fundus/official/fedclam_paper_seed0/log/metrics.npy`",
        "- Prostate metrics：`outputs/prostate/official/fedclam_paper_seed0/log/metrics.npy`",
        "- 控制台日志：`outputs/console/fedclam_official_only_driver.out.log`",
        "",
    ]

    report_path = ANALYSIS_DIR / "fedclam_official_main_results_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(report_path)
    print(csv_path)


if __name__ == "__main__":
    main()
