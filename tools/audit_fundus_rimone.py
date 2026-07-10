import argparse
import csv
import hashlib
import sys
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyfed.network.unet import UNet


SITES = ["Drishti-GS", "RIM-ONE", "REFUGE_t", "REFUGE_v"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sample_paths(paths: list[Path], n: int, seed: int) -> list[Path]:
    if len(paths) <= n:
        return paths
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(paths), size=n, replace=False).tolist())
    return [paths[i] for i in idx]


def mask_stats(mask: np.ndarray) -> dict[str, Union[float, int, str]]:
    disc = mask > 0
    cup = mask > 1
    values = ",".join(str(int(v)) for v in sorted(np.unique(mask).tolist()))
    cup_outside_disc = int(np.logical_and(cup, ~disc).sum())
    return {
        "unique_values": values,
        "disc_area": int(disc.sum()),
        "cup_area": int(cup.sum()),
        "disc_ratio": float(disc.mean()),
        "cup_ratio": float(cup.mean()),
        "cup_to_disc": float(cup.sum() / max(disc.sum(), 1)),
        "cup_outside_disc": cup_outside_disc,
    }


def write_split_csv(data_dir: Path, output_dir: Path) -> None:
    rows = []
    for split in ["train", "valid", "test"]:
        image_dir = data_dir / "RIM-ONE" / split / "image"
        if not image_dir.exists():
            continue
        for image_path in sorted(image_dir.glob("*.npy")):
            mask_path = Path(str(image_path).replace("\\image\\", "\\mask\\").replace("/image/", "/mask/"))
            image = np.load(image_path)
            mask = np.load(mask_path)
            stats = mask_stats(mask)
            rows.append({
                "site": "RIM-ONE",
                "split": split,
                "filename": image_path.name,
                "image_shape": "x".join(map(str, image.shape)),
                "mask_shape": "x".join(map(str, mask.shape)),
                "image_sha256": sha256(image_path),
                "mask_sha256": sha256(mask_path),
                **stats,
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "rimone_split_current.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_all_mask_stats(data_dir: Path, output_dir: Path) -> None:
    rows = []
    for site in SITES:
        for split in ["train", "valid", "test"]:
            mask_dir = data_dir / site / split / "mask"
            if not mask_dir.exists():
                continue
            for mask_path in sorted(mask_dir.glob("*.npy")):
                mask = np.load(mask_path)
                rows.append({
                    "site": site,
                    "split": split,
                    "filename": mask_path.name,
                    **mask_stats(mask),
                })
    with (output_dir / "fundus_mask_stats.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def overlay_one(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    img = image.copy()
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    disc = (mask > 0).astype(np.uint8)
    cup = (mask > 1).astype(np.uint8)
    disc_contours, _ = cv2.findContours(disc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cup_contours, _ = cv2.findContours(cup, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = img.copy()
    color = img.copy()
    color[disc.astype(bool)] = (0.65 * color[disc.astype(bool)] + 0.35 * np.array([0, 255, 0])).astype(np.uint8)
    color[cup.astype(bool)] = (0.55 * color[cup.astype(bool)] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)
    out = cv2.addWeighted(out, 0.45, color, 0.55, 0)
    cv2.drawContours(out, disc_contours, -1, (0, 255, 0), 2)
    cv2.drawContours(out, cup_contours, -1, (255, 0, 0), 2)
    return out


def write_overlays(data_dir: Path, output_dir: Path, n: int, seed: int) -> None:
    for split in ["train", "test"]:
        image_dir = data_dir / "RIM-ONE" / split / "image"
        if not image_dir.exists():
            continue
        target = output_dir / f"overlay_{split}"
        target.mkdir(parents=True, exist_ok=True)
        for image_path in sample_paths(sorted(image_dir.glob("*.npy")), n, seed):
            mask_path = Path(str(image_path).replace("\\image\\", "\\mask\\").replace("/image/", "/mask/"))
            image = np.load(image_path)
            mask = np.load(mask_path)
            overlay = overlay_one(image, mask)
            cv2.imwrite(str(target / f"{image_path.stem}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def dice_per_channel(logits: torch.Tensor, label: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    pred = (torch.sigmoid(logits) > 0.5).float()
    label = label.float()
    batch = label.shape[0]
    pred_flat = pred.view(batch, pred.shape[1], -1)
    label_flat = label.view(batch, label.shape[1], -1)
    inter = (pred_flat * label_flat).sum(dim=2)
    union = pred_flat.sum(dim=2) + label_flat.sum(dim=2)
    dice = (2 * inter / (union + 1e-5)).detach().cpu().numpy()
    return dice[:, 0], dice[:, 1]


def load_fundus_model(ckpt_path: Path, device: torch.device) -> UNet:
    model = UNet(norm="in")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["server"] if isinstance(ckpt, dict) and "server" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def evaluate_checkpoint(data_dir: Path, ckpt_path: Path, run_name: str, batch_size: int, device: torch.device) -> list[dict[str, object]]:
    model = load_fundus_model(ckpt_path, device)
    rows = []
    with torch.no_grad():
        for site in SITES:
            image_paths = sorted((data_dir / site / "test" / "image").glob("*.npy"))
            disc_scores: list[float] = []
            cup_scores: list[float] = []
            for start in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[start:start + batch_size]
                images = []
                labels = []
                for image_path in batch_paths:
                    mask_path = Path(str(image_path).replace("\\image\\", "\\mask\\").replace("/image/", "/mask/"))
                    image = np.load(image_path).transpose(2, 0, 1).astype(np.float32) / 255.0
                    mask = np.load(mask_path)
                    label = np.stack([(mask > 0).astype(np.float32), (mask > 1).astype(np.float32)], axis=0)
                    images.append(image)
                    labels.append(label)
                x = torch.from_numpy(np.stack(images)).to(device)
                y = torch.from_numpy(np.stack(labels)).to(device)
                logits = model(x)
                disc, cup = dice_per_channel(logits, y)
                disc_scores.extend(disc.tolist())
                cup_scores.extend(cup.tolist())
            rows.append({
                "run": run_name,
                "site": site,
                "n_test": len(image_paths),
                "disc_dice": float(np.mean(disc_scores) * 100),
                "cup_dice": float(np.mean(cup_scores) * 100),
                "avg_dice": float((np.mean(disc_scores) + np.mean(cup_scores)) / 2 * 100),
            })
    return rows


def write_summary(output_dir: Path, eval_rows: list[dict[str, object]]) -> None:
    stats = []
    stats_csv = output_dir / "fundus_mask_stats.csv"
    if stats_csv.exists():
        with stats_csv.open("r", encoding="utf-8") as f:
            stats = list(csv.DictReader(f))

    lines = [
        "# RIM-ONE 数据审计报告",
        "",
        "## 已生成文件",
        "",
        "- `rimone_split_current.csv`：当前 RIM-ONE train/test 清单与 SHA256",
        "- `fundus_mask_stats.csv`：四个 Fundus 客户端 mask 面积与标签统计",
        "- `overlay_train/`、`overlay_test/`：RIM-ONE 抽样 overlay 可视化",
        "- `fundus_channel_dice.csv`：已有 checkpoint 的 disc/cup 分解 Dice",
        "",
        "## RIM-ONE 当前划分",
        "",
    ]
    if stats:
        rim = [r for r in stats if r["site"] == "RIM-ONE"]
        for split in ["train", "valid", "test"]:
            subset = [r for r in rim if r["split"] == split]
            if not subset:
                continue
            disc = np.mean([float(r["disc_ratio"]) for r in subset])
            cup = np.mean([float(r["cup_ratio"]) for r in subset])
            lines.append(f"- {split}: {len(subset)} 张，mean disc ratio={disc:.4f}，mean cup ratio={cup:.4f}")
    if eval_rows:
        lines += ["", "## 分通道 Dice 诊断", ""]
        lines.append("| run | site | disc Dice | cup Dice | avg Dice |")
        lines.append("|---|---|---:|---:|---:|")
        for r in eval_rows:
            lines.append(f"| {r['run']} | {r['site']} | {r['disc_dice']:.2f} | {r['cup_dice']:.2f} | {r['avg_dice']:.2f} |")
    lines += [
        "",
        "## 初步解读建议",
        "",
        "- 若 RIM-ONE 的 cup Dice 明显低于 disc Dice，优先检查 cup mask 来源、编码和边界。",
        "- 若 RIM-ONE 的 disc/cup 都低，优先检查图像-mask 配对、test split 难度或 mask 来源版本。",
        "- overlay PNG 应人工快速浏览，重点看 cup 是否在 disc 内、边界是否偏移、mask 是否反色或错配。",
    ]
    (output_dir / "rimone_audit.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(r"D:\MLPractice\datasets\processed\fundus\official"))
    parser.add_argument("--output-dir", type=Path, default=Path(r"D:\MLPractice\FedCLAM\outputs\analysis\rimone_data_audit"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path(r"D:\MLPractice\FedCLAM\outputs\paper_extra\fundus\official"))
    parser.add_argument("--overlay-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_split_csv(args.data_dir, args.output_dir)
    write_all_mask_stats(args.data_dir, args.output_dir)
    write_overlays(args.data_dir, args.output_dir, args.overlay_samples, args.seed)

    eval_rows: list[dict[str, object]] = []
    if not args.skip_eval:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for run_dir in sorted(args.checkpoint_root.glob("ablation_*_paper_seed0")):
            ckpt = run_dir / "ckpt" / "model_best.pth"
            if ckpt.exists():
                eval_rows.extend(evaluate_checkpoint(args.data_dir, ckpt, run_dir.name.replace("_paper_seed0", ""), args.batch_size, device))
        if eval_rows:
            with (args.output_dir / "fundus_channel_dice.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(eval_rows[0].keys()))
                writer.writeheader()
                writer.writerows(eval_rows)
    write_summary(args.output_dir, eval_rows)
    print(f"Wrote audit to {args.output_dir}")


if __name__ == "__main__":
    main()
