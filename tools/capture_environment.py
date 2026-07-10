from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

import cv2
import matplotlib
import monai
import numpy
import PIL
import SimpleITK
import tensorboard
import torch
import torchvision


def command(*args: str) -> str:
    return subprocess.run(args, check=False, capture_output=True, text=True).stdout.strip()


payload = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "git_commit": command("git", "rev-parse", "HEAD"),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "driver": command("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
    "cuda_available": torch.cuda.is_available(),
    "versions": {
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "torch_cuda": torch.version.cuda, "numpy": numpy.__version__,
        "opencv": cv2.__version__, "SimpleITK": SimpleITK.Version_VersionString(),
        "monai": monai.__version__, "tensorboard": tensorboard.__version__,
        "matplotlib": matplotlib.__version__, "Pillow": PIL.__version__,
    },
    "determinism": {"seed": 0, "cudnn_benchmark": False, "cudnn_deterministic": True},
}
destination = Path(__file__).resolve().parents[1] / "outputs" / "environment.json"
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(destination)
