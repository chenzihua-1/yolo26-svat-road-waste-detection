

import sys
import os
import warnings
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

_root = Path(__file__).resolve().parent
_root_str = str(_root)
if _root_str in sys.path:
    sys.path.remove(_root_str)
    sys.path.append(_root_str)

from ultralytics import YOLO

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    model_yaml = Path("ultralytics/cfg/models/26/yolo26-p2.yaml")
    model = YOLO(model=str(model_yaml))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备：{device}")

    model.train(
        data=r"data.yaml",
        imgsz=512,
        epochs=300,
        batch=4,
        workers=4,
        device=device,
        optimizer="SGD",
        deterministic=False,
        close_mosaic=10,
        resume=False,
        project="runs/train",
        name="exp",
        single_cls=False,
        cache=True,
    )
