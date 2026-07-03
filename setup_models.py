"""
Run this ONCE on a machine with normal internet access (this sandbox's
network is restricted, so it cannot reach Ultralytics' weight servers).

It downloads the pretrained YOLOv8-Pose checkpoint (COCO, 17 keypoints) used
for Person Detection (section V.3), and saves a copy at
models/yolov8n-pose.pth so the Gradio demo has a model file to load.

Usage:
    python setup_models.py
"""
import shutil
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def main():
    from ultralytics import YOLO

    print("Downloading yolov8n-pose.pt (pretrained on COCO, 17 keypoints)...")
    model = YOLO("yolov8n-pose.pt")  # auto-downloads to CWD or ultralytics cache
    src = Path("yolov8n-pose.pt")
    if not src.exists():
        # ultralytics may have cached it elsewhere; locate via model.ckpt_path
        src = Path(model.ckpt_path)

    dst = MODELS_DIR / "yolov8n-pose.pth"
    shutil.copy(src, dst)
    print(f"Saved weights to: {dst.resolve()}")
    print("Done. You can now run `python app.py` or `python pipeline.py ...`")


if __name__ == "__main__":
    main()
