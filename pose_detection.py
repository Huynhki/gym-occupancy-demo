"""
Person Detection using YOLO Pose (section V.3)

Wraps Ultralytics YOLOv8/v11-Pose. The weight file is a standard PyTorch
checkpoint; this project keeps a copy with a `.pth` extension
(models/yolov8n-pose.pth) purely so it's easy to hand off as "the model
file" for the demo, but it is loaded the same way Ultralytics loads any
`.pt`/`.pth` YOLO checkpoint.
"""
import shutil
from pathlib import Path

import numpy as np


def _resolve_yolo_weight_path(weights_path):
    """
    Ultralytics (bản mới) chỉ chấp nhận checkpoint có đuôi `.pt`, trong khi
    project này lưu file model với đuôi `.pth` (xem docstring module).
    Nếu weights_path không có đuôi `.pt`, tạo (hoặc tái sử dụng) một bản
    copy cạnh file gốc với đuôi `.pt` rồi trả về đường dẫn đó để nạp vào
    YOLO(), thay vì đổi tên/di chuyển file gốc của người dùng.
    """
    path = Path(weights_path)
    if path.suffix == ".pt":
        return str(path)

    pt_path = path.with_suffix(".pt")
    if not pt_path.exists():
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file model: {weights_path}")
        shutil.copyfile(path, pt_path)
    return str(pt_path)


class PoseDetector:
    def __init__(self, weights_path="models/yolov8n-pose.pth", device="cpu", conf=0.35):
        from ultralytics import YOLO
        resolved_path = _resolve_yolo_weight_path(weights_path)
        self.model = YOLO(resolved_path)
        self.device = device
        self.conf = conf

    def detect(self, frame_bgr):
        """
        Returns a list of detected persons in one frame:
        [{"bbox": [x0,y0,x1,y1], "bbox_conf": float,
          "keypoints": [(x,y), ...17], "keypoint_conf": [float, ...17]}, ...]
        """
        results = self.model.predict(
            frame_bgr, conf=self.conf, device=self.device, classes=[0], verbose=False
        )
        people = []
        if not results:
            return people
        r = results[0]
        if r.keypoints is None or r.boxes is None:
            return people

        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        boxes_conf = r.boxes.conf.cpu().numpy()
        kpts_xy = r.keypoints.xy.cpu().numpy()       # [N, 17, 2]
        kpts_conf = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None \
            else np.ones(kpts_xy.shape[:2])

        for i in range(len(boxes_xyxy)):
            people.append({
                "bbox": boxes_xyxy[i].tolist(),
                "bbox_conf": float(boxes_conf[i]),
                "keypoints": [tuple(p) for p in kpts_xy[i].tolist()],
                "keypoint_conf": kpts_conf[i].tolist(),
            })
        return people