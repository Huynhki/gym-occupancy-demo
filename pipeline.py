"""
Full pipeline orchestrator, mirroring the decomposition tree (section IV):

Frame Extraction -> Person Detection (YOLO Pose) -> Spatial Interaction Score
-> Occupancy State Machine -> Time Accumulation -> Output Report (JSON)

4 inputs (matches the demo UI):
  --video        gym camera footage (.mp4)
  --ref_image    empty gym reference photo, RED DOTS marked at each
                 equipment's center, same camera angle as the video
  --devices      JSON file: [{id, type}, ...]   (no coordinates)
  --ground_truth JSON file: [{id, type, occupied_time_seconds}, ...] (optional,
                 only used to compute accuracy/MAE in the report, not by the
                 detection algorithm itself)

Usage:
    python pipeline.py --video video.mp4 --ref_image room_marked.jpg \
        --devices devices.json --ground_truth groundtruth.json \
        --weights models/yolov8n-pose.pth --out output.json
"""
import argparse
import json
import cv2
import numpy as np

from roi import build_roi, load_devices, load_ground_truth, match_dots_to_devices, save_roi_debug_image
from pose_detection import PoseDetector
from spatial_interaction import interaction_score
from state_machine import EquipmentStateMachine, accumulate_time


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """
    Đọc ảnh bằng OpenCV nhưng hỗ trợ đường dẫn có ký tự Unicode (tiếng Việt)
    hoặc dấu cách trên Windows.

    cv2.imread() trên Windows dùng API nội bộ không hỗ trợ tốt đường dẫn
    Unicode, nên với các path như "C:/Users/Ánh máy chấm.png" nó sẽ âm thầm
    trả về None thay vì báo lỗi rõ ràng.

    Giải pháp: đọc file thành mảng byte bằng np.fromfile (hỗ trợ Unicode
    path đầy đủ vì dùng Python file I/O), sau đó giải mã ảnh bằng
    cv2.imdecode thay vì để OpenCV tự mở file.
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except FileNotFoundError:
        return None
    if data.size == 0:
        return None
    img = cv2.imdecode(data, flags)
    return img


def imwrite_unicode(path, img, ext=".png"):
    """
    Ghi ảnh ra file hỗ trợ đường dẫn Unicode/dấu cách trên Windows,
    dùng cv2.imencode + tofile thay vì cv2.imwrite trực tiếp.
    """
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(path)
    return True


def extract_frames(video_path, sample_fps=1.0):
    """Frame Extraction (V.1): yields (frame_idx, frame_bgr) at sample_fps."""
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / sample_fps))
    idx = 0
    sampled_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            yield sampled_idx, frame
            sampled_idx += 1
        idx += 1
    cap.release()


def run_pipeline(
    video_path,
    ref_image_path,
    devices_json_path,
    ground_truth_json_path=None,
    weights_path="models/yolov8n-pose.pth",
    sam2_checkpoint=None,
    sam2_config=None,
    sample_fps=1.0,
    t_in=3,
    t_out=5,
    score_threshold=0.5,
    conf_thresh=0.5,
    progress_cb=None,
):
    # ---- Setup Inputs (II.1) ----
    devices = load_devices(devices_json_path)                       # (id, type)
    ground_truth = load_ground_truth(ground_truth_json_path) if ground_truth_json_path else {}

    # ---- ROI Configuration (IV.3, V.2): auto-detect red dots on the
    # reference image and match them, in order, to `devices` ----
    ref_img = imread_unicode(ref_image_path)
    if ref_img is None:
        raise ValueError(f"Không đọc được ảnh tham chiếu: {ref_image_path}")
    points = match_dots_to_devices(ref_img, devices)
    roi_config = build_roi(ref_img, points, sam2_checkpoint, sam2_config)

    # Debug: luôn xuất ảnh roi_debug.png để kiểm tra trực quan ROI có khớp
    # đúng vị trí + đúng thiết bị thật hay không (xem roi.save_roi_debug_image).
    import os
    debug_dir = "outputs"
    os.makedirs(debug_dir, exist_ok=True)
    debug_path = os.path.join(debug_dir, "roi_debug.png")
    try:
        save_roi_debug_image(ref_img, roi_config, debug_path)
        print(f"[debug] Đã lưu ảnh kiểm tra ROI tại: {debug_path}")
    except Exception as e:
        print(f"[debug] Không lưu được ảnh debug ROI: {e}")

    machines = {
        eid: EquipmentStateMachine(eid, roi["type"], t_in, t_out, score_threshold)
        for eid, roi in roi_config.items()
    }

    # ---- Person Detection + Spatial Interaction + State Machine ----
    detector = PoseDetector(weights_path=weights_path)
    total_frames = 0
    for frame_idx, frame in extract_frames(video_path, sample_fps):
        total_frames = frame_idx
        people = detector.detect(frame)

        for eid, roi in roi_config.items():
            best_score = 0.0
            for person in people:
                s, _ = interaction_score(
                    person["bbox"], person["keypoints"], person["keypoint_conf"],
                    roi, conf_thresh
                )
                best_score = max(best_score, s)
            machines[eid].step(frame_idx, best_score)

        if progress_cb:
            progress_cb(frame_idx)

    # ---- Time Accumulation + Output Report (V.6-V.7), + accuracy vs
    # ground truth if provided ----
    report = []
    for eid, sm in machines.items():
        sm.finalize(total_frames)
        seconds = round(accumulate_time(sm, sample_fps), 1)
        gt = ground_truth.get(eid)
        entry = {
            "id": eid,
            "type": roi_config[eid]["type"],
            "occupied_time_seconds": seconds,
            "ground_truth_seconds": gt,
        }
        if gt is not None:
            entry["abs_error_seconds"] = round(abs(seconds - gt), 1)
            if gt > 0:
                entry["relative_error_pct"] = round(abs(seconds - gt) / gt * 100.0, 2)
            elif seconds == 0:
                entry["relative_error_pct"] = 0.0
            else:
                entry["relative_error_pct"] = None  # ground truth 0 but predicted > 0 (false positive)
        report.append(entry)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--ref_image", required=True, help="Ảnh phòng gym đã đánh dấu chấm đỏ tại tâm mỗi thiết bị")
    ap.add_argument("--devices", required=True, help="JSON file: [{id, type}, ...]")
    ap.add_argument("--ground_truth", default=None, help="JSON file: [{id, type, occupied_time_seconds}, ...]")
    ap.add_argument("--weights", default="models/yolov8n-pose.pth")
    ap.add_argument("--sam2_checkpoint", default=None)
    ap.add_argument("--sam2_config", default=None)
    ap.add_argument("--sample_fps", type=float, default=1.0)
    ap.add_argument("--out", default="output.json")
    args = ap.parse_args()

    result = run_pipeline(
        args.video, args.ref_image, args.devices, args.ground_truth, args.weights,
        args.sam2_checkpoint, args.sam2_config, args.sample_fps,
    )
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))