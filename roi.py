"""
ROI Configuration module
-------------------------
Implements: Marked Image Processing -> (SAM2 Segmentation, optional) ->
Mask-Equipment Mapping -> ROI Rules by Equipment Type

Per the proposal, SAM2 is used to segment the equipment mask from a point
prompt. SAM2 requires a separate checkpoint download (facebookresearch/sam2)
which needs network access not available in every environment, so this
module supports TWO modes:

  mode="sam2"   -> uses the real SAM2 model (if installed + checkpoint found)
  mode="circle" -> a lightweight geometric fallback: builds a circular /
                   elliptical ROI around each marked point, sized per
                   equipment type. This keeps the whole pipeline runnable
                   end-to-end without any extra heavy downloads, and can be
                   swapped for SAM2 later with no change to downstream code.

Equipment-type -> spatial rule table (keypoint groups used in
spatial_interaction.py) is also defined here, matching section V.4.c of the
proposal (Standing / Sitting-Lying / Hanging).
"""
import json
import math
import numpy as np
import cv2


# Equipment category -> ordered fallback keypoint groups (COCO-17 indices)
# COCO keypoint order:
# 0 nose,1 l_eye,2 r_eye,3 l_ear,4 r_ear,5 l_shoulder,6 r_shoulder,
# 7 l_elbow,8 r_elbow,9 l_wrist,10 r_wrist,11 l_hip,12 r_hip,
# 13 l_knee,14 r_knee,15 l_ankle,16 r_ankle
EQUIPMENT_KEYPOINT_RULES = {
    "standing": [
        {"name": "ankle", "ids": [15, 16], "score": 1.0},
        {"name": "knee", "ids": [13, 14], "score": 0.7},
        {"name": "hip", "ids": [11, 12], "score": 0.5},
    ],
    "sitting_lying": [
        {"name": "hip", "ids": [11, 12], "score": 1.0},
        {"name": "torso_center", "ids": [5, 6, 11, 12], "score": 0.7},
        {"name": "shoulder_knee", "ids": [5, 6, 13, 14], "score": 0.5},
    ],
    "hanging": [
        {"name": "wrist", "ids": [9, 10], "score": 1.0},
        {"name": "elbow", "ids": [7, 8], "score": 0.7},
    ],
}

# Default ROI radius (pixels) per category when using the geometric fallback,
# expressed as a fraction of image min(H, W). Tunable per deployment.
DEFAULT_ROI_RADIUS_FRAC = {
    "standing": 0.09,
    "sitting_lying": 0.12,
    "hanging": 0.10,
}


def infer_category(type_str):
    """
    Best-effort mapping from a free-text equipment `type` to one of the
    three keypoint-rule categories, so the user only has to supply
    (id, type) [+ x, y] and does not need to know about `category` at all.
    Override by adding an explicit "category" field per item if needed.
    """
    t = (type_str or "").lower()
    hanging_kw = ["pull", "bar", "hang", "dip", "xà"]
    sitting_kw = ["press", "bench", "row", "curl", "leg", "extension",
                  "machine", "lat", "fly", "cable", "bike", "cycle",
                  "ép", "ghế", "đạp"]
    standing_kw = ["treadmill", "elliptical", "stair",
                   "squat", "rack", "chạy"]
    for kw in hanging_kw:
        if kw in t:
            return "hanging"
    for kw in sitting_kw:
        if kw in t:
            return "sitting_lying"
    for kw in standing_kw:
        if kw in t:
            return "standing"
    return "standing"  # safe default


def load_devices(devices_json_path):
    """
    Setup Input #3: equipment list file, containing ONLY (id, type) - no
    coordinates, matching what the user actually has on disk:

    [
      {"id": "M01", "type": "Elliptical"},
      {"id": "M02", "type": "Treadmill"},
      ...
    ]

    `category` is auto-inferred from `type` via `infer_category` (can be
    overridden by adding an explicit "category" field per item).
    Returns the list of device dicts, in file order (the order is used to
    match against red dots auto-detected on the reference image, see
    `match_dots_to_devices` below).
    """
    with open(devices_json_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    devices = []
    for it in items:
        cat = it.get("category") or infer_category(it.get("type"))
        assert cat in EQUIPMENT_KEYPOINT_RULES, (
            f"Unknown category '{cat}' for equipment {it['id']}. "
            f"Must be one of {list(EQUIPMENT_KEYPOINT_RULES.keys())}"
        )
        dev = {"id": it["id"], "type": it.get("type", it["id"]), "category": cat}
        # Tọa độ (x, y) tâm thiết bị là TÙY CHỌN. Nếu có, hệ thống sẽ dùng
        # trực tiếp tọa độ này thay vì tự đoán thứ tự ghép chấm đỏ -> ID,
        # tránh hoàn toàn rủi ro ghép nhầm khi layout không đều (ví dụ
        # thiết bị khác loại có chấm đánh dấu ở độ cao khác nhau trên cùng
        # một hàng, khiến thuật toán tự đoán hàng bị nhầm).
        if "x" in it and "y" in it:
            dev["x"] = float(it["x"])
            dev["y"] = float(it["y"])
        devices.append(dev)
    return devices


def load_ground_truth(ground_truth_json_path):
    """
    Setup Input #4 (for accuracy evaluation only, not used by the
    occupancy algorithm itself): a separate file with manually-observed
    occupied time per device, e.g.

    [
      {"id": "M01", "type": "Elliptical", "occupied_time_seconds": 0},
      {"id": "M03", "type": "Treadmill", "occupied_time_seconds": 57},
      ...
    ]

    Returns {id: seconds}.
    """
    with open(ground_truth_json_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    return {it["id"]: it["occupied_time_seconds"] for it in items}


def detect_red_dots(image_bgr, min_area=15):
    """
    Phát hiện các chấm đỏ trên ảnh tham chiếu bằng cách ngưỡng màu HSV,
    trả về danh sách tọa độ tâm (cx, cy) theo pixel.

    Dải màu đỏ trong HSV bao gồm 2 đoạn (vì đỏ vắt qua H=0/180):
      - [0..10] và [170..180] cho Hue
      - Saturation >= 120, Value >= 70 để lọc bỏ xám/trắng/tối.
    Blob nhỏ hơn min_area pixel bị bỏ qua (loại nhiễu nhỏ).
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower1, upper1 = np.array([0,   120, 70]), np.array([10,  255, 255])
    lower2, upper2 = np.array([170, 120, 70]), np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        centers.append((cx, cy))
    return centers


def _sort_dots_reading_order(centers, row_gap_factor=0.5):
    """
    Sắp xếp các tâm chấm đỏ theo thứ tự đọc (trên->dưới, trái->phải).

    --- Vấn đề với avg_gap (bản cũ) ---
    Phòng gym thường có nhiều loại thiết bị khác nhau trên cùng một hàng
    vật lý (ví dụ: Elliptical + Treadmill). Chấm đánh dấu trên mỗi loại
    có thể nằm ở độ cao (y) rất khác nhau dù chúng ở cùng một hàng thực
    tế — tay cầm Elliptical cao hơn mặt băng chuyền Treadmill ~90px trong
    khi khoảng cách giữa các hàng vật lý chỉ ~230px. Kết quả: avg_gap ≈ 54px,
    threshold ≈ 32px → khoảng cách 91px giữa Elliptical và Treadmill vượt
    ngưỡng → thuật toán tách nhầm thành 2 hàng riêng biệt → toàn bộ ID bị
    gán lệch.

    --- Cách sửa: dùng max_gap ---
    Ngưỡng tách hàng = max_gap * row_gap_factor (mặc định 0.5).
    Với ví dụ trên: max_gap = 229px (khoảng cách giữa 2 hàng vật lý thật),
    threshold = 114.5px. Khoảng cách 91px (nội bộ cùng hàng) không vượt
    ngưỡng → 2 Elliptical và 3 Treadmill được gom đúng vào 1 hàng → sort
    theo x cho ra thứ tự: Elliptical(trái), Treadmill×3, Elliptical(phải).

    Tính chất: max_gap * 0.5 đảm bảo chỉ những khoảng cách lớn hơn
    "nửa khoảng trống giữa các hàng thật" mới bị coi là ranh giới hàng,
    còn biến thiên độ cao do loại thiết bị khác nhau (luôn nhỏ hơn khoảng
    cách giữa 2 hàng) sẽ không bao giờ tạo thành hàng giả.
    """
    if len(centers) <= 1:
        return list(centers)

    by_y = sorted(centers, key=lambda c: c[1])
    y_gaps = [by_y[i + 1][1] - by_y[i][1] for i in range(len(by_y) - 1)]
    max_gap = max(y_gaps) if y_gaps else 0.0
    # Chỉ tách hàng mới khi gap > row_gap_factor * max_gap.
    # Với row_gap_factor=0.5: khoảng cách trong cùng một hàng (dù khác loại
    # máy) luôn < max_gap/2 nên sẽ không bị tách sai.
    row_threshold = max(max_gap * row_gap_factor, 1.0)

    rows = [[by_y[0]]]
    for prev, cur in zip(by_y, by_y[1:]):
        if (cur[1] - prev[1]) > row_threshold:
            rows.append([cur])
        else:
            rows[-1].append(cur)

    ordered = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda c: c[0]))
    return ordered


def match_dots_to_devices(image_bgr, devices):
    """
    Marked Image Processing (V.2): phát hiện chấm đỏ trên ảnh tham chiếu
    và ghép với từng thiết bị trong devices (theo thứ tự trái->phải,
    trên->dưới), vì devices.json chỉ có (id, type), không có tọa độ.

    Raises ValueError nếu số chấm phát hiện được ≠ số thiết bị, để người
    dùng biết cần sửa ảnh hoặc file JSON.

    Returns: danh sách points (id, type, category, x, y) cho build_roi.
    """
    centers = detect_red_dots(image_bgr)
    if len(centers) != len(devices):
        raise ValueError(
            f"Phát hiện {len(centers)} chấm đỏ trên ảnh nhưng devices.json có "
            f"{len(devices)} thiết bị. Vui lòng đánh dấu đúng 1 chấm đỏ cho "
            f"mỗi thiết bị trong danh sách (theo đúng thứ tự trái->phải, "
            f"trên->dưới của devices.json)."
        )
    centers_sorted = _sort_dots_reading_order(centers)
    points = []
    for dev, (cx, cy) in zip(devices, centers_sorted):
        points.append({
            "id": dev["id"], "type": dev["type"], "category": dev["category"],
            "x": cx, "y": cy,
        })
    return points


def save_roi_debug_image(image_bgr, roi_config, out_path):
    """
    Vẽ ROI (vòng tròn/bbox + ID) lên ảnh tham chiếu và lưu ra file, để kiểm
    tra trực quan xem ROI có khớp đúng vị trí + đúng thiết bị thật hay
    không. Dùng cv2.imencode + tofile để hỗ trợ đường dẫn Unicode trên
    Windows (xem pipeline.imwrite_unicode).

    Gợi ý debug: nếu chạy ra kết quả sai (toàn 0s hoặc gán nhầm thiết bị),
    hãy gọi hàm này trước rồi mở ảnh lên xem từng vòng tròn ID có nằm
    đúng lên đúng máy tập tương ứng trong ảnh hay không.
    """
    vis = image_bgr.copy()
    for eid, roi in roi_config.items():
        cx, cy = int(roi["center"][0]), int(roi["center"][1])
        x0, y0, x1, y1 = [int(v) for v in roi["bbox"]]
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(vis, str(eid), (x0, max(0, y0 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    ok, buf = cv2.imencode(".png", vis)
    if ok:
        buf.tofile(out_path)
    return out_path



def build_roi_geometric(image_shape, points):
    """
    Fallback for (5)+(6)+(7): build a per-equipment ROI mask directly from
    the marked center point, without requiring SAM2.

    ROI shape strategy per category
    --------------------------------
    "standing" / "hanging": symmetric circle around the marker dot.
      Marker is placed on the activity zone (running belt, pull-up bar)
      so symmetric coverage works.

    "sitting_lying" (bikes, benches, cable machines…): ASYMMETRIC rectangle.
      The marker is placed at the handlebar/grip — the natural visible anchor
      of the equipment — but the person's body is mostly *below* that point:
        - hips (primary keypoint) → at seat level, ~25 % of min_dim below
        - knees / ankles (fallback) → even further below at pedal level
      A symmetric circle centred at the handlebar leaves the hips outside
      the ROI, causing I(t) = 0 for every frame → detected time = 0 s.
      The fix: extend the ROI much further downward than upward.

    All fractions are relative to min(H, W) of the reference image.
    """
    H, W = image_shape[:2]
    min_dim = min(H, W)
    roi_config = {}

    # Fractions for sitting_lying asymmetric ROI
    _SL_UP   = 0.05   # small margin above marker (handlebar visible above)
    _SL_DOWN = 0.27   # reach down past seat to upper pedal area
    _SL_SIDE = 0.13   # horizontal half-width

    for p in points:
        cat = p["category"]
        cx, cy = float(p["x"]), float(p["y"])

        if cat == "sitting_lying":
            # Asymmetric rectangle: modest headroom above handlebar,
            # tall coverage below down to yên + bàn đạp.
            half_w = _SL_SIDE * min_dim
            y_top  = max(0.0, cy - _SL_UP   * min_dim)
            y_bot  = min(float(H), cy + _SL_DOWN * min_dim)
            x0, y0 = cx - half_w, y_top
            x1, y1 = cx + half_w, y_bot
            # Effective center for D(t) = midpoint of the ROI
            # (roughly at seat level, where person's hips actually are)
            eff_cx = cx
            eff_cy = (y_top + y_bot) / 2.0
            mask = np.zeros((H, W), dtype=np.uint8)
            cv2.rectangle(
                mask,
                (int(max(0, x0)), int(max(0, y0))),
                (int(min(W - 1, x1)), int(min(H - 1, y1))),
                1, thickness=-1,
            )
        else:
            # Symmetric circle for standing / hanging
            radius = DEFAULT_ROI_RADIUS_FRAC.get(cat, 0.10) * min_dim
            x0, y0 = cx - radius, cy - radius
            x1, y1 = cx + radius, cy + radius
            eff_cx, eff_cy = cx, cy
            mask = np.zeros((H, W), dtype=np.uint8)
            cv2.circle(mask, (int(cx), int(cy)), int(radius), 1, thickness=-1)

        x0 = max(0.0, x0); y0 = max(0.0, y0)
        x1 = min(float(W), x1); y1 = min(float(H), y1)
        diag = math.hypot(x1 - x0, y1 - y0)

        roi_config[p["id"]] = {
            "id": p["id"],
            "type": p.get("type", p["category"]),
            "category": cat,
            "center": [eff_cx, eff_cy],   # effective center for D(t)
            "marker": [cx, cy],            # original dot position (for debug)
            "bbox": [x0, y0, x1, y1],
            "diag": float(diag),
            "mask": mask,
            "keypoint_rules": EQUIPMENT_KEYPOINT_RULES[cat],
        }
    return roi_config


def build_roi_sam2(image_bgr, points, sam2_checkpoint, sam2_config):
    """
    Real (5)-(7) pipeline using SAM2 point-prompt segmentation.
    Only runs if `sam2` package + checkpoint are available locally.
    Falls back automatically to the geometric method otherwise (see
    `build_roi` below), so the rest of the pipeline never has to know
    which backend produced the masks.
    """
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    sam2_model = build_sam2(sam2_config, sam2_checkpoint, device="cpu")
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    H, W = image_bgr.shape[:2]
    roi_config = {}
    for p in points:
        cx, cy = float(p["x"]), float(p["y"])
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[cx, cy]]),
            point_labels=np.array([1]),
            multimask_output=False,
        )
        mask = masks[0].astype(np.uint8)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
        diag = math.hypot(x1 - x0, y1 - y0)
        cat = p["category"]
        roi_config[p["id"]] = {
            "id": p["id"],
            "type": p.get("type", p["category"]),
            "category": cat,
            "center": [cx, cy],
            "bbox": [float(x0), float(y0), float(x1), float(y1)],
            "diag": float(diag),
            "mask": mask,
            "keypoint_rules": EQUIPMENT_KEYPOINT_RULES[cat],
        }
    return roi_config


def build_roi(image_bgr, points, sam2_checkpoint=None, sam2_config=None):
    """
    Entry point used by the rest of the pipeline (section IV.3: ROI
    Configuration JSON, item (7)). Tries SAM2 if a checkpoint path is given
    and the package is importable; otherwise uses the geometric fallback.
    """
    if sam2_checkpoint:
        try:
            return build_roi_sam2(image_bgr, points, sam2_checkpoint, sam2_config)
        except Exception as e:
            print(f"[roi] SAM2 unavailable ({e}); falling back to geometric ROI.")
    return build_roi_geometric(image_bgr.shape, points)