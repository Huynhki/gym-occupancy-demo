# Smart Gym Occupancy Management — demo


LINK VIDEO DEMO: https://drive.google.com/file/d/1CnfXUf7FvCQ1FIi_-YJa7FojGBJck_Mr/view?usp=sharing
Triển khai theo kiến trúc đề tài CS117: Frame Extraction → ROI Configuration
(auto-detect chấm đỏ trên ảnh mẫu → SAM2/geometric fallback → ROI rules) →
YOLO-Pose Person Detection → Spatial Interaction Score S(t) → Occupied/Free
State Machine (T_in/T_out) → Time Accumulation → Output Report → so sánh
các các thông số sau đánh giá với kỳ vọng.

## Input
1. **Empty Gym Image** — ảnh phòng gym không người, **đã đánh dấu 1 chấm đỏ
   tại tâm mỗi thiết bị**, cùng góc quay với video. Hệ thống tự phát hiện
   chấm đỏ bằng OpenCV (`detect_red_dots` trong `roi.py`) rồi khớp theo thứ
   tự trái→phải, trên→dưới với danh sách trong `devices.json`.
2. **Gym Camera Footage** — video .mp4.
3. **devices.json** — chỉ gồm `(id, type)`, đúng định dạng bạn gửi:
   ```json
   [
     { "id": "M01", "type": "Elliptical" },
     { "id": "M02", "type": "Treadmill" }
   ]
   ```
   `category` (standing/sitting_lying/hanging) được tự suy luận từ `type`
   qua `infer_category` trong `roi.py` (Treadmill/Elliptical → standing,
   Bike → sitting_lying, ...).
4. **groundtruth.json**:
   ```json
   [
     { "id": "M01", "type": "Elliptical", "occupied_time_seconds": 0 },
     { "id": "M03", "type": "Treadmill", "occupied_time_seconds": 57 }
   ]
   ```

Lưu ý cho user: Số lượng chấm đỏ trên ảnh phải khớp đúng số thiết bị trong `devices.json`,
nếu không hệ thống sẽ báo lỗi và cho phép chỉnh lại.


## Cài đặt

```bash
pip install -r requirements.txt
python setup_models.py     # tải yolov8n-pose.pt -> models/yolov8n-pose.pth
```
(Cần internet để tải weight YOLO-Pose)

## Chạy demo

```bash
python app.py
```
## Chạy CLI

```bash
python pipeline.py --video video.mp4 --ref_image room_marked.jpg \
    --devices sample_data/devices.json \
    --ground_truth sample_data/groundtruth.json \
    --weights models/yolov8n-pose.pth --out output.json
```

## Cấu trúc file

```
roi.py                  load_devices() (id,type), load_ground_truth(),
                         match_dots_to_devices() (auto-detect chấm đỏ),
                         infer_category(), build_roi() (SAM2/geometric).
pose_detection.py        YOLO-Pose wrapper (Person Detection)
spatial_interaction.py   S(t) = alpha*O(t) + beta*I(t) + gamma*D(t)
state_machine.py         Occupied/Free FSM + Time Accumulation
pipeline.py               Orchestrator, nhận 4 input
setup_models.py           Tải pretrained YOLO-Pose weight
app.py                     Gradio dashboard, tính Accuracy/MAE
sample_data/devices.json, groundtruth.json   File mẫu (từ bạn cung cấp)
requirements.txt
```

## Tham số có thể chỉnh

- `sample_fps` (mặc định 1 FPS), `T_in` (3), `T_out` (5), `score_threshold` (0.5).
Có thể hiệu chỉnh trên tập video validation thực tế để đạt Accuracy ≥ 92%
(tương ứng sai số ≤ 8%).
