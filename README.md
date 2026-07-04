# Smart Gym Occupancy Management - Demo

Hệ thống tự động đo lường **thời gian chiếm dụng** của từng thiết bị tập gym
thông qua phân tích video camera giám sát sẵn có, không cần lắp thêm phần cứng.

📽️ **Video demo:** https://drive.google.com/file/d/1MNchXOYW1Dh4A1ToEAKG-JmNJOitcwUn/view?usp=sharing

---

## Giới thiệu

Đề tài thuộc môn **CS117 — Computational Thinking**, giải quyết bài toán:
chủ phòng gym khó xác định thiết bị nào đang được sử dụng nhiều, khung giờ
nào quá tải, và thiết bị nào cần bảo trì — nếu chỉ dựa vào quan sát thủ công.

Hệ thống khai thác nguồn video từ camera an ninh có sẵn để tự động:

- Xác định trạng thái **Occupied / Free** của từng thiết bị theo thời gian.
- Tính tổng thời gian chiếm dụng của từng thiết bị trong suốt video.
- Xuất báo cáo kết quả kèm so sánh với Ground Truth (sai số kỳ vọng ≤ 8%).

---

## Kiến trúc pipeline
Frame Extraction (1 FPS)
↓
ROI Configuration
├─ Auto-detect chấm đỏ trên ảnh mẫu (OpenCV)
├─ SAM2 segmentation (hoặc geometric fallback nếu không có SAM2)
└─ Áp dụng ROI rules theo loại thiết bị
↓
YOLO-Pose — Person Detection (bounding box + 17 keypoints)
↓
Spatial Interaction Score:  S(t) = αO(t) + βI(t) + γD(t)
↓
Occupied/Free State Machine  (ngưỡng T_in / T_out chống nhiễu)
↓
Time Accumulation  (cộng dồn frame Occupied → giây)
↓
Output Report

---

## Input cần chuẩn bị

### 1. Empty Gym Image
Ảnh phòng gym **không có người**, đã đánh dấu **1 chấm đỏ tại tâm mỗi thiết bị**,
chụp cùng góc quay với video. Hệ thống tự động phát hiện chấm đỏ bằng OpenCV
(hàm `detect_red_dots` trong `roi.py`), sau đó khớp theo thứ tự **trái → phải,
trên → dưới** với danh sách thiết bị trong `devices.json`.

> Số lượng chấm đỏ trên ảnh phải khớp đúng số thiết bị khai báo trong
> `devices.json`. Nếu không khớp, hệ thống sẽ báo lỗi và cho phép chỉnh lại.

### 2. Gym Camera Footage
Video định dạng `.mp4`, quay cùng góc với ảnh mẫu ở trên, độ phân giải tối
thiểu khuyến nghị 1280×720.

### 3. devices.json
Chỉ cần khai báo `(id, type)`:

```json
[
  { "id": "M01", "type": "Elliptical" },
  { "id": "M02", "type": "Treadmill" }
]
```

Trường `category` (`standing` / `sitting_lying` / `hanging`) được **tự động
suy luận** từ `type` thông qua hàm `infer_category` trong `roi.py`
(ví dụ: Treadmill/Elliptical → standing, Bike → sitting_lying, ...).

### 4. groundtruth.json (tùy chọn, dùng để đánh giá độ chính xác)

```json
[
  { "id": "M01", "type": "Elliptical", "occupied_time_seconds": 0 },
  { "id": "M03", "type": "Treadmill", "occupied_time_seconds": 57 }
]
```

---

## Cài đặt

```bash
pip install -r requirements.txt
python setup_models.py     # tải yolov8n-pose.pt -> models/yolov8n-pose.pth
```

> Cần kết nối internet để tải pretrained weight của YOLO-Pose ở bước này.

---

## Chạy demo

```bash
python app.py
```

Sau khi chạy, terminal sẽ in ra một đường link (dạng `http://127.0.0.1:xxxx`
hoặc link Gradio public). Mở link đó bằng trình duyệt để dùng giao diện:
tải ảnh mẫu + video lên, bấm **Analyze Occupancy Time**, và xem bảng kết quả.

---

## Chạy bằng CLI 

```bash
python pipeline.py --video video.mp4 --ref_image room_marked.jpg \
    --devices sample_data/devices.json \
    --ground_truth sample_data/groundtruth.json \
    --weights models/yolov8n-pose.pth --out output.json
```

Kết quả sẽ được ghi ra file `output.json` theo định dạng:

```json
[
  { "id": "M01", "type": "Treadmill", "occupied_time_seconds": 1320 }
]
```

---

## Cấu trúc file

| File | Chức năng |
|---|---|
| `roi.py` | `load_devices()`, `load_ground_truth()`, `match_dots_to_devices()` (auto-detect chấm đỏ), `infer_category()`, `build_roi()` (SAM2 / geometric fallback) |
| `pose_detection.py` | Wrapper cho YOLO-Pose (Person Detection) |
| `spatial_interaction.py` | Tính điểm tương tác `S(t) = αO(t) + βI(t) + γD(t)` |
| `state_machine.py` | Occupied/Free Finite State Machine + Time Accumulation |
| `pipeline.py` | Orchestrator — nhận 4 input, chạy toàn bộ pipeline |
| `setup_models.py` | Tải sẵn pretrained weight cho YOLO-Pose |
| `app.py` | Giao diện Gradio, hiển thị dashboard + tính Accuracy/MAE |
| `sample_data/devices.json`, `groundtruth.json` | File mẫu để test thử |
| `requirements.txt` | Danh sách thư viện cần cài |

---

## Tham số có thể tinh chỉnh

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `sample_fps` | 1 | Số frame lấy mẫu mỗi giây |
| `T_in` | 3 | Số frame liên tiếp cần có tương tác để xác nhận Occupied |
| `T_out` | 5 | Số frame liên tiếp không tương tác để xác nhận Free |
| `score_threshold` (θ) | 0.5 | Ngưỡng điểm S(t) tối thiểu để coi là có tương tác |

Các tham số này có thể hiệu chỉnh trên một tập video validation thực tế để
đạt **Accuracy ≥ 92%** (tương ứng sai số tương đối ≤ 8% so với Ground Truth).

---

## Giới hạn hiện tại

- Kết quả demo cho ra chính xác nhất với video/ảnh mẫu đính kèm trong link
  Drive ở trên. Với video khác góc quay/ánh sáng/bố trí thiết bị khác, độ
  chính xác có thể giảm và cần hiệu chỉnh lại `T_in`, `T_out`, `score_threshold`.
- Hệ thống chỉ xử lý video đã lưu (offline), chưa hỗ trợ video thời gian thực.
- Không thực hiện nhận diện danh tính hay theo dõi hành vi cá nhân của người tập.

---

## 📄 License

Đồ án học tập, chỉ sử dụng cho mục đích môn học CS117.
