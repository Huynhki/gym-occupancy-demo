"""
Gradio demo - "Smart Gym Occupancy Management" dashboard.

4 inputs:
  1) Empty gym image, RED DOTS marked at each equipment's center  -> gr.Image
  2) Gym camera footage                                            -> gr.Video
  3) devices.json:    [{id, type}, ...]                            -> gr.File
  4) groundtruth.json:[{id, type, occupied_time_seconds}, ...]     -> gr.File (optional)

Run:
    python app.py

Requires models/yolov8n-pose.pth to exist (run setup_models.py first).
"""
import traceback

import gradio as gr

from pipeline import run_pipeline

ACCURACY_TOLERANCE_PCT = 8.0  # per đề tài requirement: sai số <= 8%

CSS = """
.stat-card {background:#fff;border:1px solid #e5e7eb;border-radius:10px;
  padding:14px 18px;display:flex;align-items:center;justify-content:space-between;}
.stat-card .label {font-size:12px;color:#6b7280;margin-bottom:4px;}
.stat-card .value {font-size:24px;font-weight:700;color:#111827;}
.stat-card .value.red {color:#dc2626;}
.stat-card .value.green {color:#16a34a;}
.stat-card .value.amber {color:#d97706;}
.stat-card .value.blue {color:#2563eb;}
.panel-title {font-weight:600;font-size:15px;margin-bottom:6px;}
.metrics-box {background:#eef4ff;border-radius:8px;padding:10px 14px;
  display:flex;gap:28px;font-size:13px;margin-top:10px;flex-wrap:wrap;}
.metrics-box b {color:#111827;}
.algo-note {font-size:12px;color:#6b7280;margin-top:8px;}
"""


def stat_cards_html(total, occupied, free, rate):
    return f"""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:14px;">
      <div class="stat-card"><div><div class="label">Total Equipments</div>
        <div class="value">{total}</div></div>🧩</div>
      <div class="stat-card"><div><div class="label">Currently Occupied</div>
        <div class="value red">{occupied}</div></div>🧍</div>
      <div class="stat-card"><div><div class="label">Available (Free)</div>
        <div class="value green">{free}</div></div>✅</div>
      <div class="stat-card"><div><div class="label">Occupancy Rate</div>
        <div class="value amber">{rate}%</div></div>🥧</div>
    </div>
    """


def metrics_html(mae, avg_rel_err, has_gt):
    if not has_gt:
        return ('<div class="metrics-box">Chưa upload groundtruth.json nên không '
                'tính được MAE / Relative Error.</div>'
                '<div class="algo-note">ℹ️ Algorithm used: YOLO Pose + SAM2/Geometric ROI '
                '+ Spatial Interaction State Machine</div>')
    return f"""
    <div class="metrics-box">
      <div>📊 <b>Overall Evaluation Metrics</b></div>
      <div>Mean Absolute Error (MAE): <b>{mae}s</b></div>
      <div>Avg Relative Error: <b>{avg_rel_err}%</b></div>
    </div>
    <div class="algo-note">ℹ️ Algorithm used: YOLO Pose + SAM2/Geometric ROI + Spatial Interaction State Machine</div>
    """


def run_demo(ref_image_file, video_file, devices_file, groundtruth_file,
             sample_fps, t_in, t_out, score_threshold):
    empty_cards = stat_cards_html(0, 0, 0, 0)
    if ref_image_file is None or video_file is None or devices_file is None:
        return (empty_cards, [],
                "Lỗi: vui lòng upload đủ ảnh phòng gym, video, và devices.json.",
                gr.Button(value="Analyze"))

    try:
        report = run_pipeline(
            video_path=video_file,
            ref_image_path=ref_image_file,
            devices_json_path=devices_file,
            ground_truth_json_path=groundtruth_file,
            weights_path="models/yolov8n-pose.pth",
            sample_fps=sample_fps,
            t_in=int(t_in),
            t_out=int(t_out),
            score_threshold=score_threshold,
        )

        total = len(report)
        occupied = sum(1 for r in report if r["occupied_time_seconds"] > 0)
        free = total - occupied
        rate = round(100 * occupied / total, 0) if total else 0

        table = []
        errors = []          # (abs_error, rel_error) when rel_error is a number
        within_tol = []      # bool list for accuracy
        for r in report:
            gt = r.get("ground_truth_seconds")
            rel_err = r.get("relative_error_pct")
            if gt is None:
                gt_disp, err_disp = "-", "-"
            else:
                gt_disp = f"{gt}s"
                if rel_err is None:
                    err_disp = "False Positive"
                    within_tol.append(False)
                else:
                    err_disp = f"{rel_err}%"
                    errors.append((r["abs_error_seconds"], rel_err))
                    within_tol.append(rel_err <= ACCURACY_TOLERANCE_PCT)
            table.append([r["id"], r["type"], gt_disp, f"{r['occupied_time_seconds']}s", err_disp])

        if within_tol:
            mae = round(sum(e[0] for e in errors) / len(errors), 2) if errors else None
            avg_rel = round(sum(e[1] for e in errors) / len(errors), 2) if errors else None
            m_html = metrics_html(mae, avg_rel, True)
        else:
            m_html = metrics_html(None, None, False)

        cards = stat_cards_html(total, occupied, free, int(rate))
        return cards, table, m_html, gr.Button(value="Re-Analyze")

    except Exception as e:
        err = f"Lỗi: {e}\n\n{traceback.format_exc()}"
        return (empty_cards, [], 
                f'<pre style="color:#dc2626;font-size:11px;white-space:pre-wrap;">{err}</pre>', 
                gr.Button(value="Analyze"))


with gr.Blocks(title="Smart Gym Occupancy Management") as demo:
    gr.HTML(
        '<div style="display:flex;justify-content:space-between;align-items:center;'
        'border-left:5px solid #2563eb;background:#fff;padding:14px 18px;'
        'border-radius:8px;border:1px solid #e5e7eb;">'
        '<div><span style="font-size:20px;font-weight:700;">🏋️ Smart Gym Occupancy Management</span></div>'
        '<div style="color:#6b7280;font-size:13px;">Gym equipment occupancy analysis from camera footage</div>'
        '</div>'
    )

    cards_html = gr.HTML(stat_cards_html(0, 0, 0, 0))

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown('<div class="panel-title">📋 Inputs</div>')
            gr.Markdown("**1. Upload Empty Gym Image (đã đánh dấu chấm đỏ tại tâm mỗi thiết bị)**")
            ref_img_in = gr.Image(label=None, type="filepath", show_label=False)
            gr.Markdown("**2. Upload Gym Camera Footage (Operational)**")
            video_in = gr.Video(label=None, show_label=False)
            gr.Markdown("**3. Upload devices.json — danh sách (id, type)**")
            devices_file_in = gr.File(file_types=[".json"], label=None, show_label=False)
            gr.Markdown("**4. Upload groundtruth.json — (id, type, occupied_time_seconds), để tính Accuracy**")
            groundtruth_file_in = gr.File(file_types=[".json"], label=None, show_label=False)

            with gr.Accordion("Tham số nâng cao", open=False):
                sample_fps_in = gr.Slider(0.5, 5.0, value=1.0, step=0.5, label="Sampling rate (FPS)")
                t_in_in = gr.Slider(1, 10, value=3, step=1, label="T_in (frame xác nhận bắt đầu)")
                t_out_in = gr.Slider(1, 10, value=5, step=1, label="T_out (frame xác nhận kết thúc)")
                thresh_in = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Ngưỡng điểm S(t)")

            run_btn = gr.Button("Analyze", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown('<div class="panel-title">📈 Analysis Results</div>')
            table_out = gr.Dataframe(
                headers=["ID", "Equipment", "Ground Truth", "Predicted Time", "Relative Error"],
                label=None,
            )
            metrics_out = gr.HTML(metrics_html(None, None, False))

    run_btn.click(
        run_demo,
        inputs=[ref_img_in, video_in, devices_file_in, groundtruth_file_in,
                sample_fps_in, t_in_in, t_out_in, thresh_in],
        outputs=[cards_html, table_out, metrics_out, run_btn],
    )

if __name__ == "__main__":
    try:
        demo.launch(css=CSS)
    except TypeError:
        demo.launch()