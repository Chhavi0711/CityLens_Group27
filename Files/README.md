# CityLens AI Hackathon 2026
## Group 3 — Traffic Violations & Enforcement

**Team:** [Your Team Name]  
**Submission Deadline:** 22 June 2026  
**Model Architecture:** YOLOv8 + ByteTrack  
**Threshold Target:** >85% mAP50 across all categories

---

## Violations Detected

| # | Violation | Detection Method |
|---|-----------|-----------------|
| 1 | Signal Jump (Red Light) | ROI stop-line + traffic light state |
| 2 | Wrong-Way Driving | ByteTrack trajectory direction |
| 3 | No Helmet (Two-Wheeler) | YOLOv8 classification on vehicle crop |
| 4 | No Seatbelt | YOLOv8 classification on driver crop |
| 5 | Triple Riding | Person count on motorcycle crop |

---

## Project Structure

```
citylens_g3/
├── detect.py           ← Main inference script (run this on new footage)
├── train.py            ← Fine-tuning script
├── download_data.py    ← Download Roboflow datasets
├── evaluate.py         ← Generate accuracy report
├── requirements.txt    ← Python dependencies
├── README.md           ← This file
│
├── models/
│   ├── helmet_detector.pt      ← Trained weights (no-helmet detection)
│   ├── vehicle_detector.pt     ← Trained weights (vehicle detection)
│   ├── signal_detector.pt      ← Trained weights (traffic light state)
│   └── seatbelt_detector.pt    ← Trained weights (seatbelt detection)
│
├── data/
│   ├── helmet/         ← Helmet dataset (YOLOv8 format)
│   ├── vehicle/        ← Vehicle dataset
│   ├── signal/         ← Traffic signal dataset
│   └── seatbelt/       ← Seatbelt dataset
│
└── output/
    ├── annotated_frames/   ← Output frames with bounding boxes drawn
    ├── bounding_boxes.csv  ← Per-frame bounding box coordinates
    ├── violations_log.json ← Full violations log
    └── evaluation_report.json ← mAP scores per model
```

---

## Quick Start — Run Inference

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run on a video file
python detect.py --source path/to/cctv_footage.mp4

# 3. Run on a single image
python detect.py --source path/to/frame.jpg

# 4. Specify a custom stop-line position (pixel row)
python detect.py --source video.mp4 --stop-line-y 320

# Output is saved to output/ by default
```

---

## Training from Scratch

### Step 1 — Get Roboflow API Key
Create a free account at https://app.roboflow.com → Settings → API Key

```bash
export ROBOFLOW_API_KEY="your_key_here"
```

### Step 2 — Download Datasets

```bash
python download_data.py --task all
```

If any download fails, visit the fallback URL shown and download manually in YOLOv8 format.

### Step 3 — Train Models

```bash
# Train all detectors (run each in a separate terminal or sequentially)
python train.py --task vehicle    # ~20 min on GPU
python train.py --task helmet     # ~30 min on GPU
python train.py --task signal     # ~25 min on GPU
python train.py --task seatbelt   # ~30 min on GPU
```

GPU recommended. On CPU, reduce epochs to 10–15 and expect ~3–5x longer runtime.

### Step 4 — Evaluate

```bash
python evaluate.py
```

This prints mAP50, precision, recall per model and weighted average.
Target: **≥85% mAP50** on each model.

---

## Output Format

### Bounding Boxes (CSV)
`output/bounding_boxes.csv` — one row per detected violation per frame.

| frame_id | track_id | violation_type | x | y | width | height | confidence | signal_state |
|----------|----------|----------------|---|---|-------|--------|------------|--------------|
| 0042 | 7 | SIGNAL JUMP | 312 | 480 | 145 | 98 | 0.871 | red |
| 0056 | 3 | NO HELMET | 88 | 210 | 64 | 112 | 0.762 | green |

- `x, y` — top-left corner of bounding box
- `width, height` — box dimensions in pixels

### Annotated Frames
`output/annotated_frames/` — JPEG frames with:
- Colored bounding boxes (red = violation, green = normal)
- Track ID labels
- Violation type label
- Stop line overlay
- Signal state indicator

---

## Model Architecture

```
Input Frame (CCTV/Video)
        │
        ▼
┌─────────────────────┐
│  Vehicle Detector   │  YOLOv8s — detects cars, motorcycles, autos
│  (YOLOv8)          │
└────────┬────────────┘
         │ Bounding boxes + class
         ▼
┌─────────────────────┐
│  ByteTrack Tracker  │  Assigns persistent IDs, maintains trajectory
└────────┬────────────┘
         │ Track ID + position history
         ├──────────────────────────────────────────────┐
         ▼                                              ▼
┌────────────────┐                         ┌──────────────────────┐
│ Wrong-Way      │                         │ Vehicle Crop         │
│ Logic          │                         │ (ROI from bbox)      │
│ (trajectory    │                         └──────────┬───────────┘
│  direction)    │                                    │
└────────────────┘              ┌───────────┬─────────┴──────────┐
                                ▼           ▼                    ▼
                     ┌────────────┐ ┌────────────┐    ┌─────────────────┐
                     │  Helmet    │ │  Seatbelt  │    │ Signal Detector │
                     │  Detector  │ │  Detector  │    │ (traffic light  │
                     │  (YOLOv8n) │ │  (YOLOv8n) │    │  HSV + YOLO)   │
                     └────────────┘ └────────────┘    └─────────────────┘
                                                                │
                                                     ┌──────────▼──────────┐
                                                     │ Signal Jump Logic   │
                                                     │ (stop line ROI      │
                                                     │  + red light check) │
                                                     └─────────────────────┘
```

---

## Datasets Used

| Model | Dataset | Source | License |
|-------|---------|--------|---------|
| Helmet | Bike Helmet Detection | Roboflow Universe | Public |
| Vehicle | Vehicle Detection (Indian traffic) | Roboflow Universe | Public |
| Signal | Traffic Light Detection | Roboflow Universe | Public |
| Seatbelt | Seatbelt Detection | Roboflow Universe | Public |
| Base weights | COCO 2017 | COCO Consortium | CC BY 4.0 |

All datasets are open-access and free-to-use. No proprietary data used.

---

## Performance Metrics

*(Fill in after running `python evaluate.py`)*

| Model | mAP50 | mAP50-95 | Precision | Recall |
|-------|-------|----------|-----------|--------|
| helmet_detector | — | — | — | — |
| vehicle_detector | — | — | — | — |
| signal_detector | — | — | — | — |
| seatbelt_detector | — | — | — | — |
| **Weighted Average** | **—** | — | — | — |

---

## External References

- **Ultralytics YOLOv8**: https://github.com/ultralytics/ultralytics
- **Supervision (ByteTrack)**: https://github.com/roboflow/supervision
- **Roboflow Universe**: https://universe.roboflow.com
- **COCO Dataset**: https://cocodataset.org
- Paper: *Traffic Monitoring and Violation Detection Using Deep Learning* (Sargar et al., Springer 2024)
- Paper: *Edge-AI Perception Node for Cooperative Road-Safety Enforcement* (arXiv:2601.07845)

---

## Team

| Name | Role |
|------|------|
| [Member 1] | Model training (vehicle, signal) |
| [Member 2] | Model training (helmet, seatbelt) |
| [Member 3] | Tracking logic + wrong-way detection |
| [Member 4] | Data sourcing + annotation |
| [Member 5] | Documentation + README + evaluation |

---

## Contact

For any issues: arjun.mitra@airawat.org
