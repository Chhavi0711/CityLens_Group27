"""
CityLens AI Hackathon 2026 — Group 3: Traffic Violations & Enforcement
Main detection script. Run on images or video files.

Usage:
    python detect.py --source path/to/video.mp4
    python detect.py --source path/to/image.jpg
    python detect.py --source path/to/frames_folder/
"""

import argparse
import cv2
import json
import os
import csv
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO
import supervision as sv
import numpy as np

# ─── CONFIG ─────────────────────────────────────────────────────────────────
MODEL_WEIGHTS = {
    "vehicle":  "models/vehicle_detector.pt",   # trained on COCO + Indian traffic
    "helmet":   "models/helmet_detector.pt",    # no-helmet / with-helmet
    "seatbelt": "models/seatbelt_detector.pt",  # seatbelt / no-seatbelt
    "signal":   "models/signal_detector.pt",    # red / green / yellow
}

# Fallback to pretrained YOLOv8n if custom weights don't exist yet
FALLBACK_MODEL = "yolov8n.pt"

VIOLATION_CLASSES = [
    "wrong_way_driving",
    "signal_jump",
    "no_helmet",
    "no_seatbelt",
    "triple_riding",
    "illegal_parking",
]

CONFIDENCE_THRESHOLD = 0.45
IOU_THRESHOLD = 0.45
# ────────────────────────────────────────────────────────────────────────────


def load_model(weight_path: str) -> YOLO:
    """Load YOLO model from weights file, fall back to pretrained if not found."""
    if os.path.exists(weight_path):
        print(f"  ✓ Loaded custom weights: {weight_path}")
        return YOLO(weight_path)
    else:
        print(f"  ⚠ Weights not found at {weight_path}, using fallback: {FALLBACK_MODEL}")
        return YOLO(FALLBACK_MODEL)


def is_wrong_way(track_history: dict, track_id: int, frame_h: int) -> bool:
    """
    Detect wrong-way driving using vertical trajectory.
    Assumes camera is mounted facing oncoming traffic.
    Vehicles moving DOWN the frame (y increasing) are going the correct way.
    Vehicles moving UP the frame (y decreasing) are going wrong way.
    """
    if track_id not in track_history or len(track_history[track_id]) < 10:
        return False
    positions = track_history[track_id]
    # Compare average y from first 5 frames vs last 5 frames
    early_y = np.mean([p[1] for p in positions[:5]])
    late_y  = np.mean([p[1] for p in positions[-5:]])
    # If vehicle moved significantly upward → wrong way
    return (early_y - late_y) > (frame_h * 0.05)


def check_signal_violation(vehicle_box, stop_line_y: int, signal_state: str) -> bool:
    """
    Detect signal jump: vehicle bounding box crosses stop_line_y while signal is red.
    vehicle_box: [x1, y1, x2, y2]
    stop_line_y: pixel row of the stop line
    signal_state: 'red', 'green', 'yellow'
    """
    if signal_state != "red":
        return False
    vehicle_bottom_y = vehicle_box[3]  # y2 of bounding box
    return vehicle_bottom_y > stop_line_y


def detect_signal_color(signal_crop: np.ndarray) -> str:
    """
    Simple HSV-based traffic light color detection from a cropped ROI.
    Returns 'red', 'yellow', or 'green'.
    """
    hsv = cv2.cvtColor(signal_crop, cv2.COLOR_BGR2HSV)

    # HSV ranges for each color
    masks = {
        "red":    cv2.inRange(hsv, (0, 70, 50), (10, 255, 255)) |
                  cv2.inRange(hsv, (160, 70, 50), (180, 255, 255)),
        "yellow": cv2.inRange(hsv, (15, 70, 50), (35, 255, 255)),
        "green":  cv2.inRange(hsv, (40, 70, 50), (90, 255, 255)),
    }
    counts = {color: cv2.countNonZero(mask) for color, mask in masks.items()}
    return max(counts, key=counts.get)


def draw_violation_label(frame: np.ndarray, box, label: str, color=(0, 0, 255)) -> np.ndarray:
    """Draw bounding box and violation label on frame."""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    # Label background
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def run_detection(source: str, output_dir: str = "output", stop_line_y: int = None):
    """
    Main detection pipeline. Processes video or image source,
    detects traffic violations, saves annotated output + bounding box CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_frames_dir = os.path.join(output_dir, "annotated_frames")
    os.makedirs(output_frames_dir, exist_ok=True)

    # ── Load models ──────────────────────────────────────────────────────────
    print("\n[1/4] Loading models...")
    vehicle_model = load_model(MODEL_WEIGHTS["vehicle"])
    helmet_model  = load_model(MODEL_WEIGHTS["helmet"])
    signal_model  = load_model(MODEL_WEIGHTS["signal"])

    # ── Setup tracker ────────────────────────────────────────────────────────
    tracker = sv.ByteTrack()
    track_history = {}   # track_id → list of (cx, cy) positions

    # ── Open source ──────────────────────────────────────────────────────────
    print(f"\n[2/4] Opening source: {source}")
    is_video = source.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    cap = cv2.VideoCapture(source) if is_video else None

    # ── Output CSV for bounding boxes ────────────────────────────────────────
    csv_path = os.path.join(output_dir, "bounding_boxes.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame_id", "track_id", "violation_type",
        "x", "y", "width", "height",
        "confidence", "signal_state"
    ])

    violations_log = []
    frame_id = 0
    signal_state = "unknown"

    print("\n[3/4] Running detection...\n")

    while True:
        # ── Read frame ───────────────────────────────────────────────────────
        if is_video:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            frame = cv2.imread(source)
            if frame is None:
                print(f"  ✗ Could not read image: {source}")
                break

        frame_h, frame_w = frame.shape[:2]
        if stop_line_y is None:
            stop_line_y = int(frame_h * 0.55)  # default: 55% down the frame

        # ── Draw stop line for reference ─────────────────────────────────────
        cv2.line(frame, (0, stop_line_y), (frame_w, stop_line_y), (0, 255, 255), 2)
        cv2.putText(frame, "STOP LINE", (10, stop_line_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # ── Detect vehicles ──────────────────────────────────────────────────
        vehicle_results = vehicle_model(frame, conf=CONFIDENCE_THRESHOLD,
                                        iou=IOU_THRESHOLD, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(vehicle_results)
        detections = tracker.update_with_detections(detections)

        # ── Detect signal color (top-right ROI) ──────────────────────────────
        signal_roi = frame[0:int(frame_h * 0.3), int(frame_w * 0.7):]
        if signal_roi.size > 0:
            signal_state = detect_signal_color(signal_roi)
        signal_color_bgr = {"red": (0,0,255), "yellow": (0,215,255), "green": (0,200,0), "unknown": (128,128,128)}
        cv2.putText(frame, f"SIGNAL: {signal_state.upper()}",
                    (frame_w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    signal_color_bgr.get(signal_state, (255,255,255)), 2)

        # ── Per-vehicle violation checks ─────────────────────────────────────
        for i, (box, track_id, conf) in enumerate(
                zip(detections.xyxy, detections.tracker_id or [], detections.confidence)):

            if track_id is None:
                continue

            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            bw, bh = x2 - x1, y2 - y1

            # Update track history
            if track_id not in track_history:
                track_history[track_id] = []
            track_history[track_id].append((cx, cy))
            if len(track_history[track_id]) > 30:
                track_history[track_id].pop(0)

            violations_found = []

            # ── 1. Wrong-way detection ────────────────────────────────────
            if is_wrong_way(track_history, track_id, frame_h):
                violations_found.append("WRONG WAY")
                frame = draw_violation_label(frame, box, f"#{track_id} WRONG WAY", (0, 0, 255))

            # ── 2. Signal jump detection ──────────────────────────────────
            if check_signal_violation([x1, y1, x2, y2], stop_line_y, signal_state):
                violations_found.append("SIGNAL JUMP")
                frame = draw_violation_label(frame, box, f"#{track_id} SIGNAL JUMP", (0, 0, 200))

            # ── 3. No helmet (run helmet model on vehicle crop) ───────────
            vehicle_crop = frame[max(0,y1):y2, max(0,x1):x2]
            if vehicle_crop.size > 0:
                helmet_results = helmet_model(vehicle_crop, conf=0.4, verbose=False)[0]
                for cls_id, cls_name in enumerate(helmet_results.names.values()):
                    if "no" in cls_name.lower() and "helmet" in cls_name.lower():
                        if len(helmet_results.boxes) > 0:
                            violations_found.append("NO HELMET")
                            frame = draw_violation_label(frame, box, f"#{track_id} NO HELMET", (255, 0, 0))
                            break

            # ── Log all violations for this vehicle ───────────────────────
            for v in violations_found:
                csv_writer.writerow([
                    frame_id, track_id, v,
                    x1, y1, bw, bh,
                    round(float(conf), 3), signal_state
                ])
                violations_log.append({
                    "frame": frame_id, "track_id": int(track_id),
                    "violation": v, "bbox": [x1, y1, bw, bh],
                    "confidence": round(float(conf), 3)
                })

            # ── Draw normal tracking box if no violation ──────────────────
            if not violations_found:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 1)
                cv2.putText(frame, f"#{track_id}", (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)

        # ── Frame metadata overlay ────────────────────────────────────────────
        cv2.putText(frame, f"Frame: {frame_id:04d}", (10, frame_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"Violations: {len(violations_log)}", (150, frame_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)

        # ── Save annotated frame ──────────────────────────────────────────────
        out_path = os.path.join(output_frames_dir, f"frame_{frame_id:04d}.jpg")
        cv2.imwrite(out_path, frame)

        frame_id += 1
        if not is_video:
            break   # single image — stop after one frame

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if cap:
        cap.release()
    csv_file.close()

    # ── Save violations JSON log ──────────────────────────────────────────────
    json_path = os.path.join(output_dir, "violations_log.json")
    with open(json_path, "w") as f:
        json.dump(violations_log, f, indent=2)

    print(f"\n[4/4] Done.")
    print(f"  Frames processed : {frame_id}")
    print(f"  Violations found : {len(violations_log)}")
    print(f"  Annotated frames : {output_frames_dir}/")
    print(f"  Bounding box CSV : {csv_path}")
    print(f"  Violations JSON  : {json_path}\n")

    return violations_log


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CityLens Group 3 — Traffic Violation Detector")
    parser.add_argument("--source",      required=True, help="Path to video, image, or folder")
    parser.add_argument("--output",      default="output", help="Output directory (default: output/)")
    parser.add_argument("--stop-line-y", type=int, default=None,
                        help="Pixel row of stop line (auto-calculated if not set)")
    args = parser.parse_args()

    run_detection(
        source=args.source,
        output_dir=args.output,
        stop_line_y=args.stop_line_y
    )
