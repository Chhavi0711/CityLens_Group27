"""
CityLens AI Hackathon 2026 — Group 3
Training script: fine-tune YOLOv8 on traffic violation datasets.

Run once per violation type. Downloads datasets from Roboflow
and trains a YOLOv8 model. Saves weights to models/ folder.

Usage:
    python train.py --task helmet
    python train.py --task vehicle
    python train.py --task signal
    python train.py --task seatbelt
"""

import argparse
import os
from pathlib import Path
from ultralytics import YOLO

# ─── TRAINING CONFIG ──────────────────────────────────────────────────────────
TRAIN_CONFIG = {
    "helmet": {
        "model":      "yolov8n.pt",          # start from COCO pretrained nano
        "epochs":     30,
        "imgsz":      640,
        "batch":      16,
        "data_yaml":  "data/helmet/data.yaml",
        "out_name":   "helmet_detector",
        "note":       "Detects: with-helmet, no-helmet, license-plate"
    },
    "vehicle": {
        "model":      "yolov8s.pt",          # small model — better recall
        "epochs":     20,
        "imgsz":      640,
        "batch":      8,
        "data_yaml":  "data/vehicle/data.yaml",
        "out_name":   "vehicle_detector",
        "note":       "Detects: car, motorcycle, auto, truck, bus"
    },
    "signal": {
        "model":      "yolov8n.pt",
        "epochs":     25,
        "imgsz":      416,
        "batch":      16,
        "data_yaml":  "data/signal/data.yaml",
        "out_name":   "signal_detector",
        "note":       "Detects: red-light, green-light, yellow-light"
    },
    "seatbelt": {
        "model":      "yolov8n.pt",
        "epochs":     30,
        "imgsz":      640,
        "batch":      16,
        "data_yaml":  "data/seatbelt/data.yaml",
        "out_name":   "seatbelt_detector",
        "note":       "Detects: seatbelt, no-seatbelt"
    },
}

# Data augmentation config (applied during training)
AUGMENTATION_CONFIG = {
    "hsv_h":   0.015,   # hue shift — handles lighting variation
    "hsv_s":   0.7,     # saturation — day/dusk/night robustness
    "hsv_v":   0.4,     # brightness variation
    "degrees":  5.0,    # slight rotation
    "translate": 0.1,
    "scale":    0.5,
    "fliplr":   0.5,    # horizontal flip
    "mosaic":   1.0,    # mosaic augmentation — great for small objects
    "mixup":    0.1,
}


def train(task: str, resume: bool = False):
    """Fine-tune a YOLOv8 model for a specific violation detection task."""

    if task not in TRAIN_CONFIG:
        print(f"  ✗ Unknown task '{task}'. Choose from: {list(TRAIN_CONFIG.keys())}")
        return

    cfg = TRAIN_CONFIG[task]
    print(f"\n{'='*55}")
    print(f"  Training: {task.upper()} detector")
    print(f"  Note    : {cfg['note']}")
    print(f"  Model   : {cfg['model']}")
    print(f"  Epochs  : {cfg['epochs']}")
    print(f"  Data    : {cfg['data_yaml']}")
    print(f"{'='*55}\n")

    # Check data.yaml exists
    if not os.path.exists(cfg["data_yaml"]):
        print(f"  ✗ data.yaml not found at: {cfg['data_yaml']}")
        print(f"  → Run: python download_data.py --task {task}")
        print(f"    to download the dataset from Roboflow first.\n")
        return

    # Load model
    model = YOLO(cfg["model"])

    # Train
    results = model.train(
        data      = cfg["data_yaml"],
        epochs    = cfg["epochs"],
        imgsz     = cfg["imgsz"],
        batch     = cfg["batch"],
        name      = cfg["out_name"],
        project   = "models",
        resume    = resume,
        patience  = 10,         # early stopping if no improvement for 10 epochs
        save      = True,
        plots     = True,
        val       = True,

        # Augmentation
        **AUGMENTATION_CONFIG,
    )

    # Copy best weights to models/ root for easy access
    best_weights = f"models/{cfg['out_name']}/weights/best.pt"
    dest = f"models/{cfg['out_name']}.pt"
    if os.path.exists(best_weights):
        import shutil
        shutil.copy(best_weights, dest)
        print(f"\n  ✓ Best weights saved to: {dest}")

    # Print final metrics
    print(f"\n  ── Final Metrics ─────────────────────────────────")
    print(f"  mAP50   : {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print(f"  mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
    print(f"  Precision: {results.results_dict.get('metrics/precision(B)', 'N/A'):.4f}")
    print(f"  Recall   : {results.results_dict.get('metrics/recall(B)', 'N/A'):.4f}")
    print(f"  ─────────────────────────────────────────────────\n")

    return results


def validate(task: str):
    """Run validation on the trained model to get final accuracy metrics."""
    cfg = TRAIN_CONFIG[task]
    weights_path = f"models/{cfg['out_name']}.pt"

    if not os.path.exists(weights_path):
        print(f"  ✗ No trained weights found at {weights_path}. Train first.")
        return

    model = YOLO(weights_path)
    metrics = model.val(data=cfg["data_yaml"])

    print(f"\n  ── Validation Results: {task.upper()} ───────────────────")
    print(f"  mAP50    : {metrics.box.map50:.4f}")
    print(f"  mAP50-95 : {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.p.mean():.4f}")
    print(f"  Recall   : {metrics.box.r.mean():.4f}")

    # Check if we hit the 85% threshold
    accuracy = metrics.box.map50 * 100
    if accuracy >= 85:
        print(f"\n  ✅ PASSES 85% threshold! ({accuracy:.1f}%)")
    else:
        print(f"\n  ⚠ Below 85% threshold ({accuracy:.1f}%) — try more epochs or more data.")

    return metrics


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CityLens Group 3 — Train violation detector")
    parser.add_argument("--task",     required=True, choices=list(TRAIN_CONFIG.keys()),
                        help="Which detector to train")
    parser.add_argument("--validate", action="store_true",
                        help="Run validation on existing trained model")
    parser.add_argument("--resume",   action="store_true",
                        help="Resume interrupted training")
    args = parser.parse_args()

    if args.validate:
        validate(args.task)
    else:
        train(args.task, resume=args.resume)
