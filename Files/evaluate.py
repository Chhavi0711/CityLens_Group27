"""
CityLens AI Hackathon 2026 — Group 3
Evaluation script. Generates a full accuracy report across all models.
Run this before submission to verify you meet the 85% threshold.

Usage:
    python evaluate.py
    python evaluate.py --task helmet
"""

import argparse
import os
import json
from datetime import datetime
from ultralytics import YOLO

MODELS = {
    "helmet":  {"weights": "models/helmet_detector.pt",  "data": "data/helmet/data.yaml"},
    "vehicle": {"weights": "models/vehicle_detector.pt", "data": "data/vehicle/data.yaml"},
    "signal":  {"weights": "models/signal_detector.pt",  "data": "data/signal/data.yaml"},
    "seatbelt":{"weights": "models/seatbelt_detector.pt","data": "data/seatbelt/data.yaml"},
}

CATEGORY_WEIGHTS = {
    "signal":   0.35,   # signal jump — high priority
    "vehicle":  0.30,   # base detection — everything depends on this
    "helmet":   0.25,   # no helmet — common violation
    "seatbelt": 0.10,   # seatbelt
}


def evaluate_model(task: str) -> dict:
    """Run validation on a trained model and return metrics dict."""
    cfg = MODELS[task]

    if not os.path.exists(cfg["weights"]):
        print(f"  ⚠ [{task}] No weights found at {cfg['weights']} — skipping.")
        return None

    if not os.path.exists(cfg["data"]):
        print(f"  ⚠ [{task}] No data.yaml found at {cfg['data']} — skipping.")
        return None

    print(f"  Evaluating {task.upper()}...", end=" ", flush=True)
    model = YOLO(cfg["weights"])
    metrics = model.val(data=cfg["data"], verbose=False)

    result = {
        "task":       task,
        "mAP50":      round(metrics.box.map50 * 100, 2),
        "mAP50_95":   round(metrics.box.map * 100, 2),
        "precision":  round(metrics.box.p.mean() * 100, 2),
        "recall":     round(metrics.box.r.mean() * 100, 2),
        "passes_threshold": metrics.box.map50 >= 0.85,
        "weights":    cfg["weights"],
        "data":       cfg["data"],
    }
    print(f"mAP50: {result['mAP50']}%  {'✅' if result['passes_threshold'] else '⚠'}")
    return result


def run_full_evaluation(tasks=None):
    """Evaluate all models and produce a weighted average score."""
    if tasks is None:
        tasks = list(MODELS.keys())

    print(f"\n{'='*60}")
    print(f"  CityLens Group 3 — Evaluation Report")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}\n")

    results = {}
    for task in tasks:
        r = evaluate_model(task)
        if r:
            results[task] = r

    if not results:
        print("  No models evaluated. Train your models first.\n")
        return

    # Weighted average score
    total_weight = sum(CATEGORY_WEIGHTS[t] for t in results)
    weighted_map50 = sum(
        results[t]["mAP50"] * CATEGORY_WEIGHTS[t]
        for t in results
    ) / total_weight

    print(f"\n{'─'*60}")
    print(f"  {'Model':<12} {'mAP50':>8} {'Precision':>12} {'Recall':>9} {'Pass?':>7}")
    print(f"{'─'*60}")
    for task, r in results.items():
        status = "✅" if r["passes_threshold"] else "❌"
        print(f"  {task:<12} {r['mAP50']:>7.1f}%  {r['precision']:>10.1f}%  {r['recall']:>8.1f}%  {status:>6}")
    print(f"{'─'*60}")
    print(f"  {'WEIGHTED AVG':<12} {weighted_map50:>7.1f}%")
    print(f"{'─'*60}\n")

    overall_pass = weighted_map50 >= 85.0
    if overall_pass:
        print(f"  ✅ OVERALL: {weighted_map50:.1f}% — Meets the 85% threshold. Prize eligible!\n")
    else:
        gap = 85.0 - weighted_map50
        print(f"  ⚠  OVERALL: {weighted_map50:.1f}% — {gap:.1f}% below threshold.")
        print(f"     Improve the weakest models first (see suggestions below).\n")
        for task, r in results.items():
            if not r["passes_threshold"]:
                print(f"     [{task}] {r['mAP50']}% — try: more epochs, more data, heavier model (yolov8s.pt)")

    # Save JSON report
    report = {
        "timestamp": datetime.now().isoformat(),
        "models": results,
        "weighted_average_mAP50": round(weighted_map50, 2),
        "passes_threshold": overall_pass,
        "category_weights": CATEGORY_WEIGHTS,
    }
    os.makedirs("output", exist_ok=True)
    report_path = "output/evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Full report saved to: {report_path}\n")

    return report


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CityLens Group 3 — Evaluate models")
    parser.add_argument("--task", default=None,
                        choices=list(MODELS.keys()),
                        help="Evaluate a single task (default: all)")
    args = parser.parse_args()

    tasks = [args.task] if args.task else None
    run_full_evaluation(tasks)
