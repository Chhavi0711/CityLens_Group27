"""
CityLens AI Hackathon 2026 — Group 3
Dataset download script. Downloads open-access datasets from Roboflow Universe.

BEFORE RUNNING:
  1. Create a free account at https://app.roboflow.com
  2. Get your API key from: https://app.roboflow.com/settings/api
  3. Set it as an environment variable:
       export ROBOFLOW_API_KEY="your_key_here"
     OR pass it with --api-key flag.

Usage:
    python download_data.py --task helmet
    python download_data.py --task vehicle
    python download_data.py --task signal
    python download_data.py --task seatbelt
    python download_data.py --task all
"""

import argparse
import os
import sys

# ─── ROBOFLOW DATASET REGISTRY ───────────────────────────────────────────────
# Each entry maps to a free, open-access Roboflow Universe dataset.
# These are curated for Indian traffic conditions where available.

DATASETS = {
    "helmet": {
        "workspace":  "traffic-violation-ed8ag",
        "project":    "bike-helmet-detection-2vdjo-nkdyo",
        "version":    1,
        "format":     "yolov8",
        "location":   "data/helmet",
        "classes":    ["With-Helmet", "Without-Helmet"],
        "description": "1,371 images of riders with/without helmets",
        "fallback_url": "https://universe.roboflow.com/traffic-violation-ed8ag/bike-helmet-detection-2vdjo-nkdyo"
    },
    "vehicle": {
        "workspace":  "indian-road-vehicles",
        "project":    "vehicle-detection-akiss",
        "version":    2,
        "format":     "yolov8",
        "location":   "data/vehicle",
        "classes":    ["car", "motorcycle", "auto", "truck", "bus"],
        "description": "Indian road vehicle detection dataset",
        "fallback_url": "https://universe.roboflow.com/search?q=indian+traffic+vehicle+detection"
    },
    "signal": {
        "workspace":  "self-driving-car-s8xyq",
        "project":    "traffic-light-detection-twdgt",
        "version":    2,
        "format":     "yolov8",
        "location":   "data/signal",
        "classes":    ["red", "yellow", "green"],
        "description": "Traffic light state detection dataset",
        "fallback_url": "https://universe.roboflow.com/search?q=traffic+light+detection+red+green+yellow"
    },
    "seatbelt": {
        "workspace":  "seatbelt-detection",
        "project":    "seatbelt-detection-mrmwn",
        "version":    1,
        "format":     "yolov8",
        "location":   "data/seatbelt",
        "classes":    ["seatbelt", "no-seatbelt"],
        "description": "Driver seatbelt compliance detection",
        "fallback_url": "https://universe.roboflow.com/search?q=seatbelt+detection"
    },
}

# ─────────────────────────────────────────────────────────────────────────────


def download_dataset(task: str, api_key: str):
    """Download a Roboflow dataset for a given task."""
    from roboflow import Roboflow

    if task not in DATASETS:
        print(f"  ✗ Unknown task: {task}")
        return False

    ds = DATASETS[task]
    os.makedirs(ds["location"], exist_ok=True)

    print(f"\n  Downloading: {task.upper()} dataset")
    print(f"  Description: {ds['description']}")
    print(f"  Classes    : {ds['classes']}")
    print(f"  Destination: {ds['location']}/\n")

    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(ds["workspace"]).project(ds["project"])
        version = project.version(ds["version"])
        dataset = version.download(ds["format"], location=ds["location"])
        print(f"\n  ✓ Downloaded to: {ds['location']}/")
        print(f"  ✓ data.yaml at : {ds['location']}/data.yaml")
        return True

    except Exception as e:
        print(f"\n  ✗ Download failed: {e}")
        print(f"\n  → Manual fallback:")
        print(f"    1. Go to: {ds['fallback_url']}")
        print(f"    2. Click 'Download Dataset' → YOLOv8 format")
        print(f"    3. Extract into: {ds['location']}/")
        print(f"    4. Make sure data.yaml is at: {ds['location']}/data.yaml\n")
        return False


def create_manual_yaml(task: str):
    """
    Create a minimal data.yaml if you're using your own manually downloaded data.
    Edit the paths after running this.
    """
    ds = DATASETS[task]
    yaml_content = f"""# CityLens Group 3 — {task.upper()} detection
# Edit train/val paths to point to your actual dataset folders

path: {ds['location']}  # root dataset dir
train: images/train     # train images (relative to path)
val:   images/val       # val images (relative to path)
test:  images/test      # test images (optional)

# Classes
nc: {len(ds['classes'])}
names: {ds['classes']}
"""
    yaml_path = os.path.join(ds["location"], "data.yaml")
    os.makedirs(ds["location"], exist_ok=True)
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"  ✓ Created template data.yaml at: {yaml_path}")
    print(f"    → Edit train/val paths before training.\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CityLens Group 3 — Download Roboflow datasets")
    parser.add_argument("--task",    required=True,
                        choices=list(DATASETS.keys()) + ["all"],
                        help="Which dataset to download")
    parser.add_argument("--api-key", default=None,
                        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)")
    parser.add_argument("--create-yaml", action="store_true",
                        help="Just create a template data.yaml (no download)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY")

    if args.create_yaml:
        tasks = list(DATASETS.keys()) if args.task == "all" else [args.task]
        for t in tasks:
            create_manual_yaml(t)
        sys.exit(0)

    if not api_key:
        print("\n  ✗ No Roboflow API key provided.")
        print("  → Get your key at: https://app.roboflow.com/settings/api")
        print("  → Then run: export ROBOFLOW_API_KEY='your_key_here'\n")
        sys.exit(1)

    tasks = list(DATASETS.keys()) if args.task == "all" else [args.task]
    for task in tasks:
        success = download_dataset(task, api_key)
        if not success:
            print(f"  Skipping {task} — see manual fallback instructions above.\n")
