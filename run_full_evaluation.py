import sys
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import time
import tracemalloc
import psutil

from src import preprocessing, segmentation, morphology, detection, evaluation, profiling, degradation

DATASET_DIR = Path("dataset")
METHOD_KEYS = ("global", "otsu", "adaptive", "edge", "hybrid")

import csv

gt_dir = DATASET_DIR / "ground_truth"
with open(gt_dir / "counts.csv", newline="") as f:
    rows = list(csv.DictReader(f))

entries = []
for row in rows:
    name = row["filename"]
    img = cv2.imread(str(DATASET_DIR / "images" / name), cv2.IMREAD_GRAYSCALE)
    gt_mask = cv2.imread(str(gt_dir / name), cv2.IMREAD_GRAYSCALE)
    inst = cv2.imread(str(gt_dir / "instances" / name), cv2.IMREAD_UNCHANGED)
    gt_objects = evaluation.gt_objects_from_instances(inst)
    
    # Calculate gt_blobs
    n_blobs, _, _, _ = cv2.connectedComponentsWithStats(
        morphology._to_mask(gt_mask), connectivity=8
    )
    gt_blobs = max(0, n_blobs - 1)
    
    entries.append({
        "filename": name,
        "condition": row["condition"],
        "polarity": row["polarity"],
        "n_objects": int(row["n_objects"]),
        "gt_blobs": gt_blobs,
        "image": img,
        "gt_mask": gt_mask,
        "gt_objects": gt_objects,
    })

def method_params(polarity):
    return {
        "global": {"threshold": 127, "object_polarity": polarity},
        "otsu": {"object_polarity": polarity},
        "adaptive": {"block_size": 11, "C": 2, "adaptive_method": "gaussian",
                     "object_polarity": polarity},
        "edge": {"low_threshold": 50, "high_threshold": 150, "closing_kernel_size": 5},
        "hybrid": {"object_polarity": polarity, "eta_trust": 0.5, "block_size": 21, "C": 8, "edge_band": 5}
    }

results = []
hybrid_eta_data = []

for entry in entries:
    gray = entry["image"]
    gt = {"objects": entry["gt_objects"], "count": entry["n_objects"], "mask": entry["gt_mask"]}
    params = method_params(entry["polarity"])
    
    row_per_method = {}
    for key in METHOD_KEYS:
        seg = segmentation.segment(gray, key, **params[key])
        cleaned = morphology.clean_mask(
            seg["mask"], open_kernel_size=3, close_kernel_size=5, min_area=50
        )
        objects = detection.detect_objects(cleaned["final"], min_area=50)
        metrics = evaluation.evaluate_method(
            {"objects": objects, "mask": cleaned["final"],
             "processing_time_ms": seg["processing_time_ms"]},
            gt, iou_threshold=0.5
        )
        res_dict = {
            "filename": entry["filename"],
            "condition": entry["condition"],
            "method": key,
            "raw_mask_iou": evaluation.calculate_iou(seg["mask"], entry["gt_mask"]),
            **metrics,
        }
        results.append(res_dict)
        row_per_method[key] = res_dict
        
        if key == "hybrid":
            eta = seg["intermediate"]["eta"]
            best_classical = max(
                row_per_method["global"]["mask_iou"],
                row_per_method["otsu"]["mask_iou"],
                row_per_method["adaptive"]["mask_iou"],
                row_per_method["edge"]["mask_iou"]
            )
            hybrid_eta_data.append({
                "filename": entry["filename"],
                "condition": entry["condition"],
                "eta": eta,
                "branch": "high_eta" if eta >= 0.5 else "low_eta",
                "hybrid_iou": res_dict["mask_iou"],
                "best_classical_iou": best_classical,
                "diff": res_dict["mask_iou"] - best_classical
            })

df = pd.DataFrame(results)
df_eta = pd.DataFrame(hybrid_eta_data)

print("=== TABLE VII: OVERALL COMPARISON OVER 20 IMAGES ===")
overall = df.groupby("method").agg({
    "mask_iou": "mean",
    "mean_iou": "mean",
    "count_accuracy": "mean",
    "mean_measurement_error_pct": "mean",
    "false_positives": "mean",
    "missed": "mean",
    "processing_time_ms": "mean"
}).loc[["global", "otsu", "adaptive", "edge", "hybrid"]]
print(overall.round(4).to_string())

print("\n=== TABLE VIII: BY CONDITION (Mask IoU and Count Accuracy) ===")
cond_summary = df.pivot_table(
    index="method", columns="condition", values=["mask_iou", "count_accuracy"]
).loc[["global", "otsu", "adaptive", "edge", "hybrid"]][
    [("mask_iou", "normal"), ("count_accuracy", "normal"),
     ("mask_iou", "noisy"), ("count_accuracy", "noisy"),
     ("mask_iou", "uneven"), ("count_accuracy", "uneven"),
     ("mask_iou", "touching"), ("count_accuracy", "touching")]
].round(4)
print(cond_summary.to_string())

print("\n=== TABLE X: SEPARABILITY SCORE AND HYBRID BRANCH ===")
eta_summary = df_eta.groupby("condition").agg(
    mean_eta=("eta", "mean"),
    std_eta=("eta", "std"),
    fraction_high_eta=("eta", lambda s: (s >= 0.5).mean()),
    mean_diff=("diff", "mean")
)
print(eta_summary.round(4).to_string())

print("\n=== HARDWARE PROFILING (10 repeats on normal_01.png) ===")
test_img = entries[0]["image"]
hw_results = []
for m_name in METHOD_KEYS:
    m_func = lambda im, k=m_name: segmentation.segment(im, k, **method_params("darker")[k])
    prof = profiling.profile_algorithm(m_func, test_img, profile_repeats=10)
    hw_results.append({
        "method": m_name,
        "median_time_ms": prof["execution_time_ms"],
        "peak_ram_mb": prof["peak_memory_mb"],
        "ram_increase_mb": prof["memory_delta_mb"],
        "alloc_peak_mb": prof["alloc_peak_mb"]
    })
df_hw = pd.DataFrame(hw_results).set_index("method")
print(df_hw.round(3).to_string())

# Stress test cliffs
print("\n=== STRESS TEST PERFORMANCE CLIFFS ===")
# Base image normal_01
base_entry = entries[0]
base_img = base_entry["image"]
base_gt = base_entry["gt_mask"]

# 1. Noise sweep
noise_levels = [0, 100, 400, 900, 1600]
print("Noise Variance vs Mask IoU:")
for lvl in noise_levels:
    deg = degradation.inject_gaussian_noise(base_img, lvl, seed=42)
    ious = {}
    for k in METHOD_KEYS:
        seg = segmentation.segment(deg, k, **method_params("darker")[k])
        cln = morphology.clean_mask(seg["mask"], open_kernel_size=3, close_kernel_size=5, min_area=50)
        ious[k] = round(evaluation.calculate_iou(cln["final"], base_gt), 3)
    print(f"  Var {lvl:4d}: {ious}")

# 2. Gamma sweep
gamma_levels = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
print("\nGamma vs Mask IoU:")
for lvl in gamma_levels:
    deg = degradation.inject_low_light(base_img, lvl)
    ious = {}
    for k in METHOD_KEYS:
        seg = segmentation.segment(deg, k, **method_params("darker")[k])
        cln = morphology.clean_mask(seg["mask"], open_kernel_size=3, close_kernel_size=5, min_area=50)
        ious[k] = round(evaluation.calculate_iou(cln["final"], base_gt), 3)
    print(f"  Gamma {lvl:.1f}: {ious}")

# 3. Blur sweep
blur_levels = [1, 5, 11, 15, 21, 25, 31]
print("\nBlur length vs Mask IoU:")
for lvl in blur_levels:
    deg = degradation.inject_motion_blur(base_img, lvl, angle=45.0)
    ious = {}
    for k in METHOD_KEYS:
        seg = segmentation.segment(deg, k, **method_params("darker")[k])
        cln = morphology.clean_mask(seg["mask"], open_kernel_size=3, close_kernel_size=5, min_area=50)
        ious[k] = round(evaluation.calculate_iou(cln["final"], base_gt), 3)
    print(f"  Blur {lvl:2d}px: {ious}")
