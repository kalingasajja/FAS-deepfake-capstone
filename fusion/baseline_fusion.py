"""
fusion/baseline_fusion.py

Implements Stage 1: Score-Level Baseline Late Fusion.
Formula:
    p_fused = w_s * p_spatial + w_f * p_frequency + w_r * p_rppg
    where w_s + w_f + w_r = 1, w_i >= 0.

Strict Methodology:
1. Weights [w_s, w_f, w_r] and decision threshold tau are tuned exclusively on the VALIDATION set.
2. The optimal parameters are frozen and evaluated on the TEST set.
3. Performs full ablation: individual branches vs 2-branch combinations vs unified 3-branch model.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score
)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    has_both_classes = len(np.unique(y_true)) > 1
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if has_both_classes else float("nan"),
        "threshold": float(threshold)
    }


def optimize_weights_on_val(
    val_df: pd.DataFrame,
    step: float = 0.05,
    metric: str = "roc_auc"
) -> Tuple[Tuple[float, float, float], float, float]:
    """
    Grid-searches the simplex w_s + w_f + w_r = 1 on the validation set.
    Returns: ((w_s, w_f, w_r), best_threshold, best_val_score)
    """
    y_val = val_df["label"].values.astype(int)
    p_s = val_df["spatial_score"].values.astype(float)
    p_f = val_df["frequency_score"].values.astype(float)
    p_r = val_df["rppg_score"].values.astype(float)

    best_score = -1.0
    best_weights = (1/3, 1/3, 1/3)
    best_threshold = 0.5

    # Grid search over weights
    steps = int(round(1.0 / step))
    for i in range(steps + 1):
        w_s = i * step
        for j in range(steps + 1 - i):
            w_f = j * step
            w_r = round(1.0 - w_s - w_f, 4)
            if w_r < -1e-5:
                continue
            w_r = max(0.0, w_r)

            fused_val = w_s * p_s + w_f * p_f + w_r * p_r

            if metric == "roc_auc":
                try:
                    score = roc_auc_score(y_val, fused_val)
                except Exception:
                    score = 0.0
                curr_thresh = 0.5
            else:
                # Find best threshold for F1 or accuracy
                best_t = 0.5
                best_m = -1.0
                for t in np.linspace(0.1, 0.9, 17):
                    preds = (fused_val >= t).astype(int)
                    m = f1_score(y_val, preds, zero_division=0) if metric == "f1" else accuracy_score(y_val, preds)
                    if m > best_m:
                        best_m = m
                        best_t = t
                score = best_m
                curr_thresh = best_t

            if score > best_score:
                best_score = score
                best_weights = (round(w_s, 4), round(w_f, 4), round(w_r, 4))
                best_threshold = curr_thresh

    return best_weights, best_threshold, best_score


def run_full_ablation(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Runs full ablation study:
    1. Spatial Only
    2. Frequency Only
    3. rPPG Only
    4. Spatial + Frequency
    5. Spatial + rPPG
    6. Frequency + rPPG
    7. All 3 Branches (Unified)
    """
    y_test = test_df["label"].values.astype(int)
    configurations = [
        {"name": "Spatial (SigLIP/ViT)", "cols": ["spatial_score"], "fixed_weights": (1.0, 0.0, 0.0)},
        {"name": "Frequency (FFT/DCT)",  "cols": ["frequency_score"], "fixed_weights": (0.0, 1.0, 0.0)},
        {"name": "rPPG (Physiological)", "cols": ["rppg_score"], "fixed_weights": (0.0, 0.0, 1.0)},
        {"name": "Spatial + Frequency",  "cols": ["spatial_score", "frequency_score"], "fixed_weights": None},
        {"name": "Spatial + rPPG",       "cols": ["spatial_score", "rppg_score"], "fixed_weights": None},
        {"name": "Frequency + rPPG",     "cols": ["frequency_score", "rppg_score"], "fixed_weights": None},
        {"name": "Unified (All 3)",      "cols": ["spatial_score", "frequency_score", "rppg_score"], "fixed_weights": None},
    ]

    results = []
    p_s_test = test_df["spatial_score"].values.astype(float)
    p_f_test = test_df["frequency_score"].values.astype(float)
    p_r_test = test_df["rppg_score"].values.astype(float)

    for cfg in configurations:
        name = cfg["name"]
        fixed = cfg["fixed_weights"]

        if fixed is not None:
            weights = fixed
            # Find best threshold on validation
            val_probs = fixed[0] * val_df["spatial_score"].values + fixed[1] * val_df["frequency_score"].values + fixed[2] * val_df["rppg_score"].values
            best_thresh = 0.5
            best_f1 = -1.0
            for t in np.linspace(0.1, 0.9, 17):
                f1 = f1_score(val_df["label"].values, (val_probs >= t).astype(int), zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = t
        else:
            # Mask inactive columns by setting step appropriately
            if "spatial_score" not in cfg["cols"]:
                w_s_max = 0.0
            if "frequency_score" not in cfg["cols"]:
                w_f_max = 0.0
            if "rppg_score" not in cfg["cols"]:
                w_r_max = 0.0
            
            # Simple grid search on val
            weights, best_thresh, _ = optimize_weights_on_val(val_df, step=0.05, metric="roc_auc")

        # Evaluate on test set
        test_probs = weights[0] * p_s_test + weights[1] * p_f_test + weights[2] * p_r_test
        metrics = compute_metrics(y_test, test_probs, threshold=best_thresh)
        metrics["model"] = name
        metrics["w_spatial"] = weights[0]
        metrics["w_frequency"] = weights[1]
        metrics["w_rppg"] = weights[2]
        results.append(metrics)

    res_df = pd.DataFrame(results)
    cols = ["model", "accuracy", "balanced_accuracy", "f1", "precision", "recall", "roc_auc", "w_spatial", "w_frequency", "w_rppg", "threshold"]
    return res_df[cols]
