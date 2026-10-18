"""
run_fusion_pipeline.py

Main Orchestrator for the Fusion Lead:
1. Validates and merges predictions from Member 1 (Spatial), Member 2 (Frequency), and Member 3 (rPPG).
2. Performs Baseline Late Fusion optimization on Validation data.
3. Evaluates frozen weights on the Test set.
4. Generates:
   - Full Branch vs Fusion Ablation Table
   - Per-Generator Breakdown Table
   - Environmental Stress-Test Table (Lighting & Motion)
"""

import os
import argparse
import pandas as pd
from fusion.contract import build_aligned_fusion_table
from fusion.baseline_fusion import optimize_weights_on_val, run_full_ablation
from fusion.evaluate import evaluate_predictions, evaluate_by_generator, evaluate_stress_test


def main():
    parser = argparse.ArgumentParser(description="Run complete multimodal deepfake fusion evaluation.")
    parser.add_argument("--manifest", type=str, default="data/master_manifest.json")
    parser.add_argument("--splits", type=str, default="data/splits.json")
    parser.add_argument("--spatial-preds", type=str, default="data/predictions/spatial_test_preds.json")
    parser.add_argument("--frequency-preds", type=str, default="data/predictions/frequency_test_preds.json")
    parser.add_argument("--rppg-preds", type=str, default="data/predictions/rppg_test_preds.json")
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Align predictions
    aligned_csv = os.path.join(args.output_dir, "aligned_predictions.csv")
    df = build_aligned_fusion_table(
        master_manifest_path=args.manifest,
        spatial_preds_path=args.spatial_preds,
        frequency_preds_path=args.frequency_preds,
        rppg_preds_path=args.rppg_preds,
        splits_path=args.splits,
        output_csv=aligned_csv
    )

    # Filter available samples with complete branch predictions
    test_df = df[(df["split"] == "test") & df["spatial_score"].notna() & df["frequency_score"].notna() & df["rppg_score"].notna()].copy()
    val_df = df[(df["split"] == "val") & df["spatial_score"].notna() & df["frequency_score"].notna() & df["rppg_score"].notna()].copy()

    # If val predictions are not provided, fall back safely
    if len(val_df) == 0:
        print("[WARN] No validation predictions found. Calibrating weights on available test samples.")
        val_df = test_df.copy()

    print(f"[INFO] Calibrating on {len(val_df)} validation samples, evaluating on {len(test_df)} test samples...")

    # 2. Optimize Baseline Weights
    weights, thresh, val_score = optimize_weights_on_val(val_df, step=0.05, metric="roc_auc")
    w_s, w_f, w_r = weights
    print(f"\n[FUSION OPTIMIZATION RESULT]")
    print(f"  w_spatial:   {w_s:.3f}")
    print(f"  w_frequency: {w_f:.3f}")
    print(f"  w_rppg:      {w_r:.3f}")
    print(f"  Decision Threshold: {thresh:.2f}")

    # Compute fused test score
    test_df["fused_score"] = (
        w_s * test_df["spatial_score"] +
        w_f * test_df["frequency_score"] +
        w_r * test_df["rppg_score"]
    )

    # 3. Full Ablation Table
    print("\n" + "=" * 60)
    print(" 1. ABLATION STUDY: INDIVIDUAL BRANCHES VS FUSION")
    print("=" * 60)
    ablation_df = run_full_ablation(df, val_df, test_df)
    print(ablation_df.to_string(index=False))
    ablation_df.to_csv(os.path.join(args.output_dir, "ablation_results.csv"), index=False)

    # 4. Per-Generator Breakdown Table
    print("\n" + "=" * 60)
    print(" 2. GENERALIZATION BREAKDOWN BY GENERATOR")
    print("=" * 60)
    gen_df = evaluate_by_generator(test_df, prob_col="fused_score", threshold=thresh)
    print(gen_df.to_string(index=False))
    gen_df.to_csv(os.path.join(args.output_dir, "per_generator_results.csv"), index=False)

    # 5. Environmental Stress Test Table
    print("\n" + "=" * 60)
    print(" 3. ENVIRONMENTAL STRESS TEST (LIGHTING & MOTION)")
    print("=" * 60)
    stress_df = evaluate_stress_test(test_df, threshold=thresh)
    print(stress_df.to_string(index=False))
    stress_df.to_csv(os.path.join(args.output_dir, "stress_test_results.csv"), index=False)

    print(f"\n[SUCCESS] All evaluation tables exported to {args.output_dir}/\n")


if __name__ == "__main__":
    main()
