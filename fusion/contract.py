"""
fusion/contract.py

Data Contract & Alignment Validator for the 3 Branches:
1. Spatial Branch (SigLIP / ViT)
2. Frequency Branch (FFT / DCT)
3. rPPG Branch (POS / MSTmap / TransRPPG)

This module validates that each branch's exported predictions conform
to the common schema, merges them by video_id, and checks for missing or misaligned samples.
"""

import os
import json
import pandas as pd
from typing import Dict, List, Any, Optional


REQUIRED_COLUMNS = ["video_id", "score"]


def validate_branch_file(file_path: str, expected_branch: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Validates that a branch output file (JSON or CSV) contains valid entries.
    Schema expected:
      - JSON format:
        {
          "branch": "spatial",
          "predictions": [
             {"video_id": "...", "score": 0.85, "feature_vector": [...]},
             ...
          ]
        }
      - OR CSV format:
        video_id,score[,label]
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Branch prediction file not found: {file_path}")

    records = []
    if file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            branch_name = data.get("branch", "unknown")
            if expected_branch and branch_name != expected_branch:
                print(f"[WARN] Expected branch '{expected_branch}', found '{branch_name}' in {file_path}")
            records = data.get("predictions", [])
        elif isinstance(data, list):
            records = data
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        records = df.to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported file format: {file_path}. Use .json or .csv.")

    # Validate each entry
    for idx, item in enumerate(records):
        for col in REQUIRED_COLUMNS:
            if col not in item:
                raise ValueError(f"Item #{idx} missing required field '{col}' in {file_path}")
        # Ensure score is numeric and bounded [0, 1]
        score = float(item["score"])
        if not (0.0 <= score <= 1.0):
            print(f"[WARN] Score outside [0, 1] for video_id '{item['video_id']}': {score}. Clamping.")
            item["score"] = max(0.0, min(1.0, score))

    print(f"[OK] Validated {len(records)} entries in {file_path}")
    return records


def build_aligned_fusion_table(
    master_manifest_path: str,
    spatial_preds_path: str,
    frequency_preds_path: str,
    rppg_preds_path: str,
    splits_path: Optional[str] = None,
    output_csv: Optional[str] = None
) -> pd.DataFrame:
    """
    Merges predictions from all 3 branches with ground-truth labels and metadata.
    Produces an aligned DataFrame:
    [video_id, split, label, category, generator, spatial_score, frequency_score, rppg_score, lighting, head_movement]
    """
    with open(master_manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest_map = {item["video_id"]: item for item in manifest}

    # Load splits if provided
    split_assignment = {}
    if splits_path and os.path.exists(splits_path):
        with open(splits_path, "r", encoding="utf-8") as f:
            splits_data = json.load(f)
        for s_name in ["train", "val", "test"]:
            for vid in splits_data.get(f"{s_name}_video_ids", []):
                split_assignment[vid] = s_name

    # Load branch predictions
    spatial_preds = {x["video_id"]: x for x in validate_branch_file(spatial_preds_path, "spatial")}
    frequency_preds = {x["video_id"]: x for x in validate_branch_file(frequency_preds_path, "frequency")}
    rppg_preds = {x["video_id"]: x for x in validate_branch_file(rppg_preds_path, "rppg")}

    # Match all videos
    aligned_rows = []
    missing_counts = {"spatial": 0, "frequency": 0, "rppg": 0}

    for vid, item in manifest_map.items():
        sp = spatial_preds.get(vid)
        fq = frequency_preds.get(vid)
        rp = rppg_preds.get(vid)

        if not sp:
            missing_counts["spatial"] += 1
        if not fq:
            missing_counts["frequency"] += 1
        if not rp:
            missing_counts["rppg"] += 1

        meta = item.get("metadata", {})
        row = {
            "video_id": vid,
            "split": split_assignment.get(vid, "unassigned"),
            "label": item["label"],
            "category": item["category"],
            "generator": item["generator"],
            "spatial_score": sp["score"] if sp else None,
            "frequency_score": fq["score"] if fq else None,
            "rppg_score": rp["score"] if rp else None,
            "lighting": meta.get("lighting"),
            "head_movement": meta.get("head_movement"),
            "skin_color": meta.get("skin_color")
        }
        aligned_rows.append(row)

    df = pd.DataFrame(aligned_rows)
    print("\n" + "=" * 50)
    print(" Aligned Fusion Table Summary")
    print("=" * 50)
    print(f"Total Videos: {len(df)}")
    print(f"Missing Spatial Predictions:   {missing_counts['spatial']}")
    print(f"Missing Frequency Predictions: {missing_counts['frequency']}")
    print(f"Missing rPPG Predictions:      {missing_counts['rppg']}")
    print("=" * 50 + "\n")

    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"[SUCCESS] Aligned fusion table saved to {output_csv}")

    return df
