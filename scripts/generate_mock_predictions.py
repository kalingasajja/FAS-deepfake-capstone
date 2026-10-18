"""
scripts/generate_mock_predictions.py

Generates realistic mock predictions for the 3 branches:
- Spatial (SigLIP/ViT)
- Frequency (FFT/DCT)
- rPPG (POS/MSTmap)

Simulates realistic behavior based on paper findings:
1. Under normal conditions, all 3 branches achieve ~75-88% accuracy.
2. Under 'lighting: bad' and 'head_movement: yes', rPPG performance drops to ~50% (random guess),
   while Spatial and Frequency remain resilient (~75-80%).
3. This allows immediate testing of the fusion pipeline and verifies that the fusion module
   outperforms each single branch.
"""

import os
import json
import random
import numpy as np


def generate_mock_predictions(
    splits_path: str = "data/splits.json",
    output_dir: str = "data/predictions",
    seed: int = 42
):
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    # Load samples
    if os.path.exists(splits_path):
        with open(splits_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        samples = data.get("test_samples", [])
    else:
        # Fallback to master manifest test if splits not yet generated
        samples = []

    if not samples:
        print("[WARN] No samples found in splits.json. Run create_splits.py first.")
        return

    spatial_preds = []
    freq_preds = []
    rppg_preds = []

    for s in samples:
        vid = s["video_id"]
        label = s["label"]  # 0: real, 1: fake
        meta = s.get("metadata", {})
        lighting = meta.get("lighting")
        movement = meta.get("head_movement")

        # 1. Spatial branch (SigLIP / ViT): strong on visual artifacts
        # Base true probability: Real -> ~0.15, Fake -> ~0.85
        if label == 1:
            p_s = np.clip(np_rng.normal(0.82, 0.12), 0.01, 0.99)
        else:
            p_s = np.clip(np_rng.normal(0.18, 0.12), 0.01, 0.99)

        # 2. Frequency branch (FFT/DCT): strong on GAN/diffusion boundary noise
        if label == 1:
            p_f = np.clip(np_rng.normal(0.80, 0.14), 0.01, 0.99)
        else:
            p_f = np.clip(np_rng.normal(0.20, 0.14), 0.01, 0.99)

        # 3. rPPG branch: physiological blood-volume-pulse
        # If bad lighting or high motion, rPPG degrades significantly!
        is_degraded = (lighting == "bad") or (movement == "yes")
        if is_degraded:
            # Degraded rPPG: behaves close to random noise (mean 0.50, high variance)
            p_r = np.clip(np_rng.normal(0.50, 0.25), 0.01, 0.99)
        else:
            # Good conditions: rPPG is highly discriminative
            if label == 1:
                p_r = np.clip(np_rng.normal(0.85, 0.10), 0.01, 0.99)
            else:
                p_r = np.clip(np_rng.normal(0.15, 0.10), 0.01, 0.99)

        spatial_preds.append({"video_id": vid, "score": round(float(p_s), 4)})
        freq_preds.append({"video_id": vid, "score": round(float(p_f), 4)})
        rppg_preds.append({"video_id": vid, "score": round(float(p_r), 4)})

    # Save to disk
    with open(os.path.join(output_dir, "spatial_test_preds.json"), "w", encoding="utf-8") as f:
        json.dump({"branch": "spatial", "predictions": spatial_preds}, f, indent=2)

    with open(os.path.join(output_dir, "frequency_test_preds.json"), "w", encoding="utf-8") as f:
        json.dump({"branch": "frequency", "predictions": freq_preds}, f, indent=2)

    with open(os.path.join(output_dir, "rppg_test_preds.json"), "w", encoding="utf-8") as f:
        json.dump({"branch": "rppg", "predictions": rppg_preds}, f, indent=2)

    print(f"[SUCCESS] Exported mock test predictions for {len(samples)} samples to {output_dir}/")


if __name__ == "__main__":
    generate_mock_predictions()
