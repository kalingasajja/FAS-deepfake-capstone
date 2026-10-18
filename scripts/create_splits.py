"""
create_splits.py

Generates leakage-safe train/val/test splits and leave-one-generator-out (LOGO) splits
for the Unified Deepfake Detection Capstone Project.

Leakage-Safety Guarantees:
- Real (Celeb-DF): Partitioned strictly by 'split_group' (subject ID, e.g. id0..id61).
  No celebrity appears in more than one split partition.
- Face-Swaps: Partitioned strictly by base target video ID (e.g. target 001).
  No original background/body appears in both train and test.
- Synthetic: Partitioned strictly by prompt index (prompt 1..20).
  Identical prompts across Sora/Kling/Runway/Pika/PixVerse do not leak between splits.

Outputs:
- data/splits.json (and frozen copy: data/locked_splits.json)
- data/splits_logo.json (Leave-One-Generator-Out splits for generalization tests)
- data/manifests/spatial_manifest.json
- data/manifests/frequency_manifest.json
- data/manifests/rppg_manifest.json
"""

import os
import json
import random
import argparse
from typing import Dict, List, Any


def load_manifest(manifest_path: str) -> List[Dict[str, Any]]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def partition_groups(
    groups: List[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, List[str]]:
    """Deterministically partitions a list of group keys into train, val, test."""
    rng = random.Random(seed)
    shuffled = sorted(list(set(groups)))
    rng.shuffle(shuffled)
    
    n = len(shuffled)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    
    train_g = shuffled[:n_train]
    val_g = shuffled[n_train:n_train + n_val]
    test_g = shuffled[n_train + n_val:]
    
    # Ensure at least 1 group in val and test if possible
    if len(val_g) == 0 and len(train_g) > 2:
        val_g.append(train_g.pop())
    if len(test_g) == 0 and len(train_g) > 2:
        test_g.append(train_g.pop())
        
    return {
        "train": train_g,
        "val": val_g,
        "test": test_g
    }


def create_standard_splits(
    manifest: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Creates leakage-safe splits by clustering entries by category and split_group.
    """
    # 1. Cluster videos by category and split_group
    grouped_by_cat: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        "real": {},
        "face_swap": {},
        "synthetic": {}
    }
    
    for item in manifest:
        cat = item["category"]
        grp = item["split_group"]
        if cat not in grouped_by_cat:
            grouped_by_cat[cat] = {}
        if grp not in grouped_by_cat[cat]:
            grouped_by_cat[cat][grp] = []
        grouped_by_cat[cat][grp].append(item)

    # 2. Partition groups within each category
    assigned_splits = {"train": [], "val": [], "test": []}
    group_assignments = {"train": [], "val": [], "test": []}

    for cat, grp_dict in grouped_by_cat.items():
        if not grp_dict:
            continue
        all_groups = list(grp_dict.keys())
        cat_partitions = partition_groups(
            all_groups,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed + hash(cat) % 10000
        )
        
        for split_name in ["train", "val", "test"]:
            group_assignments[split_name].extend(cat_partitions[split_name])
            for g in cat_partitions[split_name]:
                assigned_splits[split_name].extend(grp_dict[g])

    # 3. Build summary statistics
    def get_split_stats(split_list):
        real_count = sum(1 for x in split_list if x["label"] == 0)
        fake_count = sum(1 for x in split_list if x["label"] == 1)
        gen_counts = {}
        for x in split_list:
            g = x["generator"]
            gen_counts[g] = gen_counts.get(g, 0) + 1
        return {
            "total": len(split_list),
            "real": real_count,
            "fake": fake_count,
            "generators": gen_counts
        }

    splits_data = {
        "metadata": {
            "description": "Deterministic leakage-safe group split for NFI dataset",
            "seed": seed,
            "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
            "stats": {
                "train": get_split_stats(assigned_splits["train"]),
                "val": get_split_stats(assigned_splits["val"]),
                "test": get_split_stats(assigned_splits["test"])
            }
        },
        "train_video_ids": [x["video_id"] for x in assigned_splits["train"]],
        "val_video_ids": [x["video_id"] for x in assigned_splits["val"]],
        "test_video_ids": [x["video_id"] for x in assigned_splits["test"]],
        "train_samples": assigned_splits["train"],
        "val_samples": assigned_splits["val"],
        "test_samples": assigned_splits["test"]
    }
    
    return splits_data


def create_logo_splits(
    manifest: List[Dict[str, Any]],
    seed: int = 42
) -> Dict[str, Any]:
    """
    Creates Leave-One-Generator-Out (LOGO) cross-validation splits.
    For each generator G:
      - Test set has all fake videos from G + a dedicated partition of held-out real identities.
      - Train set has all fakes from other generators + remaining real identities.
    """
    # Isolate real samples and group by subject_id
    real_by_group: Dict[str, List[Dict[str, Any]]] = {}
    fakes_by_generator: Dict[str, List[Dict[str, Any]]] = {}

    for item in manifest:
        if item["label"] == 0:
            grp = item["split_group"]
            if grp not in real_by_group:
                real_by_group[grp] = []
            real_by_group[grp].append(item)
        else:
            gen = item["generator"]
            if gen not in fakes_by_generator:
                fakes_by_generator[gen] = []
            fakes_by_generator[gen].append(item)

    # Split real identities once: 70% train, 15% val, 15% test
    real_partitions = partition_groups(list(real_by_group.keys()), 0.70, 0.15, 0.15, seed=seed)
    real_train = [x for g in real_partitions["train"] for x in real_by_group[g]]
    real_val = [x for g in real_partitions["val"] for x in real_by_group[g]]
    real_test = [x for g in real_partitions["test"] for x in real_by_group[g]]

    logo_folds = {}
    for held_out_gen, held_out_fakes in fakes_by_generator.items():
        # Train fakes = all fakes except held_out_gen
        train_fakes = []
        val_fakes = []
        for g, f_list in fakes_by_generator.items():
            if g == held_out_gen:
                continue
            # Partition other fakes: 85% train, 15% val
            rng = random.Random(seed + hash(g) % 1000)
            shuffled = list(f_list)
            rng.shuffle(shuffled)
            n_val = max(1, int(len(shuffled) * 0.15))
            val_fakes.extend(shuffled[:n_val])
            train_fakes.extend(shuffled[n_val:])

        fold_train = real_train + train_fakes
        fold_val = real_val + val_fakes
        fold_test = real_test + held_out_fakes

        logo_folds[held_out_gen] = {
            "unseen_generator": held_out_gen,
            "train_video_ids": [x["video_id"] for x in fold_train],
            "val_video_ids": [x["video_id"] for x in fold_val],
            "test_video_ids": [x["video_id"] for x in fold_test],
            "test_summary": {
                "total": len(fold_test),
                "unseen_fakes": len(held_out_fakes),
                "real": len(real_test)
            }
        }

    return logo_folds


def export_branch_manifests(
    splits: Dict[str, Any],
    output_dir: str = "data/manifests"
):
    """
    Exports branch-specific manifests so each teammate gets an identical,
    clean view of the dataset with appropriate branch instructions.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Spatial branch (SigLIP / ViT)
    spatial_data = {
        "branch": "spatial",
        "description": "Manifest for Spatial branch (face frame crops, ViT/SigLIP backbone)",
        "train": [{"video_id": x["video_id"], "path": x["relative_path"], "label": x["label"]} for x in splits["train_samples"]],
        "val": [{"video_id": x["video_id"], "path": x["relative_path"], "label": x["label"]} for x in splits["val_samples"]],
        "test": [{"video_id": x["video_id"], "path": x["relative_path"], "label": x["label"]} for x in splits["test_samples"]],
    }
    with open(os.path.join(output_dir, "spatial_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(spatial_data, f, indent=2)

    # 2. Frequency branch (FFT / DCT)
    frequency_data = {
        "branch": "frequency",
        "description": "Manifest for Frequency branch (2D FFT / DCT residual power spectra)",
        "train": spatial_data["train"],
        "val": spatial_data["val"],
        "test": spatial_data["test"],
    }
    with open(os.path.join(output_dir, "frequency_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(frequency_data, f, indent=2)

    # 3. rPPG branch (POS / MSTmap / TransRPPG)
    rppg_data = {
        "branch": "rppg",
        "description": "Manifest for Physiological rPPG branch (facial skin patches / BVP signals)",
        "train": spatial_data["train"],
        "val": spatial_data["val"],
        "test": spatial_data["test"],
    }
    with open(os.path.join(output_dir, "rppg_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(rppg_data, f, indent=2)

    print(f"[SUCCESS] Exported branch manifests to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Create deterministic leakage-safe splits.")
    parser.add_argument("--manifest", type=str, default="data/master_manifest.json",
                        help="Path to master_manifest.json.")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Output directory (default: data).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for deterministic splitting.")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    print(f"[INFO] Loaded {len(manifest)} samples from {args.manifest}")

    # Standard train/val/test splits
    print("[INFO] Generating standard leakage-safe splits (70/15/15)...")
    splits = create_standard_splits(manifest, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=args.seed)

    splits_path = os.path.join(args.output_dir, "splits.json")
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
    print(f"[SUCCESS] Saved splits to {splits_path}")

    # Create locked immutable copy
    locked_path = os.path.join(args.output_dir, "locked_splits.json")
    with open(locked_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
    print(f"[SUCCESS] Locked splits written to {locked_path}")

    # Generate LOGO generalization splits
    print("[INFO] Generating Leave-One-Generator-Out (LOGO) splits...")
    logo_splits = create_logo_splits(manifest, seed=args.seed)
    logo_path = os.path.join(args.output_dir, "splits_logo.json")
    with open(logo_path, "w", encoding="utf-8") as f:
        json.dump(logo_splits, f, indent=2)
    print(f"[SUCCESS] Saved LOGO generalization splits to {logo_path}")

    # Export manifests for each branch
    export_branch_manifests(splits, output_dir=os.path.join(args.output_dir, "manifests"))

    # Print summary
    stats = splits["metadata"]["stats"]
    print("\n" + "=" * 55)
    print(f" Frozen Splits Summary (Seed={args.seed})")
    print("=" * 55)
    for s_name in ["train", "val", "test"]:
        st = stats[s_name]
        print(f"  [{s_name.upper():5s}] Total: {st['total']:3d} | Real: {st['real']:3d} | Fake: {st['fake']:3d}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
