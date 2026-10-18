"""
build_dataset_manifest.py

Builds the master dataset manifest for the Netherlands Forensic Institute (NFI)
rPPG Deepfake Detection dataset (Hugging Face: NetherlandsForensicInstitute/rppg-deepfake-detection).

Features:
- Handles both online metadata querying (via Hugging Face API) and local directory scanning.
- Extracts leakage-safe group keys:
    * Real (Celeb-DF): grouped by subject identity (e.g. 'id0', 'id10')
    * Face-Swaps: grouped by target video base ID (e.g. '001', '009')
    * Synthetic (Sora, Kling, Runway, Pika, PixVerse, Luma): grouped by prompt index (e.g. 'prompt_1')
- Merges annotations.json metadata (gender, skin_color, head_movement, lighting).
- Outputs master_manifest.json with standardized schema for all 3 branches.
"""

import os
import re
import json
import argparse
import urllib.request
from typing import Dict, Any, List, Optional

HF_DATASET_NAME = "NetherlandsForensicInstitute/rppg-deepfake-detection"
HF_API_URL = f"https://huggingface.co/api/datasets/{HF_DATASET_NAME}"
HF_RAW_ANNOTATIONS_URL = f"https://huggingface.co/datasets/{HF_DATASET_NAME}/raw/main/annotations.json"


def fetch_remote_annotations() -> Dict[str, Any]:
    """Fetches annotations.json directly from the Hugging Face repository."""
    try:
        req = urllib.request.Request(
            HF_RAW_ANNOTATIONS_URL,
            headers={"User-Agent": "Deepfake-Fusion-Capstone/1.0"}
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f"[WARN] Could not fetch remote annotations.json: {e}")
        return {}


def fetch_remote_file_list() -> List[str]:
    """Queries Hugging Face API to get the list of all video files in the repository."""
    req = urllib.request.Request(
        HF_API_URL,
        headers={"User-Agent": "Deepfake-Fusion-Capstone/1.0"}
    )
    with urllib.request.urlopen(req, timeout=20) as response:
        data = json.loads(response.read().decode("utf-8"))
    
    files = []
    for item in data.get("siblings", []):
        filename = item.get("rfilename", "")
        if filename.endswith(".mp4") or filename.endswith(".avi"):
            files.append(filename)
    return files


def scan_local_file_list(dataset_dir: str) -> List[str]:
    """Scans local dataset directory for video files."""
    video_files = []
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.lower().endswith((".mp4", ".avi")):
                rel_path = os.path.relpath(os.path.join(root, f), dataset_dir)
                video_files.append(rel_path.replace("\\", "/"))
    return video_files


def extract_metadata_and_group(rel_path: str, annotations: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parses relative path to extract:
    - label (0: real, 1: fake)
    - category ('real', 'face_swap', 'synthetic')
    - generator (e.g. 'CelebDF', 'AkoolAI', 'Sora', etc.)
    - split_group (leakage-safe cluster ID)
    - metadata (lighting, head_movement, etc.)
    """
    normalized_path = rel_path.replace("\\", "/")
    parts = normalized_path.split("/")
    filename = parts[-1]
    
    # 1. Real videos (real/CelebDF/id0_0000.mp4)
    if normalized_path.startswith("real/"):
        category = "real"
        generator = parts[1] if len(parts) > 2 else "CelebDF"
        label = 0
        label_name = "real"
        
        # Extract Celeb-DF subject ID (e.g. 'id0' from 'id0_0000.mp4')
        match = re.match(r"^(id\d+)_\d+", filename)
        if match:
            split_group = f"real_subject_{match.group(1)}"
        else:
            split_group = f"real_{os.path.splitext(filename)[0]}"
            
        video_metadata = {
            "gender": None,
            "skin_color": None,
            "head_movement": None,
            "lighting": None
        }

    # 2. Fake videos (fake/AkoolAI_FS/001_AkoolAI_fa021.mp4 or fake/Sora_SYN/1.mp4)
    elif normalized_path.startswith("fake/"):
        label = 1
        label_name = "fake"
        subfolder = parts[1] if len(parts) > 2 else "unknown"
        
        # Face-swap category (e.g. AkoolAI_FS)
        if "_FS" in subfolder:
            category = "face_swap"
            generator = subfolder.replace("_FS", "")
            
            # Format: <target_id>_<generator>_fa<source_id>.mp4
            # e.g., '001_AkoolAI_fa021.mp4' -> base target: '001', source face: 'fa021'
            fs_match = re.match(r"^(\d+)_([A-Za-z0-9]+)_([A-Za-z0-9]+)", filename)
            if fs_match:
                target_id = fs_match.group(1)
                source_id = fs_match.group(3)
                # Split group is based on target video base ID to prevent scene/pose leakage
                split_group = f"target_{target_id}"
            else:
                split_group = f"fs_{os.path.splitext(filename)[0]}"
                
            video_metadata = {
                "gender": None,
                "skin_color": None,
                "head_movement": None,
                "lighting": None
            }

        # Full synthetic category (e.g. Sora_SYN, KlingAI_SYN)
        elif "_SYN" in subfolder:
            category = "synthetic"
            generator = subfolder.replace("_SYN", "")
            
            # Filename is typically '1.mp4', '2.mp4', or 'deepfake1.mp4'
            num_match = re.search(r"(\d+)", filename)
            prompt_idx = num_match.group(1) if num_match else filename
            # Split group is based on prompt index since prompts 1..20 are shared across generators
            split_group = f"prompt_{prompt_idx}"
            
            # Check annotations.json for metadata
            # Annotations keys are formatted as "<Generator>,<filename>" (e.g. "Sora,1.mp4")
            ann_key = f"{generator},{filename}"
            raw_ann = annotations.get(ann_key, {})
            video_metadata = {
                "gender": raw_ann.get("gender"),
                "skin_color": raw_ann.get("skin_color"),
                "head_movement": raw_ann.get("head_movement"),
                "lighting": raw_ann.get("lighting")
            }
        else:
            category = "unknown_fake"
            generator = subfolder
            split_group = f"fake_{os.path.splitext(filename)[0]}"
            video_metadata = {}
    else:
        # Fallback
        category = "unknown"
        generator = "unknown"
        label = 0
        label_name = "unknown"
        split_group = f"misc_{os.path.splitext(filename)[0]}"
        video_metadata = {}

    clean_stem = os.path.splitext(filename)[0]
    video_id = f"{category}_{generator}_{clean_stem}"
    
    return {
        "video_id": video_id,
        "relative_path": normalized_path,
        "filename": filename,
        "label": label,
        "label_name": label_name,
        "category": category,
        "generator": generator,
        "split_group": split_group,
        "metadata": video_metadata
    }


def build_manifest(dataset_dir: Optional[str] = None, output_path: str = "data/master_manifest.json"):
    """Generates the master manifest and saves it to output_path."""
    print("[INFO] Building master manifest...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 1. Load annotations
    annotations = {}
    if dataset_dir:
        local_ann_path = os.path.join(dataset_dir, "annotations.json")
        if os.path.exists(local_ann_path):
            with open(local_ann_path, "r", encoding="utf-8") as f:
                annotations = json.load(f)
            print(f"[INFO] Loaded local annotations from {local_ann_path}")
    
    if not annotations:
        print("[INFO] Fetching annotations from Hugging Face...")
        annotations = fetch_remote_annotations()
        print(f"[INFO] Fetched {len(annotations)} annotation records.")

    # 2. Get file list
    if dataset_dir and os.path.exists(dataset_dir):
        print(f"[INFO] Scanning local directory: {dataset_dir}")
        file_list = scan_local_file_list(dataset_dir)
        print(f"[INFO] Found {len(file_list)} local video files.")
    else:
        print("[INFO] Querying Hugging Face repository file listing...")
        file_list = fetch_remote_file_list()
        print(f"[INFO] Found {len(file_list)} video files in remote repository.")

    # 3. Parse each video
    manifest = []
    category_counts = {}
    generator_counts = {}
    
    for f in sorted(file_list):
        entry = extract_metadata_and_group(f, annotations)
        manifest.append(entry)
        
        c = entry["category"]
        g = entry["generator"]
        category_counts[c] = category_counts.get(c, 0) + 1
        generator_counts[g] = generator_counts.get(g, 0) + 1

    # 4. Save manifest
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 50)
    print(f" Master Manifest Summary: {len(manifest)} Videos")
    print("=" * 50)
    print("By Category:")
    for c, count in category_counts.items():
        print(f"  - {c:15s}: {count:4d}")
    print("\nBy Generator:")
    for g, count in generator_counts.items():
        print(f"  - {g:15s}: {count:4d}")
    print("=" * 50)
    print(f"[SUCCESS] Master manifest written to: {output_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build master dataset manifest for NFI deepfake dataset.")
    parser.add_argument("--dataset-dir", type=str, default=None,
                        help="Optional local path to the downloaded NFI dataset.")
    parser.add_argument("--output", type=str, default="data/master_manifest.json",
                        help="Output JSON path (default: data/master_manifest.json).")
    args = parser.parse_args()
    
    build_manifest(dataset_dir=args.dataset_dir, output_path=args.output)
