"""
predict.py
Command-Line Interface for the Unified Multimodal Deepfake & Face Authenticity Detector.

Usage:
    python predict.py --input path/to/video.mp4
    python predict.py --input path/to/image.jpg
"""

import argparse
import sys
import os

from pipeline.unified_pipeline import UnifiedDeepfakeDetector


def main():
    parser = argparse.ArgumentParser(description="Unified Multimodal Deepfake Detection (Spatial + Frequency + rPPG)")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input video (.mp4, .avi) or image (.jpg, .png)")
    parser.add_argument("--spatial-weights", type=str, default=None, help="Optional path to trained spatial MLP weights (.pth)")
    parser.add_argument("--frames", type=int, default=45, help="Number of video frames to sample (default: 45)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"\n[ERROR] File not found: {args.input}\n")
        sys.exit(1)

    print("\n" + "="*60)
    print(" UNIFIED MULTIMODAL FACE AUTHENTICITY DETECTION PIPELINE ")
    print("="*60)
    print(f"Input file: {args.input}")

    detector = UnifiedDeepfakeDetector(spatial_weights=args.spatial_weights)

    ext = os.path.splitext(args.input)[1].lower()
    is_image = ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

    print("\n[Processing] Running facial detection & multimodal analysis...")
    if is_image:
        result = detector.process_image(args.input)
    else:
        result = detector.process_video(args.input, max_frames=args.frames)

    if "error" in result:
        print(f"\n[ERROR] {result['error']}\n")
        sys.exit(1)

    print("\n" + "-"*60)
    print(" DETECTION RESULTS ")
    print("-"*60)
    verdict = result["verdict"]
    color_code = "\033[91m" if verdict == "FAKE" else "\033[92m"
    reset_code = "\033[0m"

    print(f"VERDICT:              {color_code}{verdict}{reset_code}")
    print(f"Confidence:           {result['confidence_percentage']}%")
    print(f"Fused Score (p):      {result['fused_score']:.4f} (Threshold: {result['decision_threshold']})")
    print(f"Face Detected:        {result['face_detected']}")

    if result["mode"] == "video":
        print(f"Estimated Heart Rate: {result['estimated_bpm']} BPM")
        print(f"Frames Analyzed:      {result['total_frames_analyzed']}")

    print("\n--- MODALITY BRANCH BREAKDOWN ---")
    s = result["scores"]
    w = result["weights_used"]
    print(f"1. Spatial Branch (SigLIP ViT):   Score = {s['spatial']} (Weight = {w['spatial']})")
    print(f"2. Frequency Branch (2D FFT):     Score = {s['frequency']} (Weight = {w['frequency']})")
    if s["rppg"] is not None:
        print(f"3. Physiological Branch (rPPG):   Score = {s['rppg']} (Weight = {w['rppg']})")
    else:
        print(f"3. Physiological Branch (rPPG):   N/A (Static image mode)")

    print("="*60 + "\n")


if __name__ == "__main__":
    main()
