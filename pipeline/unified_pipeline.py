"""
pipeline/unified_pipeline.py
The master Unified Multimodal Face Authenticity Detection Pipeline.
Combines:
  1. Face Detection & Bounding Box Normalization (Lalit)
  2. Spatial Branch (Rohith: SigLIP ViT)
  3. Frequency Branch (Lalit: 2D FFT & DCT)
  4. Physiological Branch (Abhishek: POS rPPG)
  5. Calibrated Multimodal Late Fusion (Kalinga)
"""

import os
import cv2
import numpy as np

from .face_detector import FaceDetector, crop_resize_face
from .spatial_branch import SpatialDeepfakeDetector
from .frequency_branch import FrequencyDeepfakeDetector
from .rppg_branch import PhysiologicalRPPGDetector


class UnifiedDeepfakeDetector:
    """
    End-to-End Multimodal Deepfake and Face Authenticity Detector.
    """
    def __init__(self, spatial_weights: str = None, device: str = None):
        print("[Pipeline] Initializing Face Detector...")
        self.face_detector = FaceDetector()

        print("[Pipeline] Initializing Spatial Branch (SigLIP ViT)...")
        self.spatial_branch = SpatialDeepfakeDetector(weights_path=spatial_weights, device=device)

        print("[Pipeline] Initializing Frequency Branch (2D FFT / DCT)...")
        self.frequency_branch = FrequencyDeepfakeDetector(size=224)

        print("[Pipeline] Initializing Physiological Branch (POS rPPG)...")
        self.rppg_branch = PhysiologicalRPPGDetector(fps=30.0)

        # Calibrated late fusion parameters
        self.w_s = 0.30
        self.w_f = 0.55
        self.w_r = 0.15
        self.threshold = 0.35

    def process_image(self, image_input) -> dict:
        """
        Analyzes a single static image (or image file path).
        For static images, falls back to the calibrated 2-branch ablation (Spatial + Frequency).
        """
        if isinstance(image_input, str):
            bgr = cv2.imread(image_input)
        else:
            bgr = image_input

        if bgr is None:
            return {"error": "Could not read input image"}

        # 1. Face detection
        box = self.face_detector.detect(bgr)
        face_found = box is not None
        cropped_face = crop_resize_face(bgr, box, size=224)

        # 2. Branch inference
        p_spatial = self.spatial_branch.predict_image(cropped_face)
        p_freq, power_spec = self.frequency_branch.predict_image(cropped_face)

        # Static image: 2-branch ablation weights (normalized: 0.35 Spatial, 0.65 Frequency)
        w_s_img = 0.35
        w_f_img = 0.65
        p_fused = float(w_s_img * p_spatial + w_f_img * p_freq)

        is_fake = p_fused >= self.threshold
        verdict = "FAKE" if is_fake else "REAL"
        confidence = p_fused if is_fake else (1.0 - p_fused)

        return {
            "mode": "image",
            "face_detected": face_found,
            "verdict": verdict,
            "is_fake": is_fake,
            "fused_score": round(p_fused, 4),
            "confidence_percentage": round(confidence * 100.0, 1),
            "decision_threshold": self.threshold,
            "scores": {
                "spatial": round(p_spatial, 4),
                "frequency": round(p_freq, 4),
                "rppg": None  # rPPG requires temporal video
            },
            "weights_used": {
                "spatial": w_s_img,
                "frequency": w_f_img,
                "rppg": 0.0
            },
            "visuals": {
                "cropped_face": cropped_face,
                "power_spectrum": power_spec
            }
        }

    def process_video(self, video_path: str, max_frames: int = 60) -> dict:
        """
        Analyzes a video file through all 3 multimodal branches.
        """
        if not os.path.exists(video_path):
            return {"error": f"Video file not found: {video_path}"}

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0
        self.rppg_branch.fps = float(fps)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return {"error": "Invalid or empty video"}

        # Read evenly spaced frames across the video
        sample_indices = np.linspace(0, total_frames - 1, min(total_frames, max_frames), dtype=int)
        
        frames = []
        cropped_faces = []
        face_boxes = []

        cur_idx = 0
        target_idx_set = set(sample_indices)
        max_idx = sample_indices[-1]

        while cap.isOpened() and cur_idx <= max_idx:
            ret, frame = cap.read()
            if not ret:
                break
            if cur_idx in target_idx_set:
                frames.append(frame)
            cur_idx += 1
        cap.release()

        if not frames:
            return {"error": "Could not extract frames from video"}

        # 1. Face detection on middle frame
        mid_frame = frames[len(frames) // 2]
        box = self.face_detector.detect(mid_frame)
        face_found = box is not None

        for f in frames:
            crop = crop_resize_face(f, box, size=224)
            cropped_faces.append(crop)

        # 2. Run all 3 branches
        p_spatial = self.spatial_branch.predict_clip(cropped_faces)
        p_freq, last_spec = self.frequency_branch.predict_clip(cropped_faces)
        p_rppg, bvp_signal, bpm = self.rppg_branch.predict_clip(cropped_faces)

        # 3. Calibrated late fusion
        p_fused = float(self.w_s * p_spatial + self.w_f * p_freq + self.w_r * p_rppg)

        is_fake = p_fused >= self.threshold
        verdict = "FAKE" if is_fake else "REAL"
        confidence = p_fused if is_fake else (1.0 - p_fused)

        return {
            "mode": "video",
            "face_detected": face_found,
            "total_frames_analyzed": len(frames),
            "estimated_bpm": round(bpm, 1),
            "verdict": verdict,
            "is_fake": is_fake,
            "fused_score": round(p_fused, 4),
            "confidence_percentage": round(confidence * 100.0, 1),
            "decision_threshold": self.threshold,
            "scores": {
                "spatial": round(p_spatial, 4),
                "frequency": round(p_freq, 4),
                "rppg": round(p_rppg, 4)
            },
            "weights_used": {
                "spatial": self.w_s,
                "frequency": self.w_f,
                "rppg": self.w_r
            },
            "visuals": {
                "cropped_face": cropped_faces[len(cropped_faces) // 2] if cropped_faces else None,
                "power_spectrum": last_spec,
                "bvp_signal": bvp_signal.tolist() if isinstance(bvp_signal, np.ndarray) else []
            }
        }
