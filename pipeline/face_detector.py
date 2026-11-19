"""
pipeline/face_detector.py
Face detection and boundary-aware cropping module.
Source: Adapted from Lalit's FaceDetector (origin/Lalit)
"""

import cv2
import numpy as np


class FaceDetector:
    """
    MTCNN face detector if available, otherwise graceful fallback to OpenCV Haar cascade.
    Returns (center_x, center_y, side) bounding box tuple.
    """
    def __init__(self):
        self.mtcnn = None
        try:
            import torch
            from facenet_pytorch import MTCNN
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.mtcnn = MTCNN(keep_all=True, device=device, post_process=False)
            self.detector_type = "MTCNN"
        except Exception:
            self.detector_type = "Haar"
        
        # OpenCV Haar Cascade fallback
        self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def _box(self, x1, y1, x2, y2, s=1.0):
        w, h = (x2 - x1) / s, (y2 - y1) / s
        # 1.3x expansion to capture facial boundaries, blending seams, and hairline
        return ((x1 + x2) / 2.0 / s, (y1 + y2) / 2.0 / s, max(w, h) * 1.3)

    def detect(self, bgr_image: np.ndarray):
        """
        Detects primary face in BGR image.
        Returns (cx, cy, side) or None if no face is detected.
        """
        if bgr_image is None or bgr_image.size == 0:
            return None
            
        H, W = bgr_image.shape[:2]
        s = min(1.0, 640.0 / max(H, W))
        small = cv2.resize(bgr_image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1.0 else bgr_image

        if self.mtcnn is not None:
            try:
                rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                boxes, _ = self.mtcnn.detect(rgb_small)
                if boxes is not None and len(boxes) > 0:
                    b = max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
                    cx, cy, side = self._box(*b, 1.0)
                    return (cx / s, cy / s, side / s)
            except Exception:
                pass

        # Fallback to Haar Cascade
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        faces = self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            cx, cy, side = self._box(x, y, x + w, y + h, 1.0)
            return (cx / s, cy / s, side / s)

        return None


def crop_resize_face(bgr_image: np.ndarray, box: tuple, size: int = 224) -> np.ndarray:
    """
    Crops square face region with boundary padding and resizes to target dimension (size x size).
    """
    if box is None:
        return cv2.resize(bgr_image, (size, size))

    cx, cy, side = box
    s = int(round(side))
    x1 = int(round(cx - side / 2.0))
    y1 = int(round(cy - side / 2.0))
    x2 = x1 + s
    y2 = y1 + s

    H, W = bgr_image.shape[:2]
    pl, pt, pr, pb = max(0, -x1), max(0, -y1), max(0, x2 - W), max(0, y2 - H)
    if pl or pt or pr or pb:
        bgr_image = cv2.copyMakeBorder(bgr_image, pt, pb, pl, pr, cv2.BORDER_REPLICATE)
        x1 += pl
        x2 += pl
        y1 += pt
        y2 += pt

    crop = bgr_image[y1:y2, x1:x2]
    interp = cv2.INTER_AREA if s > size else cv2.INTER_CUBIC
    return cv2.resize(crop, (size, size), interpolation=interp)
