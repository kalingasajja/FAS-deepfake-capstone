"""
app.py
FastAPI Web Application for Unified Multimodal Face Authenticity Detection.
Zero-dependency upload (Pure Base64 JSON API).
"""

import os
import io
import base64
import tempfile
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from pipeline.unified_pipeline import UnifiedDeepfakeDetector

app = FastAPI(title="Multimodal Deepfake Authenticity Detector")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UploadPayload(BaseModel):
    filename: str
    data: str

# Lazy-loaded detector
detector = None

def get_detector():
    global detector
    if detector is None:
        detector = UnifiedDeepfakeDetector()
    return detector


def encode_image_base64(bgr_image: np.ndarray) -> str:
    if bgr_image is None or bgr_image.size == 0:
        return ""
    success, buffer = cv2.imencode(".jpg", bgr_image)
    if not success:
        return ""
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


def generate_fft_plot_base64(power_spectrum: np.ndarray) -> str:
    if power_spectrum is None or power_spectrum.size == 0:
        return ""
    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=100)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    ax.imshow(power_spectrum, cmap="inferno")
    ax.axis("off")
    plt.tight_layout(pad=0.2)
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    b64_str = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def generate_bvp_plot_base64(bvp_signal: list, fps: float = 30.0) -> str:
    if not bvp_signal or len(bvp_signal) < 10:
        return ""
    signal = np.array(bvp_signal)
    time_axis = np.arange(len(signal)) / fps

    fig, ax = plt.subplots(figsize=(6, 2.2), dpi=100)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    ax.plot(time_axis, signal, color="#ef4444", linewidth=2.0)
    ax.fill_between(time_axis, signal, color="#ef4444", alpha=0.15)
    ax.set_title("Extracted Capillary Blood Volume Pulse (rPPG)", color="#94a3b8", fontsize=10)
    ax.set_xlabel("Time (seconds)", color="#64748b", fontsize=8)
    ax.tick_params(colors="#64748b", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    plt.tight_layout(pad=0.3)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    b64_str = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/predict")
async def predict_authenticity(payload: UploadPayload):
    det = get_detector()
    filename = payload.filename.lower()
    is_image = any(filename.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"])

    raw_b64 = payload.data.split(",")[1] if "," in payload.data else payload.data
    contents = base64.b64decode(raw_b64)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        if is_image:
            res = det.process_image(tmp_path)
        else:
            res = det.process_video(tmp_path, max_frames=50)

        if "error" in res:
            return JSONResponse(status_code=400, content={"error": res["error"]})

        cropped_b64 = encode_image_base64(res["visuals"]["cropped_face"])
        fft_b64 = generate_fft_plot_base64(res["visuals"]["power_spectrum"])
        bvp_b64 = generate_bvp_plot_base64(res["visuals"].get("bvp_signal", []))

        out = {
            "mode": res["mode"],
            "face_detected": res["face_detected"],
            "verdict": res["verdict"],
            "is_fake": res["is_fake"],
            "fused_score": res["fused_score"],
            "confidence_percentage": res["confidence_percentage"],
            "decision_threshold": res["decision_threshold"],
            "scores": res["scores"],
            "weights_used": res["weights_used"],
            "estimated_bpm": res.get("estimated_bpm"),
            "total_frames": res.get("total_frames_analyzed"),
            "visuals": {
                "face_crop": cropped_b64,
                "fft_spectrum": fft_b64,
                "bvp_waveform": bvp_b64
            }
        }
        return JSONResponse(content=out)

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    print("\n" + "="*60)
    print(" MULTIMODAL DEEPFAKE SENTINEL LIVE DASHBOARD ")
    print("="*60)
    print(" Server running at: http://localhost:8000\n")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
