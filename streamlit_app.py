"""
streamlit_app.py
Streamlit Web Application for Multimodal Face Authenticity Detection.

Usage:
    streamlit run streamlit_app.py
"""

import tempfile
import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pipeline.unified_pipeline import UnifiedDeepfakeDetector

st.set_page_config(page_title="Deepfake Sentinel", page_icon="🛡️", layout="wide")

st.title("🛡️ Multimodal Deepfake & Face Authenticity Detector")
st.markdown("Combines **Spatial Vision Transformer (SigLIP)**, **Frequency 2D FFT / DCT Spectra**, and **Physiological rPPG (POS)** into a calibrated late-fusion decision.")

@st.cache_resource
def load_detector():
    return UnifiedDeepfakeDetector()

detector = load_detector()

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Upload Media")
    uploaded_file = st.file_uploader("Upload portrait video or face image", type=["mp4", "avi", "jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        filename = uploaded_file.name.lower()
        is_image = any(filename.endswith(ext) for ext in [".jpg", ".jpeg", ".png"])
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        if is_image:
            st.image(tmp_path, caption="Uploaded Image", use_container_width=True)
        else:
            st.video(tmp_path)

        if st.button("🚀 Run Multimodal Inspection", type="primary", use_container_width=True):
            with st.spinner("Analyzing spatial semantics, frequency spectra, and capillary pulse..."):
                if is_image:
                    result = detector.process_image(tmp_path)
                else:
                    result = detector.process_video(tmp_path, max_frames=50)

            os.remove(tmp_path)

            with col_right:
                st.subheader("Detection Results")
                if "error" in result:
                    st.error(result["error"])
                else:
                    is_fake = result["is_fake"]
                    if is_fake:
                        st.error(f"🚨 **VERDICT: DEEPFAKE DETECTED** (Confidence: {result['confidence_percentage']}%)")
                    else:
                        st.success(f"✅ **VERDICT: AUTHENTIC REAL FACE** (Confidence: {result['confidence_percentage']}%)")

                    st.metric("Fused Score (p_fused)", f"{result['fused_score']:.4f}", f"Threshold: {result['decision_threshold']}")

                    st.markdown("### Modality Breakdown")
                    st.write(f"**Spatial (SigLIP ViT):** `{result['scores']['spatial']}` (Weight: {result['weights_used']['spatial']})")
                    st.progress(float(result['scores']['spatial'] or 0.0))

                    st.write(f"**Frequency (2D FFT):** `{result['scores']['frequency']}` (Weight: {result['weights_used']['frequency']})")
                    st.progress(float(result['scores']['frequency'] or 0.0))

                    if result['scores']['rppg'] is not None:
                        st.write(f"**Physiological (rPPG):** `{result['scores']['rppg']}` (Weight: {result['weights_used']['rppg']}) &bull; Heart Rate: `{result['estimated_bpm']} BPM`")
                        st.progress(float(result['scores']['rppg'] or 0.0))

                    st.markdown("### Visual Representations")
                    tab1, tab2 = st.tabs(["Aligned Face Crop", "2D FFT Power Spectrum"])
                    with tab1:
                        if result["visuals"]["cropped_face"] is not None:
                            st.image(result["visuals"]["cropped_face"][..., ::-1], caption="Aligned Face Crop (224x224)", use_container_width=True)
                    with tab2:
                        if result["visuals"]["power_spectrum"] is not None:
                            fig, ax = plt.subplots(figsize=(4, 4))
                            ax.imshow(result["visuals"]["power_spectrum"], cmap="inferno")
                            ax.axis("off")
                            st.pyplot(fig)
