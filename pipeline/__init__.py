"""
Unified Multimodal Face Authenticity Detection Pipeline
Integrates:
- Spatial Branch (SigLIP + MLP from Rohith)
- Frequency Branch (2D FFT / DCT from Lalit)
- Physiological Branch (POS rPPG from Abhishek)
- Calibrated Late Fusion Layer (Kalinga)
"""
from .unified_pipeline import UnifiedDeepfakeDetector
