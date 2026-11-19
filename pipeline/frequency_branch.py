"""
pipeline/frequency_branch.py
Frequency Forgery Detection Branch using 2D FFT, Azimuthal Radial Profiles, and DCT Spectra.
Source: Adapted directly from Lalit's implementation (origin/Lalit)
"""

import cv2
import numpy as np
import scipy.fft


class FrequencyDeepfakeDetector:
    """
    Analyzes 2D Fast Fourier Transform (FFT) and Discrete Cosine Transform (DCT)
    to detect upsampling artifacts, high-frequency grid residues, and spectral slope anomalies.
    """
    def __init__(self, size: int = 224):
        self.size = size
        self.nb = 32  # Number of radial frequency bins
        
        # Precompute radial distance matrices and 2D Hanning window
        yy, xx = np.indices((size, size))
        self.R = np.hypot(yy - (size - 1) / 2.0, xx - (size - 1) / 2.0)
        self.RB = np.minimum((self.R / (size / 2.0) * self.nb).astype(int), self.nb)
        self.HAN = np.outer(np.hanning(size), np.hanning(size)).astype(np.float32)

    def _log_spectrum(self, channel: np.ndarray):
        """
        Computes 2D FFT with Hanning windowing and returns log10 power spectrum and raw power.
        """
        ch_zero_mean = (channel - channel.mean()) * self.HAN
        F = np.fft.fftshift(np.fft.fft2(ch_zero_mean))
        power = np.abs(F) ** 2
        return np.log10(power + 1e-8), power

    def _radial_profile(self, P2: np.ndarray):
        """
        Azimuthal integration: averages power across concentric frequency rings.
        """
        valid_mask = self.RB < self.nb
        counts = np.bincount(self.RB[valid_mask], minlength=self.nb)
        sums = np.bincount(self.RB[valid_mask], P2[valid_mask], minlength=self.nb)
        return sums / np.maximum(counts, 1)

    def extract_frame_features(self, rgb_frame: np.ndarray) -> dict:
        """
        Extracts frequency and spectral residue features from an aligned RGB crop.
        """
        if rgb_frame.shape[0] != self.size or rgb_frame.shape[1] != self.size:
            rgb_frame = cv2.resize(rgb_frame, (self.size, self.size))

        ycc = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2YCrCb).astype(np.float32)
        Y = ycc[..., 0]

        # 1. 2D FFT on luminance channel
        P2, Pw = self._log_spectrum(Y)
        prof = self._radial_profile(P2)

        # 2. Spectral slope (decay rate across frequencies)
        # Natural images have 1/f^2 decay (steep negative slope); generative models have flatter slopes
        freq_indices = np.log(np.arange(4, self.nb) + 0.5)
        slope = float(np.polyfit(freq_indices, prof[4:], 1)[0])

        # 3. High-frequency energy ratio
        inside_circle = self.R < (self.size / 2.0)
        high_freq_circle = inside_circle & (self.R > 0.5 * (self.size / 2.0))
        total_inside_power = Pw[inside_circle].sum()
        hf_power = Pw[high_freq_circle].sum()
        hf_ratio = float(np.log10(hf_power / (total_inside_power + 1e-12) + 1e-12))

        # 4. Periodic peak disparity (upsampling checkerboard artifacts)
        expected_2d = prof[np.minimum(self.RB, self.nb - 1)]
        peak_mask = (self.R >= 0.12 * (self.size / 2.0)) & inside_circle
        residual = (P2 - expected_2d)[peak_mask]
        peak_max = float(residual.max()) if len(residual) > 0 else 0.0

        return {
            "spectral_slope": slope,
            "hf_ratio": hf_ratio,
            "peak_residual": peak_max,
            "power_spectrum": P2
        }

    def predict_image(self, bgr_crop: np.ndarray) -> tuple:
        """
        Predicts forgery probability based on spectral power and residue anomalies.
        Returns: (p_frequency in [0.0, 1.0], power_spectrum_2d)
        """
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        feats = self.extract_frame_features(rgb)

        # Normalization and logistic mapping:
        # Generative models exhibit higher HF energy ratio and flatter (less negative) spectral slope
        slope_score = 1.0 / (1.0 + np.exp( (feats["spectral_slope"] + 1.2) * 3.5 ))
        hf_score = 1.0 / (1.0 + np.exp( -(feats["hf_ratio"] + 1.5) * 4.0 ))
        peak_score = np.clip(feats["peak_residual"] / 5.0, 0.0, 1.0)

        # Weighted combination of frequency indicators
        p_frequency = float(np.clip(0.45 * hf_score + 0.35 * slope_score + 0.20 * peak_score, 0.0, 1.0))
        return p_frequency, feats["power_spectrum"]

    def predict_clip(self, bgr_frames: list) -> tuple:
        """
        Averages frequency probability across video frames.
        Returns: (p_frequency in [0.0, 1.0], last_power_spectrum)
        """
        if not bgr_frames:
            return 0.5, np.zeros((self.size, self.size))
            
        scores = []
        last_spec = None
        for f in bgr_frames:
            prob, spec = self.predict_image(f)
            scores.append(prob)
            last_spec = spec
            
        return float(np.mean(scores)), last_spec
