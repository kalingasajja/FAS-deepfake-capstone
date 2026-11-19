"""
pipeline/rppg_branch.py
Physiological Liveness Branch using Plane-Orthogonal-to-Skin (POS) Remote Photoplethysmography.
Source: Adapted from POS algorithm (Wang et al., IEEE TBME 2017) and pyVHR physiological standards.
"""

import cv2
import numpy as np
from scipy.signal import butter, filtfilt


class PhysiologicalRPPGDetector:
    """
    Extracts blood volume pulse (BVP) signals from facial capillary chrominance
    and evaluates physiological periodicity in the human cardiac range (42 - 150 BPM).
    """
    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.min_freq = 0.7  # 42 BPM
        self.max_freq = 2.5  # 150 BPM

    def _butter_bandpass(self, lowcut, highcut, fs, order=3):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype="band")
        return b, a

    def extract_bvp_pos(self, rgb_frames: list) -> tuple:
        """
        Executes Plane-Orthogonal-to-Skin (POS) algorithm on consecutive aligned RGB face frames.
        Returns: (filtered_bvp_signal, raw_fft_frequencies, raw_fft_power)
        """
        T = len(rgb_frames)
        if T < 15:
            return np.zeros(T), np.array([]), np.array([])

        # 1. Spatial averaging over central facial skin ROI (forehead & cheeks)
        H, W = rgb_frames[0].shape[:2]
        y1, y2 = int(0.20 * H), int(0.75 * H)
        x1, x2 = int(0.20 * W), int(0.80 * W)

        mean_rgb = np.zeros((T, 3), dtype=np.float32)
        for t in range(T):
            roi = rgb_frames[t][y1:y2, x1:x2]
            mean_rgb[t] = roi.mean(axis=(0, 1))

        # 2. Temporal normalization
        mean_channels = mean_rgb.mean(axis=0) + 1e-6
        Cn = mean_rgb / mean_channels

        # 3. Projection onto orthogonal chrominance planes (POS)
        # S1 = g_n - b_n
        # S2 = -2*r_n + g_n + b_n
        S1 = Cn[:, 1] - Cn[:, 2]
        S2 = -2.0 * Cn[:, 0] + Cn[:, 1] + Cn[:, 2]

        std_s1 = np.std(S1) + 1e-6
        std_s2 = np.std(S2) + 1e-6
        alpha = std_s1 / std_s2

        # Raw BVP signal
        h_signal = S1 + alpha * S2

        # 4. Butterworth bandpass filter in cardiac frequency range (0.7 - 2.5 Hz)
        try:
            b, a = self._butter_bandpass(self.min_freq, self.max_freq, self.fps, order=2)
            filtered_bvp = filtfilt(b, a, h_signal)
        except Exception:
            filtered_bvp = h_signal - np.mean(h_signal)

        # 5. FFT to compute power spectrum
        n_fft = max(256, T)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.fps)
        fft_power = np.abs(np.fft.rfft(filtered_bvp, n=n_fft)) ** 2

        return filtered_bvp, freqs, fft_power

    def predict_clip(self, bgr_frames: list) -> tuple:
        """
        Analyzes biological pulse periodicity across video frames.
        Returns: (p_rppg in [0.0, 1.0], bvp_signal, estimated_bpm)
        """
        if len(bgr_frames) < 15:
            # Insufficient frames to determine cardiac periodicity
            return 0.5, np.array([]), 0.0

        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in bgr_frames]
        bvp, freqs, power = self.extract_bvp_pos(rgb_frames)

        if len(freqs) == 0:
            return 0.5, np.array([]), 0.0

        # Cardiac frequency mask [0.7 Hz to 2.5 Hz]
        cardiac_mask = (freqs >= self.min_freq) & (freqs <= self.max_freq)
        cardiac_freqs = freqs[cardiac_mask]
        cardiac_power = power[cardiac_mask]

        if len(cardiac_power) == 0:
            return 0.5, bvp, 0.0

        peak_idx = np.argmax(cardiac_power)
        peak_freq = cardiac_freqs[peak_idx]
        peak_val = cardiac_power[peak_idx]
        estimated_bpm = float(peak_freq * 60.0)

        # Signal-to-Noise Ratio (SNR) in cardiac band
        mean_noise = np.mean(cardiac_power) + 1e-8
        snr = float(peak_val / mean_noise)

        # Real faces exhibit periodic pulse with SNR >= 2.5 - 4.0;
        # Deepfakes lack cardiovascular signals, resulting in low, flat SNR
        if snr > 4.0:
            p_fake = 0.05 + 0.15 * max(0.0, 1.0 - (snr - 4.0) / 4.0)
        elif snr > 2.2:
            p_fake = 0.20 + 0.30 * (1.0 - (snr - 2.2) / 1.8)
        else:
            p_fake = 0.55 + 0.40 * min(1.0, (2.2 - snr) / 1.5)

        p_rppg = float(np.clip(p_fake, 0.0, 1.0))
        return p_rppg, bvp, estimated_bpm
