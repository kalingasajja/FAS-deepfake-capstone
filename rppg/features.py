import glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, periodogram
from sklearn.metrics import roc_auc_score

SIG_DIR = r"C:\datasets\nfi_signals"
FS, SECONDS, LO, HI = 30.0, 8.0, 0.7, 4.0
FEATS = ["snr_mean", "snr_max", "peak_ratio_mean", "entropy_mean", "hr_spread", "corr_mean"]


def resample(t, rgb):
    n = int(SECONDS * FS)
    grid = np.arange(n) / FS
    out = np.zeros((n, rgb.shape[1], 3), np.float64)
    for r in range(rgb.shape[1]):
        for c in range(3):
            y = rgb[:, r, c]
            ok = ~np.isnan(y)
            if ok.sum() < 10:
                return None
            out[:, r, c] = np.interp(grid, t[ok], y[ok])
    return out


def pos(x):
    xn = x / (x.mean(axis=0) + 1e-8)
    X = xn[:, 1] - xn[:, 2]
    Y = xn[:, 1] + xn[:, 2] - 2 * xn[:, 0]
    return X - (X.std() / (Y.std() + 1e-8)) * Y


def bandpass(x):
    b, a = butter(3, [LO / (FS / 2), HI / (FS / 2)], btype="band")
    return filtfilt(b, a, x)


def roi_features(p):
    f, P = periodogram(p, fs=FS, window="hann", nfft=2048)
    m = (f >= LO) & (f <= HI)
    f, P = f[m], P[m]
    fp = f[np.argmax(P)]
    near = np.abs(f - fp) <= 0.15
    sig = near | (np.abs(f - 2 * fp) <= 0.15)
    snr = 10 * np.log10((P[sig].sum() + 1e-12) / (P[~sig].sum() + 1e-12))
    peak_ratio = P[near].sum() / (P.sum() + 1e-12)
    pn = P / (P.sum() + 1e-12)
    ent = -(pn * np.log(pn + 1e-12)).sum() / np.log(len(pn))
    return fp * 60, snr, peak_ratio, ent


def video_features(rgb):
    pulses = [bandpass(pos(rgb[:, r, :])) for r in range(rgb.shape[1])]
    per = np.array([roi_features(p) for p in pulses])
    corr = np.corrcoef(np.stack(pulses))
    iu = np.triu_indices(len(pulses), 1)
    return dict(snr_mean=per[:, 1].mean(), snr_max=per[:, 1].max(),
                peak_ratio_mean=per[:, 2].mean(), entropy_mean=per[:, 3].mean(),
                hr_spread=per[:, 0].std(), corr_mean=corr[iu].mean())


def kind_of(vid):
    if vid.startswith("real"):
        return "real"
    return "face_swap" if vid.startswith("face_swap") else "synthetic"


rows = []
for f in sorted(glob.glob(os.path.join(SIG_DIR, "*.npz"))):
    vid = os.path.splitext(os.path.basename(f))[0]
    d = np.load(f)
    rec = dict(video_id=vid, split=str(d["split"]), label=int(d["label"]),
               kind=kind_of(vid), face_rate=float(d["det"].mean()))
    rgb = resample(d["t"], d["rgb"])
    if rgb is None:
        rec.update({k: np.nan for k in FEATS})
    else:
        rec.update(video_features(rgb))
    rows.append(rec)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(SIG_DIR, "features.csv"), index=False)
print("Videos:", len(df), "| videos with NaN features:", int(df[FEATS].isna().any(axis=1).sum()))
print("Face rate below 0.7:", df[df.face_rate < 0.7].groupby("kind").size().to_dict())

# We only ever look at train and val here. Test stays untouched.
dev = df[df.split.isin(["train", "val"])]
print("\nMean feature value by group (train+val):")
print(dev.groupby("kind")[FEATS].mean().round(3))

print("\nSingle-feature AUC for detecting FAKE (0.5 = no signal; below 0.5 = higher value means real):")
res = []
for feat in FEATS:
    r = {"feature": feat}
    for s in ["train", "val"]:
        sub = df[(df.split == s) & df[feat].notna()]
        r[s] = round(roc_auc_score(sub.label, sub[feat]), 3)
    res.append(r)
print(pd.DataFrame(res).to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, feat in zip(axes, ["snr_mean", "corr_mean", "entropy_mean"]):
    groups = ["real", "face_swap", "synthetic"]
    ax.boxplot([dev[dev.kind == g][feat].dropna() for g in groups])
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(groups)
    ax.set_title(feat)
plt.tight_layout()
plt.savefig(os.path.join(SIG_DIR, "feature_boxplots.png"), dpi=120)
print("\nSaved plot:", os.path.join(SIG_DIR, "feature_boxplots.png"))