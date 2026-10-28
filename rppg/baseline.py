import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

SIG_DIR = r"C:\datasets\nfi_signals"
FEATS = ["snr_mean", "snr_max", "peak_ratio_mean", "entropy_mean", "hr_spread", "corr_mean"]

df = pd.read_csv(os.path.join(SIG_DIR, "features.csv"))
train = df[df.split == "train"].reset_index(drop=True)
val = df[df.split == "val"].reset_index(drop=True)   # test is NOT loaded below


def boot_ci(y, s, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    out = []
    for _ in range(n):
        b = rng.choice(idx, len(idx))
        if len(set(y[b])) < 2:
            continue
        out.append(roc_auc_score(y[b], s[b]))
    return np.percentile(out, [2.5, 97.5])


models = {
    "logreg": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                            LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)),
    "forest": make_pipeline(SimpleImputer(strategy="median"),
                            RandomForestClassifier(n_estimators=300, max_depth=3, min_samples_leaf=5,
                                                   class_weight="balanced", random_state=0)),
}

yv = val.label.values
best = None
for name, m in models.items():
    m.fit(train[FEATS], train.label)
    s_tr = m.predict_proba(train[FEATS])[:, 1]
    s_va = m.predict_proba(val[FEATS])[:, 1]
    lo, hi = boot_ci(yv, s_va)
    print(f"{name}: train AUC {roc_auc_score(train.label, s_tr):.3f} | "
          f"val AUC {roc_auc_score(yv, s_va):.3f} (95% CI {lo:.3f} to {hi:.3f})")
    if best is None or roc_auc_score(yv, s_va) > best[1]:
        best = (name, roc_auc_score(yv, s_va), s_va)

name, auc, s_va = best
print(f"\nBest on val: {name} (AUC {auc:.3f})")

grid = np.linspace(0.05, 0.95, 91)
f1s = [precision_recall_fscore_support(yv, s_va >= t, average="binary", zero_division=0)[2] for t in grid]
thr = float(grid[int(np.argmax(f1s))])
p, r, f, _ = precision_recall_fscore_support(yv, s_va >= thr, average="binary", zero_division=0)
print(f"Threshold tuned on val: {thr:.2f} -> precision {p:.3f} recall {r:.3f} F1 {f:.3f}")

print("\nVal AUC by fake type (real vs that type):")
for k in ["face_swap", "synthetic"]:
    sub = val.kind.isin(["real", k]).values
    print(f"  real vs {k}: {roc_auc_score(yv[sub], s_va[sub]):.3f}  (n={sub.sum()})")

lr = models["logreg"]
coef = pd.Series(lr[-1].coef_[0], index=FEATS).round(3)
print("\nLogistic regression weights (positive = pushes toward FAKE):")
print(coef.to_string())

val.assign(score=s_va)[["video_id", "label", "kind", "score"]].to_csv(
    os.path.join(SIG_DIR, "val_scores_baseline.csv"), index=False)
print("\nSaved val scores.")