import argparse, json, os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

REPO = r"C:\dev\FAS-deepfake-capstone"
SIG_DIR = r"C:\datasets\nfi_signals"
FEATS = ["snr_mean", "snr_max", "peak_ratio_mean", "entropy_mean", "hr_spread", "corr_mean"]
OUT_JSON = os.path.join(REPO, "data", "predictions", "rppg_test_preds.json")
VAL_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rppg_val_preds.json")

ap = argparse.ArgumentParser()
ap.add_argument("--report", action="store_true",
                help="also print test metrics (do this ONCE, at the very end)")
args = ap.parse_args()

df = pd.read_csv(os.path.join(SIG_DIR, "features.csv"))
train, val, test = [df[df.split == s].reset_index(drop=True) for s in ["train", "val", "test"]]

manifest = json.load(open(os.path.join(REPO, "data", "manifests", "rppg_manifest.json")))
test_ids = [e["video_id"] for e in manifest["test"]]
assert len(test_ids) == len(set(test_ids)), "duplicate ids in manifest test list"
assert set(test_ids) == set(test.video_id), "test ids in features.csv differ from the manifest"

model = make_pipeline(
    SimpleImputer(strategy="median"),
    RandomForestClassifier(n_estimators=300, max_depth=3, min_samples_leaf=5,
                           class_weight="balanced", random_state=0))
model.fit(train[FEATS], train.label)          # fit on TRAIN only
s_val = model.predict_proba(val[FEATS])[:, 1]
s_test = model.predict_proba(test[FEATS])[:, 1]

print("Test videos:", len(test), "| with missing features (imputed):",
      int(test[FEATS].isna().any(axis=1).sum()))

score = dict(zip(test.video_id, s_test))
preds = [{"video_id": v, "score": round(float(score[v]), 4)} for v in test_ids]

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
if os.path.exists(OUT_JSON):
    old = json.load(open(OUT_JSON))
    print("\nExisting file keys:", list(old.keys()))
    ol = old.get("predictions", [])
    print("Existing entries:", len(ol), "| first:", ol[0] if ol else None)
json.dump({"branch": "rppg", "predictions": preds}, open(OUT_JSON, "w"), indent=2)
print("\nWrote", OUT_JSON, "| entries:", len(preds), "| first:", preds[0])

vscore = dict(zip(val.video_id, s_val))
json.dump({"branch": "rppg", "split": "val",
           "predictions": [{"video_id": v, "score": round(float(vscore[v]), 4)} for v in val.video_id]},
          open(VAL_JSON, "w"), indent=2)
print("Wrote", VAL_JSON)

if args.report:
    yv, yt = val.label.values, test.label.values
    grid = np.linspace(0.05, 0.95, 91)
    f1s = [precision_recall_fscore_support(yv, s_val >= t, average="binary", zero_division=0)[2] for t in grid]
    thr = float(grid[int(np.argmax(f1s))])
    rng = np.random.default_rng(0)
    aucs = []
    for _ in range(2000):
        b = rng.choice(len(yt), len(yt))
        if len(set(yt[b])) > 1:
            aucs.append(roc_auc_score(yt[b], s_test[b]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    p, r, f, _ = precision_recall_fscore_support(yt, s_test >= thr, average="binary", zero_division=0)
    print(f"\nTEST AUC {roc_auc_score(yt, s_test):.3f} (95% CI {lo:.3f} to {hi:.3f})")
    print(f"At the val-tuned threshold {thr:.2f}: precision {p:.3f} recall {r:.3f} F1 {f:.3f}")
    for k in ["face_swap", "synthetic"]:
        sub = test.kind.isin(["real", k]).values
        print(f"  real vs {k}: AUC {roc_auc_score(yt[sub], s_test[sub]):.3f} (n={sub.sum()})")