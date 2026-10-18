# Team Guide — Unified Face Authenticity Detection

> **Read this before touching any code.**  
> This document tells every team member exactly what to do, what files belong to them,
> and how to plug into the fusion pipeline.

---

## Who Does What

| Person | Role | Their deliverable |
|:---|:---|:---|
| **You (Kalin)** | Dataset Lead + Fusion | `data/splits.json`, `fusion/`, `run_fusion_pipeline.py` |
| **Member 1** | Spatial Branch (SigLIP / ViT) | `data/predictions/spatial_test_preds.json` |
| **Member 2** | Frequency Branch (FFT / DCT) | `data/predictions/frequency_test_preds.json` |
| **Member 3** | rPPG Branch (POS / pyVHR LSTM) | `data/predictions/rppg_test_preds.json` |

---

## Step-by-Step: What Kalin Runs (Dataset Lead)

Run **one command** at the start of the project, before anyone trains a model.

```bash
# If the NFI dataset is NOT yet downloaded (fetches file list + annotations from HuggingFace)
python run_all.py

# If you already downloaded the dataset locally (fastest, no internet needed)
python run_all.py --dataset-dir data/nfi_dataset

# To also run baseline fusion on mock predictions immediately
python run_all.py --run-fusion
```

`run_all.py` runs all 6 steps in sequence and prints a final checklist of created files.

**Expected terminal output (abridged):**
```
── Step 1 ─ Loading file list & annotations ──────────────────────
  Remote: 492 video files, 127 annotation records

── Step 2 ─ Building master_manifest.json ────────────────────────
  ✔  data/master_manifest.json
      face_swap :  85
      real      : 280
      synthetic : 127

── Step 3 ─ Creating splits (seed=42, 70/15/15) ──────────────────
  [TRAIN]  total=344  real=196  fake=148
  [VAL  ]  total= 74  real= 42  fake= 32
  [TEST ]  total= 74  real= 42  fake= 32

── Step 4 ─ Creating LOGO generalization splits ──────────────────
  leave-out AkoolAI        : test= 56  (fakes=14  real=42)
  leave-out Sora           : test= 68  (fakes=26  real=42)
  ...

── Step 5 ─ Exporting per-branch manifests ───────────────────────
  spatial_manifest.json   train=344  val=74  test=74
  frequency_manifest.json train=344  val=74  test=74
  rppg_manifest.json      train=344  val=74  test=74

── Step 6 ─ Generating mock branch predictions ───────────────────
  spatial_test_preds.json  (74 entries)
  frequency_test_preds.json(74 entries)
  rppg_test_preds.json     (74 entries)

══════════════════════════════════════════════════════════════
 ALL STEPS COMPLETE
```

This produces everything in `data/`:
```
data/
├── master_manifest.json          ← full catalog of all 492 videos
├── splits.json                   ← THE SHARED GROUND TRUTH (don't touch after this)
├── locked_splits.json            ← immutable backup of splits
├── splits_logo.json              ← leave-one-generator-out folds
└── manifests/
    ├── spatial_manifest.json     ← give this to Member 1
    ├── frequency_manifest.json   ← give this to Member 2
    └── rppg_manifest.json        ← give this to Member 3
```

> **Do not regenerate splits after experiments start. The locked split = the truth.**

---

## Step-by-Step: What Member 1 Does (Spatial — SigLIP / ViT)

### What you receive from Kalin
- `data/manifests/spatial_manifest.json`
- Dataset download instructions (HuggingFace link below)

### What the manifest looks like
```json
{
  "branch": "spatial",
  "train": [
    {"video_id": "real_CelebDF_id0_0000", "path": "real/CelebDF/id0_0000.mp4", "label": 0},
    {"video_id": "synthetic_Sora_1",      "path": "fake/Sora_SYN/1.mp4",       "label": 1}
  ],
  "val":  [...],
  "test": [...]
}
```

### What you train
1. Download the dataset: `huggingface-cli download NetherlandsForensicInstitute/rppg-deepfake-detection --repo-type dataset`
2. Use `train` split to fine-tune SigLIP / ViT on face crops.
3. Pick the best checkpoint based on `val` AUC.
4. Run inference on **all `test` samples**.

### What you export
A file `data/predictions/spatial_test_preds.json` **with exactly this structure**:
```json
{
  "branch": "spatial",
  "predictions": [
    {
      "video_id": "real_CelebDF_id0_0000",
      "score": 0.082
    },
    {
      "video_id": "synthetic_Sora_1",
      "score": 0.941
    }
  ]
}
```

Rules:
- `video_id` must **exactly match** the `video_id` fields from `spatial_manifest.json["test"]`.
- `score` is a float in `[0.0, 1.0]` — **higher = more likely FAKE**.
- Optionally include `"feature_vector": [...]` (list of floats, your final embedding before the classifier head) to unlock feature-level fusion.

---

## Step-by-Step: What Member 2 Does (Frequency — FFT / DCT)

### What you receive from Kalin
- `data/manifests/frequency_manifest.json`

### What you train
1. Extract face crops for each video in your `train` and `val` splits.
2. Compute 2D FFT / DCT residual spectra on the face crops.
3. Train a CNN or ViT classifier on the frequency maps.
4. Evaluate on `val` to pick the best threshold.
5. Run inference on all `test` samples.

### What you export
`data/predictions/frequency_test_preds.json` — **same schema as Member 1**:
```json
{
  "branch": "frequency",
  "predictions": [
    {"video_id": "real_CelebDF_id0_0000", "score": 0.045},
    {"video_id": "synthetic_Sora_1",      "score": 0.876}
  ]
}
```

---

## Step-by-Step: What Member 3 Does (rPPG — POS / pyVHR LSTM)

### What you receive from Kalin
- `data/manifests/rppg_manifest.json`
- The NFI GitHub code as reference: https://github.com/NetherlandsForensicInstitute/rPPG-Deepfake-Detection

### What you do
1. Install pyVHR from source (see NFI GitHub README for exact conda env).
2. Run `extraction/extract_signals.py` on each video to extract BVP / rPPG signals:
   ```bash
   python extraction/extract_signals.py --dataset-dir <path_to_nfi_dataset> --method pos --calc-bpm
   ```
   This produces a `signals.ndjson` file.
3. Use `train.py` with your `train`/`val` split to train the LSTM classifier.
4. Use `predict.py` to run inference on all `test` videos.
5. Convert the LSTM output probabilities into the common JSON format.

### What you export
`data/predictions/rppg_test_preds.json` — same schema:
```json
{
  "branch": "rppg",
  "predictions": [
    {"video_id": "real_CelebDF_id0_0000", "score": 0.103},
    {"video_id": "synthetic_Sora_1",      "score": 0.712}
  ]
}
```

> **Important:** The rPPG branch will perform poorly on videos labeled `"lighting": "bad"` or
> `"head_movement": "yes"` in the metadata. This is expected and is exactly what the fusion layer
> compensates for. Report your per-condition metrics using the `lighting` and `head_movement`
> fields from `master_manifest.json`.

---

## Step-by-Step: Kalin Runs Final Fusion

Once all 3 prediction files are in `data/predictions/`:

```bash
python run_fusion_pipeline.py \
  --manifest data/master_manifest.json \
  --splits   data/splits.json \
  --spatial-preds   data/predictions/spatial_test_preds.json \
  --frequency-preds data/predictions/frequency_test_preds.json \
  --rppg-preds      data/predictions/rppg_test_preds.json \
  --output-dir results/
```

### What this produces
```
results/
├── aligned_predictions.csv     ← ground truth + all 3 branch scores + metadata
├── ablation_results.csv        ← 7-way table: each branch alone vs 2-combos vs all-3
├── per_generator_results.csv   ← AUC/F1 for each of the 11 generators (Sora, Kling, etc.)
└── stress_test_results.csv     ← branch vs fusion accuracy under bad lighting & motion
```

---

## Dataset Download

```bash
pip install huggingface_hub

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='NetherlandsForensicInstitute/rppg-deepfake-detection',
    repo_type='dataset',
    local_dir='data/nfi_dataset'
)
"
```

Then pass the local path to `build_dataset_manifest.py`:
```bash
python scripts/build_dataset_manifest.py --dataset-dir data/nfi_dataset
```

---

## video_id Naming Convention

All video IDs follow this pattern so they never collide across branches:

| Category | Pattern | Example |
|:---|:---|:---|
| Real | `real_{source}_{filename_stem}` | `real_CelebDF_id0_0000` |
| Face-Swap | `face_swap_{generator}_{filename_stem}` | `face_swap_AkoolAI_001_AkoolAI_fa021` |
| Synthetic | `synthetic_{generator}_{filename_stem}` | `synthetic_Sora_1` |

---

## Common Mistakes to Avoid

| Mistake | Consequence | Fix |
|:---|:---|:---|
| Training on `test` split | Data leakage, inflated metrics | Only use `train` and `val` during training |
| Different `video_id` keys | Fusion can't align branch outputs | Copy IDs verbatim from your manifest file |
| Shuffling `splits.json` and regenerating | Changes test set, invalidates comparisons | Never regenerate splits after experiment start |
| Reporting accuracy only | Hides class imbalance | Always report Precision, Recall, F1, and ROC-AUC |
| Ignoring `metadata.lighting` | Can't explain rPPG degradation | Include stress-test breakdown in your branch report |

---

## Environment Setup

```bash
# Create shared environment (Python 3.10)
conda create -n deepfake-fusion python=3.10
conda activate deepfake-fusion

# Core ML dependencies (all 4 members need these)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install scikit-learn pandas numpy matplotlib huggingface_hub datasets

# Member 3 extra (rPPG)
conda install -c conda-forge cudatoolkit=12.3
pip install tensorflow keras keras-tuner pyVHR
```

---

## Contact & Handoff

- **Kalin (Fusion Lead):** Provides `data/manifests/<branch>_manifest.json` to each member. Reviews prediction files for schema compliance before running fusion.
- **All members:** Drop prediction files in `data/predictions/` with the exact filename and schema shown above.
- **Deadline for predictions:** Before Kalin runs `run_fusion_pipeline.py` for the final evaluation.
