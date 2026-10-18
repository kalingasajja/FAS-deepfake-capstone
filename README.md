# Unified Face Authenticity Detection: Common Dataset & Fusion Pipeline

Capstone Project: **“Unified Face Authenticity Detection Against Physical and Digital Attacks”**  
**Role:** Dataset Pipeline & Multimodal Fusion Lead

---

## 1. Architecture Overview

```
                      Input Portrait Video (.mp4)
                                  │
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
  Member 1: Spatial      Member 2: Frequency        Member 3: rPPG
  SigLIP / ViT-Base      2D FFT / DCT Spectra      pyVHR / POS / LSTM
  (Visual Artifacts)     (Frequency Forgeries)     (Physiological BVP)
         │                        │                        │
  p_spatial in [0,1]     p_frequency in [0,1]      p_rppg in [0,1]
         └────────────────────────┬────────────────────────┘
                                  ▼
                        Your Role: Fusion Layer
         Stage 1: Calibrated Late Fusion (w_s p_s + w_f p_f + w_r p_r)
         Stage 2: Gated Multimodal Unit (GMU) / Cross-Attention
                                  ▼
                   Final Authenticity Prediction
```

---

## 2. Dataset Strategy & Leakage Safety

We use the **Netherlands Forensic Institute (NFI)** rPPG Deepfake Detection dataset ([Hugging Face](https://huggingface.co/datasets/NetherlandsForensicInstitute/rppg-deepfake-detection)):
* **Total Videos:** 492 clips (~1.88 GB).
* **Real Videos (280 clips):** Celeb-DF v2 portrait videos spanning **31 distinct subject identities** (`id0` to `id61`).
* **Face-Swap Deepfakes (85 clips):** Akool AI, Deepfaker, FaceSwapper, Kaggle, Reface AI.
* **Fully Synthetic Videos (127 clips):** OpenAI Sora, Kling AI, Luma Dream Machine, Pika, PixVerse, Runway Gen-2/3.
* **Confounder Metadata:** Annotated in `annotations.json` for `lighting` (`well-lit` vs `bad`) and `head_movement` (`yes` vs `no`).

### Leakage-Safe Partitioning Rules
To prevent artificial performance inflation across splits:
1. **Real Videos (`real/CelebDF/`):** Clustered by `subject_id` (`id0`, `id1`...). No celebrity appears across both Train and Test.
2. **Face-Swaps (`fake/*_FS/`):** Clustered by base target video ID (`001`, `009`...). No target scene/torso leaks into Test.
3. **Synthetic Videos (`fake/*_SYN/`):** Clustered by prompt index (`prompt_1`..`prompt_20`). Identical generation prompts do not cross splits.

---

## 3. Directory Layout

```
deepfake-proj/
├── data/
│   ├── master_manifest.json          # Master catalog of all 492 videos with metadata
│   ├── splits.json                   # 70% Train / 15% Val / 15% Test (leakage-safe)
│   ├── locked_splits.json            # Immutable frozen copy for project reproducibility
│   ├── splits_logo.json              # Leave-One-Generator-Out splits for generalization
│   ├── manifests/
│   │   ├── spatial_manifest.json     # For Member 1 (Spatial)
│   │   ├── frequency_manifest.json   # For Member 2 (Frequency)
│   │   └── rppg_manifest.json        # For Member 3 (rPPG)
│   └── predictions/                  # Where teammates drop their prediction files
├── scripts/
│   ├── build_dataset_manifest.py     # Generates master_manifest.json from HF / local directory
│   ├── create_splits.py              # Generates leakage-safe splits and branch manifests
│   └── generate_mock_predictions.py  # Generates mock branch predictions for fast testing
├── fusion/
│   ├── contract.py                   # Validates and merges branch predictions
│   ├── baseline_fusion.py            # Late score fusion (w_s p_s + w_f p_f + w_r p_r) + ablation
│   ├── learned_fusion.py             # ScoreMLP, Gated Multimodal Unit, Cross-Attention
│   └── evaluate.py                   # Accuracy, F1, AUC, per-generator & stress-test reports
└── run_fusion_pipeline.py            # Orchestrator to run the entire evaluation
```

---

## 4. Teammate Output Data Contract

Each teammate exports their branch predictions as a JSON or CSV file to `data/predictions/`:

### Expected JSON Schema (`data/predictions/<branch>_test_preds.json`)
```json
{
  "branch": "spatial",
  "predictions": [
    {
      "video_id": "real_CelebDF_id0_0000",
      "score": 0.082,
      "feature_vector": [0.12, -0.45, 0.89]
    },
    {
      "video_id": "synthetic_Sora_1",
      "score": 0.941,
      "feature_vector": [0.78, 0.23, -0.11]
    }
  ]
}
```

* `score`: Must be a continuous float in `[0.0, 1.0]`, representing the predicted probability that the video is **FAKE**.
* `feature_vector` (Optional): 1D array of intermediate embeddings for feature-level fusion.

---

## 5. Quickstart & Workflow

### Step 1: Generate Master Manifest & Frozen Splits
```bash
# Build master manifest from Hugging Face or local path
python scripts/build_dataset_manifest.py

# Generate deterministic, leakage-safe splits
python scripts/create_splits.py --seed 42
```

### Step 2: Test Fusion with Simulated Predictions
```bash
# Generate simulated predictions (includes simulated rPPG degradation under bad lighting)
python scripts/generate_mock_predictions.py

# Run complete late-fusion optimization and ablation reporting
python run_fusion_pipeline.py
```

### Step 3: Integrate Real Teammate Predictions
Once Member 1, 2, and 3 produce their test predictions:
```bash
python run_fusion_pipeline.py \
  --spatial-preds data/predictions/spatial_test_preds.json \
  --frequency-preds data/predictions/frequency_test_preds.json \
  --rppg-preds data/predictions/rppg_test_preds.json \
  --output-dir results/
```

Outputs generated in `results/`:
1. `aligned_predictions.csv`: Unified ground-truth + multi-branch score matrix.
2. `ablation_results.csv`: Individual branch metrics vs 2-branch and 3-branch fused performance.
3. `per_generator_results.csv`: Detection accuracy across all 11 generative models (Akool, Sora, Kling, Runway, etc.).
4. `stress_test_results.csv`: Robustness evaluation under low light and high motion.

---
---

# Literature Review & System Scoping

## 1. Problem Statement

Facial recognition-based attendance systems are vulnerable to two overlapping attack classes:

1. **Presentation attacks (PAs)** — printed photos, screen replays (phone/tablet/monitor showing a photo or video of the enrolled person), and masks.
2. **Digital/deepfake attacks** — real-time face-swapped or reenacted video injected either by showing a deepfake video to the camera, or by feeding a synthetic video directly through a virtual camera driver (bypassing the physical camera entirely).

Most existing commercial and academic systems treat these as **two separate problems** (a "liveness/anti-spoofing" module and a "deepfake detector"), and further rely on **active challenge-response** (blink, turn head, smile) which is increasingly defeatable by playing a reactive pre-recorded or live-driven synthetic video.

**Research gap:** a real-time, passive (or low-friction), unified detector that is robust to physical presentation attacks, digital/deepfake attacks, and their combination, under real-world attendance-system constraints (variable hardware, lighting, low latency, no specialized sensors in most deployments).

---

## 2. Threat Model Taxonomy

Structure your "Related Work" chapter around this taxonomy — it is the organizing spine most joint-detection papers use.

| Category | Attack type | Example | Primary detectable artifact |
|---|---|---|---|
| **Physical / Presentation** | Print attack | Photo held up to camera | Flat texture, no depth, moiré from paper |
| | Screen replay | Video/photo played on phone or monitor | Screen reflection, moiré, bezel edges, refresh-rate banding |
| | 3D mask | Silicone/resin mask | Material reflectance, absent rPPG signal |
| **Digital / Synthetic** | Face-swap deepfake (offline, played to camera) | DeepFakes-style video shown on a screen | Blending artifacts + screen-replay artifacts combined |
| | Reenactment deepfake | Face2Face / NeuralTextures-style driven video | Subtle boundary/texture artifacts, temporal flicker |
| | **Real-time injected deepfake (virtual camera)** | Live face-swap software feeding a virtual webcam driver | **No physical camera artifacts at all** — must be caught via appearance/temporal/model-level artifacts only, since there is no screen to detect |
| **Hybrid / Emerging** | Adversarially fabricated rPPG | Synthetic pulsing to fake a heartbeat signal | Statistically implausible pulse waveform |
| | Challenge-response bypass | Pre-recorded reactive clips, or live deepfake driven to comply with prompts (blink/turn on command) | Unnatural timing/latency between prompt and reaction; inconsistency between micro-expression and challenge |

**Key point for your gap analysis:** the virtual-camera injection attack is the most under-covered in older FAS literature because it eliminates every "recapture" artifact (screen bezel, moiré, reflection) that classical anti-spoofing relies on. This is worth stating explicitly as a driving motivation for a unified appearance + physiological + temporal approach rather than a purely texture-based one.

---

## 3. Literature Review Structure (suggested chapter/section layout)

1. **Introduction** — attendance-fraud problem statement, real-world incidents/motivation
2. **Background** — face recognition pipeline basics, where liveness/deepfake checks sit in it
3. **Presentation Attack Detection (PAD) literature**
   - Hand-crafted feature era (LBP, image quality assessment, distortion analysis)
   - Deep learning era (auxiliary depth/rPPG supervision, central difference networks, domain generalization methods)
4. **Physiological signal-based liveness (rPPG)**
   - Heart-rate extraction methods (CHROM, POS, deep rPPG)
   - rPPG for PAD (advantages, and the fabrication vulnerability)
5. **Deepfake / Face Forgery Detection literature**
   - Dataset-driven CNN detectors (FaceForensics++ lineage)
   - Frequency-domain and artifact-based detectors
   - Generalization to unseen manipulation methods
6. **Joint / Unified Physical-Digital Attack Detection** (your core related work — small but growing area)
7. **Generalization & Domain Adaptation** — why single-dataset accuracy is misleading; cross-dataset protocols
8. **Research Gap & Proposed Direction**

---

## 4. Comparison Table Template (fill in as you read)

| Paper | Attack types covered | Modality (RGB / depth / IR / rPPG) | Method family | Cross-dataset generalization reported? | Real-time feasible? |
|---|---|---|---|---|---|
| Yu et al., Joint FAS+Forgery Benchmark (2022/2024) | PA + deepfake | RGB + rPPG | Two-branch physiological network + fusion | Yes | Moderate |
| TransRPPG | 3D mask | RGB (rPPG) | Transformer on rPPG signal maps | Partial | Yes |
| FaceForensics++ (Rössler et al., 2019) | Deepfake only | RGB | XceptionNet on face crop | Limited (in-dataset) | Yes |
| MoE joint physical/digital (2024) | PA + deepfake | RGB | Mixture-of-Experts | Yes | Unclear |
| OR-PAD (2025) | rPPG-fabrication attack | RGB (rPPG) | Benchmark + evaluation of rPPG methods | Yes | N/A (benchmark) |

*(Extend this table as you read each paper — it becomes your literature review's core evidence table and later your baseline comparison table for experiments.)*

---

## 5. Proposed System Scoping — Decisions to Make Early

These decisions will materially change which literature is relevant, so resolve them before finalizing your review's scope:

1. **Hardware assumption**: Standard RGB webcam/phone camera only, or can you assume IR/depth (like Face ID)? This is the single biggest scope decision — depth sensors make most RGB-only spoofing literature secondary.
2. **Interaction mode**: Fully passive (single frame or short passive clip) vs. active challenge-response vs. hybrid. Given attendance-system UX constraints (speed, queues), passive or short-passive-clip is likely the right target — state this explicitly as a design constraint.
3. **Latency budget**: Define a target (e.g., <1–2 seconds per person) — this determines whether rPPG (needs several seconds of video) is feasible or only usable as a secondary/step-up check.
4. **Attack scope for v1**: Decide whether real-time virtual-camera injection is in-scope for your first model or a stated future-work limitation — it is the hardest attack class and may warrant being explicitly deferred with justification.

---

## 6. Suggested Architecture Direction (for your proposal chapter)

Based on the joint-detection literature, a reasonable architecture to propose:

- **Face detection & tracking** → domain-specific face crop (as in FaceForensics++)
- **Dual-branch feature extraction**:
  - Spatial/appearance branch (CNN or ViT backbone on RGB crop) — catches texture, blending, and print/screen artifacts
  - Frequency branch (FFT/DCT features) — catches GAN upsampling artifacts and moiré/screen-replay patterns
  - Optional physiological branch (short-window rPPG) — catches absence of pulse in masks/photos/screens, used as a secondary signal or step-up check given latency cost
- **Fusion + joint multi-task training** on merged live/bonafide samples from both PA and deepfake datasets, treating "attack" as a shared negative class with the physical/digital distinction as an auxiliary task (per Yu et al.'s finding that joint training improves generalization)
- **Domain generalization technique** (e.g., adversarial domain adaptation, meta-learning, or simple heavy augmentation across cameras/lighting/compression) layered on top, since cross-dataset generalization — not in-dataset accuracy — is the actual deployment bottleneck

---

## 7. Evaluation Protocol Recommendations

- Report **cross-dataset** (train on one set of datasets, test on unseen ones), not just in-dataset accuracy — this is the standard the FAS community now expects and reviewers will ask for it.
- Use **HTER (Half Total Error Rate)** and **APCER/BPCER** (standard ISO/IEC 30107 metrics) alongside plain accuracy, since these are the field's standard reporting metrics.
- Include an explicit **ablation** isolating the contribution of the rPPG/physiological branch, since that is often your most novel/expensive component.
- If possible, construct or use a **held-out "unknown attack" split** (train without a given attack type, test on it) to simulate real-world novel-attack robustness — this is more convincing than i.i.d. splits.

---

## 7a. Vision Transformer (ViT) Approaches — Dedicated Section

ViTs have been applied to both halves of your problem (liveness/PAD and deepfake detection), and the results speak directly to the generalization problem identified in the Yu et al. survey (Section 8, ref. #2 below). This section should be its own subsection in your review, likely under "Generalized Deep Learning Methods" (PAD side) and "Deepfake / Face Forgery Detection" (digital side), with a bridging paragraph on multimodal ViT+physiological fusion as your most novel angle.

### ViT for Face Anti-Spoofing (physical attacks)

| Paper | Idea | Key result | Insight |
|---|---|---|---|
| **ViTranZFAS** (George & Marcel, IJCB 2021, arXiv:2011.08019) | Freeze pretrained ViT, fine-tune only the final FC+sigmoid head | ~12.7% HTER cross-dataset vs. 29–38% for CNN baselines (DenseNetPAD, ResNetPAD, DeepPixBiS) | Large-scale pretraining + minimal fine-tuning resists overfitting to dataset-specific nuisance patterns far better than end-to-end CNN training |
| **FLIP** (Srivatsan et al., ICCV 2023, arXiv:2309.16649) | Initialize ViT with CLIP (vision-language) weights; align image embedding with natural-language class descriptions via contrastive learning | Beats 5-shot "adaptive ViT" methods using **zero** target-domain samples | Language semantics act as an additional generalization signal — genuinely new lever beyond anything in the CNN era |
| **DiVT** (Domain-Invariant Vision Transformer, WACV 2023) | Two auxiliary losses (single-domain concentration + domain-invariance) on top of ViT | SOTA on domain-generalized FAS protocols | Loss-based regularization can substitute for more training data |
| **Multimodal ViT + MAE** (Yu et al., arXiv:2302.05744) | Masked-autoencoder self-supervised pretraining across RGB+depth+IR inside a single ViT | Improves multimodal fusion vs. late-fused CNN branches | Relevant if you adopt a depth/IR sensor per your Section 5 hardware decision |
| **FM-ViT** (Flexible Modal ViT) | ViT that handles variable/missing modality combinations at inference | Generalizes across RGB-only, RGB+D, RGB+D+IR without retraining per combination | Useful if your deployment hardware varies across sites (phone vs. kiosk vs. webcam) |

### ViT for Deepfake / Forgery Detection

| Paper | Idea | Key result | Insight |
|---|---|---|---|
| **M2TR** (Wang et al., ICMR 2022, arXiv:2104.09770) | Multi-scale patch transformer + frequency-domain filters + cross-modality fusion block | Outperforms prior SOTA by clear margins; released SR-DF dataset (4,000 videos, more compression-robust than FF++) | Concrete, working implementation of the "spatial + frequency dual-branch" architecture proposed in Section 6 |
| **Swin-Y-Net** (Khalid et al.) | Swin Transformer encoder + U-Net decoder producing a segmentation mask before classification | AUC ~0.99 on Celeb-DF and FaceForensics++ | Segmentation-style supervision (similar in spirit to pixel-wise supervision in FAS) generalizes across DeepFakes, FaceSwap, Face2Face, FaceShifter, NeuralTextures |
| Plain Swin Transformer baselines | Swin vs. VGG16/ResNet18/AlexNet on image-based deepfake classification | Swin outperforms CNN baselines (e.g., 71.29% vs. lower CNN accuracy on a harder dataset) | Even without fusion tricks, transformer backbones alone beat CNNs on this task |

### ViT + Physiological (rPPG) Fusion — your most novel angle

This is the closest existing work to combining spatial-artifact detection with physiological liveness in a single attention-based model — worth framing as your core "gap to fill" if you want a genuinely novel contribution.

- **TransRPPG** (Yu et al., IEEE SPL 2021, arXiv:2104.07419) — the first "pure rPPG transformer." Builds multi-scale spatial-temporal maps (MSTmaps) from facial skin and background regions, then applies a transformer to mine global relationships within the MSTmaps for a binary liveness prediction, specifically for **3D mask attacks**. Notably lightweight (547K parameters, 763M FLOPs) — explicitly designed for mobile deployment, which matters for your attendance-system latency budget. However, it is **rPPG-only** — it does not fuse spatial appearance features, so it doesn't fully answer your question.
- **"Face Spoofing Detection using Swin Transformer and rPPG Signal" ("Deep Guard")** — this is the direct answer to your question: a Swin Transformer branch extracts spatial/texture features (skin reflection, edge smoothness, texture consistency) while a parallel rPPG module extracts heartbeat-based color-variation features from consecutive aligned frames; the two feature sets are fused (concatenation + dimensionality reduction, e.g., PCA, or fully-connected fusion layers) before final classification. Reported as delivering high precision and real-time performance, explicitly framed as more robust than either modality alone because appearance and physiological cues fail against *different* attack types (screens/masks vs. high-fidelity photos, respectively).
- **MFLD-RSTF** (Yadav et al., 2024) — a related multimodal architecture (not pure ViT, but conceptually the same fusion idea) combining rPPG with deep spatio-temporal features, validated on NUAA/SiW/Replay-Attack with reported improvements over prior methods in both intra- and cross-database testing.
- **The Yu et al. joint benchmark paper (arXiv:2208.05401)** you already have in your list explicitly calls this fusion direction out as future work: exploring more advanced multi-task learning and multi-modal fusion strategies, and exploiting semantic facial motion cues and contextual dynamic artifacts alongside rPPG for joint spoofing and forgery detection — i.e., a transformer-based spatial+rPPG+forgery-artifact joint model is an **explicitly identified open gap**, not something already solved.

---

## 8. Proposed Contributions

### 8.1 Architecture-level contribution
<img width="2720" height="2560" alt="unified_liveness_deepfake_architecture" src="https://github.com/user-attachments/assets/1944b0ee-938f-468c-b980-09ca8fcfc694" />

**Core proposal: a three-branch fusion transformer, trained and evaluated jointly on physical presentation attacks and digital deepfakes.**

- **Spatial branch** — CLIP-pretrained ViT (per FLIP's finding that vision-language pretraining generalizes better than plain ImageNet pretraining). Catches blending artifacts, print/screen texture, moiré.
- **Frequency branch** — FFT/DCT feature extraction (per M2TR). Catches GAN upsampling artifacts and screen-replay banding invisible in the spatial domain.
- **Physiological branch** — an rPPG transformer using multi-scale spatial-temporal maps (per TransRPPG). Catches absence of pulse in masks/photos/screens.

---

## 9. Reading List (references discussed)

1. Rössler et al., "FaceForensics++: Learning to Detect Manipulated Facial Images," ICCV 2019. arXiv:1901.08971
2. Huang et al., "A Survey on Deep Learning-based Face Anti-Spoofing," APSIPA Transactions on Signal and Information Processing, 2024.
3. Yu et al., "Deep Learning for Face Anti-Spoofing: A Survey," IEEE TPAMI.
4. "Presentation attack detection: a systematic literature review," ACM Computing Surveys, 2024.
5. Yu, Cai, Li, Yang, Shi, Kot, "Benchmarking Joint Face Spoofing and Forgery Detection with Visual and Physiological Cues," IEEE TDSC 2024 (arXiv:2208.05401).
6. "Unified Physical-Digital Attack Detection Challenge," CVPR 2024.
7. "Paired-Sampling Contrastive Framework for Joint Physical-Digital Face Attack Detection," arXiv:2508.14980, 2025.
8. TransRPPG: "Remote Photoplethysmography Transformer for 3D Mask Face Presentation Attack Detection," IEEE SPL 2021.
9. "Oulu Remote-photoplethysmography Presentation Attacks Database (OR-PAD)," IJCV 2025.
10. Srivatsan, Naseer, Nandakumar, "FLIP: Cross-domain Face Anti-spoofing with Language Guidance," ICCV 2023 (arXiv:2309.16649).
11. Wang et al., "M2TR: Multi-modal Multi-scale Transformers for Deepfake Detection," ICMR 2022 (arXiv:2104.09770).
