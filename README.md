# Robust Face Liveness & Deepfake Detection for Attendance Systems
## Literature Review Outline & System Scoping Document

---

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

**Why this matters for your proposal:** None of the existing ViT+rPPG work (TransRPPG, Deep Guard, MFLD-RSTF) is evaluated against **deepfake/forgery attacks** — they're all pure presentation-attack (PAD) papers. Combining (a) a ViT/Swin spatial-artifact branch, (b) an rPPG physiological branch, and (c) training/evaluation across *both* physical PAs and digital deepfakes in one unified benchmark would sit at the intersection of three separate research threads that currently don't overlap — a legitimate, citable gap for your contribution statement.

---

## 8. Proposed Contributions

This section translates the identified research gaps (Sections 2, 7, 7a) into a concrete set of contributions for the proposal/thesis. Nothing here is a single existing paper — each item closes a specific, citable gap identified across the FAS survey, the joint-benchmark paper, and the ViT literature.

### 8.1 Architecture-level contribution
<img width="2720" height="2560" alt="unified_liveness_deepfake_architecture" src="https://github.com/user-attachments/assets/1944b0ee-938f-468c-b980-09ca8fcfc694" />


**Core proposal: a three-branch fusion transformer, trained and evaluated jointly on physical presentation attacks and digital deepfakes.**

- **Spatial branch** — CLIP-pretrained ViT (per FLIP's finding that vision-language pretraining generalizes better than plain ImageNet pretraining). Catches blending artifacts, print/screen texture, moiré.
- **Frequency branch** — FFT/DCT feature extraction (per M2TR). Catches GAN upsampling artifacts and screen-replay banding invisible in the spatial domain.
- **Physiological branch** — an rPPG transformer using multi-scale spatial-temporal maps (per TransRPPG). Catches absence of pulse in masks/photos/screens.

**What is actually novel:** no existing paper fuses all three branches into one model trained across *both* attack families simultaneously — each existing paper (TransRPPG, Deep Guard, M2TR, FLIP) covers at most two of these three elements. The three-way fusion, applied to a joint physical+digital label space, is the architectural contribution.

**Fusion mechanism:** combine the weighted batch/layer normalization strategy from the Yu et al. joint benchmark paper (arXiv:2208.05401) — to prevent one modality from dominating — with the cross-modal contrastive alignment idea from FLIP (arXiv:2309.16649), aligning modalities in a shared embedding space rather than naive feature concatenation.

### 8.2 Training-strategy contributions

- **Joint multi-task labeling.** Replace binary live/spoof supervision with a 3-way (or richer) label space — live / physical-attack / digital-attack — so the model learns *what specifically* is wrong rather than only *that* something is wrong. Directly addresses the "arbitrary, unfaithful spoof cue" problem identified in the Yu et al. TPAMI survey.
- **Domain generalization by construction.** Bake in meta-learning (query/support domain splits) or explicit feature disentanglement from the outset rather than treating generalization as a post-hoc evaluation concern — every survey reviewed identifies this as the field's largest unsolved problem.
- **Addressing the physical/digital data imbalance.** Deepfakes are cheap to generate at scale; physical presentation attacks are expensive to collect — a gap the Yu et al. survey explicitly flags as harmful to multi-task representation learning. Proposed fix: a generative simulation module (in the spirit of "Joint Physical-Digital Facial Attack Detection via Simulating Spoofing Clues") to synthetically balance classes rather than naive upsampling/downsampling.

### 8.3 Framework-level deliverables (things that need to be built, not just designed)

| Component | Why it doesn't exist yet | What to build |
|---|---|---|
| Unified benchmark dataset | Existing joint benchmarks (Yu et al., 2022) exclude real-time virtual-camera injection attacks | Merge OULU-NPU / CelebA-Spoof / SiW-M (physical) with FaceForensics++ / Celeb-DF (digital), plus a newly captured set of virtual-camera-injection samples |
| Attendance-specific evaluation protocol | No FAS benchmark simulates a queue-based, low-latency, variable-hardware attendance scenario | New protocol: cross-device, cross-lighting, sub-2-second inference budget, plus an "unknown attack" holdout split |
| Lightweight/mobile variant | ViT-based methods are too heavy for real-time queues (ViTranZFAS: 85.8M params) | Knowledge-distill the three-branch model into a compact student network (TransRPPG-scale: <1M params) for deployment |
| Interpretability output | Nearly every FAS survey reviewed flags explainability as missing | Spoof-region localization map via a multi-task auxiliary head on the same model (see Section 8.3.1) — not a separate network — so a human reviewer can see *why* a face was flagged, important for attendance systems where false rejections need manual review |

#### 8.3.1 Spoof localization as a multi-task auxiliary head (not a second model)

Spoof localization should be implemented as an **auxiliary output head on the same shared backbone**, not as a separately trained model. This is both cheaper and, per the pixel-wise supervision literature, actively improves the primary classifier rather than merely adding an explainability side-output.

**Why one model, not two:**
- Pixel-wise/mask supervision is a training signal, not a downstream task — the same backbone features used for classification can simultaneously drive a small localization decoder in a single forward pass. This is how George & Marcel's cascaded confidence-map approach, CDCN/pyramid-supervision depth-map decoders, and Swin-Y-Net's segmentation-then-classification design all work.
- The localization signal acts as a **regularizer**: forcing the model to explain *where* the spoof is prevents it from latching onto unfaithful shortcut patterns (e.g., screen bezel) that plain binary-loss training is prone to.
- A shared representation means classification and localization share the same generalization behavior — training two separate models would require validating two independent generalization gaps that might disagree on ambiguous cases.
- The added inference cost is small (a lightweight conv/deconv decoder on top of the existing backbone), which matters given the edge-deployment latency constraints already identified for this project.

**Proposed architecture:**

```
Input face crop
      │
Shared backbone (three-branch fusion, per Section 8.1)
      │
      ├──> Localization head (small conv/deconv decoder) → binary spoof mask / depth map
      │
      └──> Classification head (pooling + FC) → live / physical-attack / digital-attack
```

**Combined training objective:**

```
L_total = λ1 · L_classification + λ2 · L_localization
```

where `L_localization` is a pixel-wise binary cross-entropy (for mask supervision) or L1/L2 loss (for depth supervision) against ground-truth, and `λ1`/`λ2` balance the two objectives.

**Binary mask vs. depth map — recommendation:** favor binary mask supervision over depth-map supervision as the default for this project. Depth maps are more physically intuitive but fail on inherently 3D attacks (3D masks, mannequins, since these have genuine facial depth indistinguishable from a live face) and are costly to generate accurate ground truth for. Binary masks are cheaper to generate, generalize across more attack types, and can be extended to ternary labels (real / spoofed / uncertain background) to better handle partial attacks such as a phone screen only covering part of the frame — the most likely real-world case for this project's attendance-fraud scenario. If 3D mask attacks are in-scope, pair localization with the rPPG branch rather than relying on depth-based localization alone, since masks lack a pulse signal that depth maps cannot capture.

**Known limitations to scope explicitly (not oversell in the proposal):**
- Localization quality tracks the same generalization problem as detection accuracy — on unseen attack types (e.g., partial print, half-mask), predictions become chaotic and regions beyond the actual spoof medium get incorrectly flagged.
- Naive pixel-wise supervision treats all patches with equal weight, which biases the model since subtle clues like moiré patterns vary in intensity across regions — mitigate with a learnable attention module before computing the localization loss, rather than plain unweighted pixel-wise loss.

**Label-generation work required (this is the real added effort, not a second model):**
- Physical attacks: binary masks can often be derived heuristically (e.g., whole-face-region "spoof" label for print/replay) or via a pseudo-depth generator applied only to genuine faces, with spoof samples assigned a flat/zero depth map.
- Digital attacks (deepfakes): if generating your own manipulated samples, masks come for free from knowing which pixels were blended/swapped; if using existing deepfake datasets, approximate via face-parsing plus the known manipulation region.

### 8.4 Concrete improvements over specific prior work

- **Over TransRPPG:** add spatial and frequency branches it lacks, so the model is no longer blind to attacks with no exploitable pulse signal (e.g., a real photo of a living person, which also lacks a depth cue).
- **Over Deep Guard (Swin + rPPG):** extend from presentation-attack-only to joint PA + deepfake detection, and replace plain feature concatenation with domain-generalization-aware fusion training.
- **Over FLIP:** FLIP is spatial-only; add the physiological and frequency branches, and extend its language-guided contrastive supervision to describe *attack type* ("a photo of a screen replay attack" vs. "a photo of a face-swap deepfake") rather than only live/spoof — a richer supervision signal using the same underlying mechanism.
- **Over M2TR:** M2TR has no physiological branch and is not evaluated against physical presentation attacks at all; extending it into the three-branch design directly closes that gap.

### 8.5 Systems-level contributions

- **Adaptive branch weighting at inference.** Skip the rPPG branch (which needs several seconds of video) when latency is tight, falling back to spatial+frequency only — a runtime-adaptive design that no reviewed paper addresses, since none are built for latency-constrained deployment.
- **Federated/privacy-preserving training pipeline**, relevant if attendance data is sensitive: each site/department trains locally and aggregates centrally without sharing raw biometric data, per the privacy-preserving training direction flagged in the Yu et al. TPAMI survey.
- **Continual/few-shot update mechanism** so the model can absorb a newly discovered attack type (e.g., a new deepfake tool) without full retraining, using the meta-learning approach from zero/few-shot FAS literature (Qin et al. and related work).

### 8.6 Suggested prioritization / roadmap

1. **Phase 1 (proof of concept):** spatial + frequency two-branch fusion, trained jointly on merged physical + digital attack datasets. This alone is a legitimate, publishable contribution, since no existing work performs joint training at this scope.
2. **Phase 2:** add the rPPG branch and cross-modal fusion — the most novel piece of the proposed architecture.
3. **Phase 3:** domain generalization and distillation, to make the model deployment-feasible under the attendance-system latency/hardware constraints from Section 5.
4. **Stretch goal:** virtual-camera-injection attack collection and evaluation — the hardest, most under-studied threat class identified in Section 2; reasonable to present as a partially-solved future-work chapter even if not fully addressed in v1.

---

## 9. Reading List (references discussed)

1. Rössler et al., "FaceForensics++: Learning to Detect Manipulated Facial Images," ICCV 2019. arXiv:1901.08971
2. Huang et al., "A Survey on Deep Learning-based Face Anti-Spoofing," APSIPA Transactions on Signal and Information Processing, 2024.
3. Yu et al., "Deep Learning for Face Anti-Spoofing: A Survey," IEEE TPAMI.
4. "Presentation attack detection: a systematic literature review," ACM Computing Surveys, 2024.
5. Yu, Cai, Li, Yang, Shi, Kot, "Benchmarking Joint Face Spoofing and Forgery Detection with Visual and Physiological Cues," IEEE TDSC 2024 (arXiv:2208.05401).
6. "Unified Physical-Digital Attack Detection Challenge," CVPR 2024.
7. "Paired-Sampling Contrastive Framework for Joint Physical-Digital Face Attack Detection," arXiv:2508.14980, 2025.
8. Mixture-of-Experts unified physical/digital face attack detection paper, 2024.
9. TransRPPG: "Remote Photoplethysmography Transformer for 3D Mask Face Presentation Attack Detection," IEEE SPL 2021.
10. "Oulu Remote-photoplethysmography Presentation Attacks Database (OR-PAD)," IJCV 2025.
11. Liu et al., CNN-RNN rPPG + depth-map estimation for face anti-spoofing, CVPR 2018.
12. "Generalized Face Liveness Detection via De-fake Face Generator," arXiv:2401.09006.
13. "A Compact Deep Learning Model for Face Spoofing Detection," arXiv:2101.04756.
14. George & Marcel, "On the Effectiveness of Vision Transformers for Zero-shot Face Anti-Spoofing," IJCB 2021 (arXiv:2011.08019).
15. Srivatsan, Naseer, Nandakumar, "FLIP: Cross-domain Face Anti-spoofing with Language Guidance," ICCV 2023 (arXiv:2309.16649).
16. Yu et al., "Rethinking Vision Transformer and Masked Autoencoder in Multimodal Face Anti-spoofing," arXiv:2302.05744.
17. Liu et al., "FM-ViT: Flexible Modal Vision Transformers for Face Anti-Spoofing," IEEE TIFS, 2023.
18. Wang et al., "M2TR: Multi-modal Multi-scale Transformers for Deepfake Detection," ICMR 2022 (arXiv:2104.09770).
19. Yu, Li, Wang, Zhao, "TransRPPG: Remote Photoplethysmography Transformer for 3D Mask Face Presentation Attack Detection," IEEE SPL 2021 (arXiv:2104.07419).
20. "Face Spoofing Detection using Swin Transformer and rPPG Signal" ("Deep Guard"), IJRASET, 2025.
21. Yadav et al., "MFLD-RSTF: Multimodal Face Anti-spoofing with rPPG and Deep Spatio-temporal Features," SIGMAA 2024 (Springer).


---

## 10. Immediate Next Steps

1. Read the Yu et al. joint benchmark paper in full — it is your closest prior work and its related-work section is itself a mini literature review you can mine for further citations.
2. Lock in the four scoping decisions in Section 5 — they determine the rest of your review's boundaries.
3. Build the comparison table in Section 4 as you read, rather than after — it becomes both your review's evidence table and your future baseline list.
4. Identify 2–3 candidate public datasets that combine or can be merged to cover both PA and deepfake attacks (e.g., OULU-NPU/CelebA-Spoof + FaceForensics++) for your own experiments.
