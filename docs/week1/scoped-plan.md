# Digital Deepfake Detection — Image Processing Track
## Scoped Plan: Face-Swap + Reenactment Deepfakes (Blending, Boundary, Temporal, and Screen-Replay Cues)

> **Scope note.** The parent project (*Robust Face Liveness & Deepfake Detection for Attendance Systems*) splits into physical-attack detection, physiological/rPPG liveness, and digital/synthetic attack detection. This document covers **only my slice**: the **image-processing side of the two "Digital / Synthetic" rows** in the parent taxonomy —
> 1. **Face-swap deepfake (offline, played to camera)** — blending artifacts + screen-replay artifacts combined
> 2. **Reenactment deepfake (Face2Face / NeuralTextures-style driven video)** — subtle boundary/texture artifacts, temporal flicker
>
> Out of scope here (owned by other parts of the project): the rPPG/physiological branch, real-time virtual-camera injection, and physical-only presentation attacks (print/mask). In the parent doc's proposed three-branch architecture (Section 6/8.1), this document corresponds to the **spatial + frequency branches**, evaluated specifically against face-swap and reenactment attacks.

---

## 1. Problem Understanding

Face-swap and reenactment are the two classical "digital/synthetic" attack families, and they stress the detector in different ways:

- **Face-swap** replaces the identity: a source face is composited onto a target body/background. Almost every face-swap pipeline — from the original autoencoder-based DeepFakes tool through GAN-based swaps — ends with a **blending step** that stitches a generated face region back into the target frame. That stitch leaves a boundary.
- **Reenactment** keeps the identity but transfers expression/pose from a driver video (Face2Face) or re-renders the mouth/expression region using a learned neural texture (NeuralTextures). There is no full-face swap, so the tell-tale blending boundary is weaker or absent; instead, artifacts show up as **local texture warping around eyes/mouth** and **frame-to-frame inconsistency**, since each frame is often synthesized with limited temporal awareness.

The added wrinkle in this project's threat model is the **"played to camera" condition**: an attacker doesn't inject the deepfake file directly — they display it on a phone/monitor and let the attendance system's camera recapture it. That recapture step (screen → camera) stacks a second, independent artifact family (moiré, bezel, refresh-rate banding, gamma/color shift, re-compression) **on top of** whatever native forgery artifacts the deepfake generation left. A detector tuned only on "clean" digital deepfake files (as almost all public benchmarks are) is not automatically robust to this combination — this is the central problem this scope needs to address, not just "can we detect FaceForensics++-style forgeries."

Constraints inherited from the parent doc's scoping section (Section 5) that shape this work:
- **Hardware:** standard RGB camera only — no depth/IR assumption for this branch.
- **Interaction mode:** passive, short-clip analysis (a few frames to ~1 second), not active challenge-response.
- **Latency budget:** sub-1–2 seconds per person, which rules out anything needing many seconds of temporal context.

---

## 2. Types of Attacks (my scope)

| Attack | Generation mechanism | What changes in the pixels | Primary detectable artifact | Screen-replay interaction |
|---|---|---|---|---|
| **Face-swap deepfake, played to camera** | Identity A's face is generated/warped and composited onto identity B's video (classic autoencoder DeepFakes, FaceSwap, or GAN-based swaps like SimSwap/FaceShifter-style methods), then the result is shown on a screen and recaptured | Full facial region replaced; a blending mask stitches generated content into the original frame; then a second capture (screen→camera) is layered on top | Blending-boundary discontinuity, GAN upsampling ("checkerboard") fingerprints, identity-consistency mismatch between face and context — **plus** moiré, bezel edges, refresh-rate banding from the recapture | Recapture can **mask** subtle native artifacts (screen resolution/anti-aliasing smooths them) while **adding** a strong new periodic-frequency fingerprint (moiré) — the two artifact families compound rather than simply add |
| **Reenactment deepfake (Face2Face / NeuralTextures-style driven video)** | Same identity, but expression/pose is driven by a second (attacker's) performance; Face2Face re-renders a parametric face model, NeuralTextures re-renders via a learned neural texture over a 3D proxy | Localized re-rendering around eyes/mouth/jaw; no full-face blending boundary; per-frame synthesis with limited temporal consistency | Subtle texture/boundary artifacts around the re-rendered region, unnatural micro-expression timing, **temporal flicker** (frame-to-frame instability not present in real video), audio-visual (lip-motion) mismatch | If also displayed on a screen and recaptured, moiré and banding are superimposed on an already-subtle boundary signal — this case is expected to be **harder** than face-swap+replay because the native artifact is weaker to begin with |

---

## 3. Image / Video Fundamentals & Representations

The features in Section 4 all come from a small set of underlying representations. Getting these right matters more than the model architecture on top of them:

- **Spatial / pixel domain (RGB).** Face detection + alignment (e.g., MTCNN/dlib landmarks) crops and normalizes the face before anything else. Blending-boundary and texture-warping cues live here. — [Face Detection with MTCNN using Python (YouTube)](https://www.youtube.com/watch?v=-0qQ_ukDbl8)
- **Frequency domain.** 2D FFT and block-wise DCT of the face crop. GAN decoders built from transposed convolutions leave periodic upsampling artifacts that show up as peaks in the **azimuthally-averaged power spectrum**; JPEG/H.264 compression and moiré both also have strong, distinctive frequency signatures — this is *the* representation that lets a detector separate "GAN artifact" from "recapture artifact" instead of confusing the two. — [Introduction to Image Processing with 2D Fourier Transform (YouTube)](https://www.youtube.com/watch?v=tlwIWjeuu8U) · [JPEG DCT, Discrete Cosine Transform — Computerphile (YouTube)](https://www.youtube.com/watch?v=Q2aEzeMDHMA) · [Transposed Convolutions Explained (checkerboard/upsampling artifacts) (YouTube)](https://www.youtube.com/watch?v=xoAv6D05j7g)
- **Color space.** YCbCr is preferred over RGB for compression- and moiré-related cues, since chroma subsampling and screen-camera aliasing behave differently across luma vs. chroma channels; phase-spectrum-based methods use it too. — [Colourspaces — Computerphile (YouTube)](https://www.youtube.com/watch?v=LFXN9PiOGtY)
- **Temporal / video representation.** Short frame stacks or optical flow between consecutive frames, rather than single-frame analysis — necessary for catching flicker and unnatural mouth/expression dynamics, which are the *primary* cue for reenactment specifically since spatial boundary cues are weak there. — [Optical Flow — Computerphile (YouTube)](https://www.youtube.com/watch?v=5AUypv5BNbI)
- **Pixel-wise mask representation.** A grayscale "blending map" (as in Face X-ray) or segmentation mask that marks *where* the forged region is, used as an auxiliary supervision signal rather than only a binary label — this is the same idea the parent doc's Section 8.3.1 proposes for the whole system, and it applies directly to this branch.
- **Compression-aware representation.** Since attendance-system video will always be compressed (and recapture adds a second compression pass), features need to be evaluated at multiple compression levels (raw / light H.264 / heavy H.264), not just on raw video — this is standard practice in the FaceForensics++ benchmark protocol. — [H.264 Video Compression explained (YouTube)](https://www.youtube.com/watch?v=1PMqXdWJHNs)
- **Screen-replay / recapture artifact (moiré).** The interference pattern from a camera sensor grid sampling a screen's pixel grid — the core "recapture" cue referenced throughout Sections 4–9. — [The Moiré Effect Explained (YouTube)](https://www.youtube.com/shorts/H9eIAtwbk3I)

---

## 4. Feature Set

| Feature category | Concrete features | Targets face-swap | Targets reenactment | Robust to screen-replay? |
|---|---|---|---|---|
| **Blending-boundary** | Face X-ray–style grayscale boundary map; edge-consistency around face contour | Strong — this is the native cue | Weak — reenactment rarely has a hard full-face boundary | Degrades — recapture blur/moiré can wash out a subtle boundary |
| **Identity/appearance consistency** | Face-vs-context discrepancy, identity embedding mismatch between face region and surrounding hair/ears/neck | Strong | Not applicable (same identity) | Mostly preserved — identity mismatch is a semantic cue, less sensitive to pixel-level recapture noise |
| **GAN "fingerprint" texture** | Checkerboard/upsampling artifacts, local texture statistics | Moderate–strong (depends on generator) | Weak–moderate (NeuralTextures artifacts are subtler) | Degrades — recapture resampling can overwrite fine fingerprint texture |
| **Frequency-domain (DCT/FFT)** | Azimuthal power-spectrum peaks, DCT coefficient statistics, phase-spectrum anomalies | Strong | Moderate | **Double-edged**: native GAN frequency fingerprint is fragile under recapture, but the recapture itself adds a strong, learnable moiré/banding signature — a frequency branch can pivot to detecting *that* instead |
| **Temporal / motion** | Optical-flow inconsistency near the boundary, frame-to-frame prediction variance ("flicker score"), lip-sync/mouth-motion embedding distance (LipForensics-style) | Moderate | **Strong — primary cue** | Largely preserved — motion-level cues survive recapture better than pixel-level texture cues, since they operate on relative frame-to-frame change |
| **Screen-replay / recapture** | Moiré energy (Haar-wavelet subband analysis), bezel/edge detection, refresh-rate banding, gamma/color-gamut shift, double-compression signature | Applies whenever a screen is involved | Applies whenever a screen is involved | N/A — this *is* the replay cue |

The practical takeaway for this branch: **no single feature family covers both attack types under recapture.** Boundary and GAN-fingerprint cues are the strongest native signal for face-swap but degrade first under recapture; motion/temporal cues are the strongest signal for reenactment and hold up best under recapture. This directly motivates a spatial+frequency+temporal fusion rather than a single-cue detector — a smaller, three-cue version of the parent doc's full three-branch (spatial/frequency/physiological) design, swapping the physiological branch for a lightweight temporal-motion branch since that's out of scope here.

---

## 5. Key Papers (this branch's core reading list)

| Paper | Venue / Year | Idea | Reported result (as stated by authors) | Relevance to this scope |
|---|---|---|---|---|
| **[FaceForensics++](https://arxiv.org/abs/1901.08971)** (Rössler et al., arXiv:1901.08971) | ICCV 2019 | Foundational benchmark: 1,000 real YouTube videos manipulated with four methods (DeepFakes, Face2Face, FaceSwap, NeuralTextures), released at three compression levels; Xception baseline | Domain-specific training clearly outperforms generic classifiers, especially under compression | The benchmark this whole sub-area is built on — covers both my attack rows directly (DeepFakes/FaceSwap = face-swap row, Face2Face/NeuralTextures = reenactment row) |
| **[Face X-ray](https://arxiv.org/abs/1912.13458)** (Li et al., arXiv:1912.13458) | CVPR 2020 (Oral) | Predicts a grayscale "blending boundary" map instead of a binary label; trained by synthetically blending real images, so it needs no knowledge of the specific manipulation method | Generalizes to unseen manipulation techniques better than classifiers trained on manipulation-specific artifacts | Direct match for the face-swap row's "blending artifacts" cue; also the template for the parent doc's auxiliary localization head (Section 8.3.1) |
| **[F3-Net / "Thinking in Frequency"](https://arxiv.org/abs/2007.09355)** (Qian et al., arXiv:2007.09355) | ECCV 2020 | Two-stream network combining frequency-aware decomposed image components and local frequency statistics, using DCT as the frequency transform | Outperformed prior methods on FaceForensics++ across all compression levels, with the largest margin on low-quality/compressed video | Grounds the frequency-branch design in Section 6 of the parent doc; the compression robustness result matters directly for an attendance-system deployment |
| **[LipForensics](https://arxiv.org/abs/2012.07657)** (Haliassos et al., arXiv:2012.07657) | CVPR 2021 | Pretrains a spatio-temporal network on lipreading, then fine-tunes on real/fake mouth-motion embeddings — targets *semantic* motion irregularities instead of low-level pixel artifacts | Video-level cross-dataset AUC of 82.4% (Celeb-DF v2), 73.5% (DFDC), 97.1% (FaceShifter), 97.6% (DeeperForensics) when trained only on FaceForensics++ | The strongest published case for a temporal/motion cue as the primary reenactment detector — directly relevant since reenactment's boundary signal is weak |
| **[Self-Blended Images (SBI)](https://arxiv.org/abs/2204.08376)** (Shiohara & Yamasaki, arXiv:2204.08376) | CVPR 2022 (Oral) | Generates training data by blending a real image with a transformed copy of *itself*, reproducing blending-boundary and source/target statistical artifacts without needing any real forged videos at all | Outperformed prior state-of-the-art cross-dataset evaluation by 4.90 and 11.78 points AUC on DFDC and DFDCP respectively | Most cited "generalization" recipe in the field right now; a natural fit for the parent doc's "generative simulation to balance data" idea (Section 8.2) — extendable to synthesize recapture artifacts too (see Section 11 below) |
| **[M2TR](https://arxiv.org/abs/2104.09770)** (Wang et al., arXiv:2104.09770) | ICMR 2022 | Multi-scale patch transformer combined with frequency-domain filters and a cross-modality fusion block; also released the SR-DF dataset | Reported to outperform prior state-of-the-art by a clear margin, with SR-DF built to be more compression-robust than FF++ | Already flagged in the parent doc as the concrete working example of the spatial+frequency dual-branch design proposed there |
| **[Swin-Y-Net (SWYNT)](https://ieeexplore.ieee.org/document/10089585/)** (Khalid, Akbar & Gul) | ICRAI 2023 | Swin Transformer encoder + U-Net decoder, producing a segmentation mask before final classification | AUC ≈ 0.99 reported on Celeb-DF and FaceForensics++ | Already flagged in the parent doc; segmentation-before-classification mirrors the Face X-ray localization idea in a transformer backbone |
| **["Through the Lens" / DeepMoiréFake benchmark](https://arxiv.org/abs/2510.23225)** (arXiv:2510.23225) | NeurIPS 2025 (Datasets & Benchmarks) | First systematic benchmark of deepfake detectors under real screen-recapture conditions; built a 12,832-video dataset (802 source videos × 4 screens × 2 phones × 2 conditions) by physically photographing deepfake playback off real screens with smartphones, drawing source videos from Celeb-DF, DFD, DFDC, UADFV, and FF++ | Across 15 top-performing detectors, moiré-induced distortion degraded performance by as much as 25.4% (accuracy-style metrics) and 33.1 percentage points (some AUC-style comparisons) relative to clean playback | **This is the closest existing work to the exact combined attack this scope is about** — direct evidence that "blending artifacts + screen-replay artifacts combined" is a real, current, under-solved research gap, not a hypothetical one |
| **[LNCLIP-DF / GenD](https://arxiv.org/abs/2508.06248)** (Yermakov et al., arXiv:2508.06248) | 2025 | Fine-tunes only the Layer Normalization parameters (≈0.03% of weights) of a frozen CLIP/DINO vision encoder, plus L2-normalized hyperspherical feature space and metric learning, instead of building new architecture on top | State-of-the-art average cross-dataset AUROC across 14 benchmark datasets spanning 2019–2025, evaluated as the broadest such comparison to date, per the authors | Represents where the field's spatial branch is currently heading — foundation-model adaptation over from-scratch CNN/ViT training; relevant if the group later swaps the spatial branch's backbone |
| **[FTCN](https://arxiv.org/abs/2108.06693)** (Zheng et al., arXiv:2108.06693) — cited via cross-dataset comparisons in follow-up work | ICCV 2021 | Fully temporal convolutional network exploiting inter-frame inconsistency across an entire video, rather than per-frame appearance | Reported around 86.9% AUC on Celeb-DF v2 in later cross-dataset comparison tables, competitive among artifact-based (non-blending-synthesis) methods | A second temporal-cue reference point alongside LipForensics, useful for the ablation the parent doc's evaluation protocol (Section 7) asks for |

---

## 6. Architecture Comparison

| Architecture family | Representative paper(s) | Core mechanism | Strength | Weakness for this scope |
|---|---|---|---|---|
| CNN spatial baseline | Xception (FaceForensics++ baseline) | End-to-end CNN classifier on the aligned face crop | Simple, fast, still a fair in-dataset baseline | Overfits to dataset-specific artifacts; weak cross-dataset and weak under recapture |
| Boundary/blending-focused | Face X-ray, Self-Blended Images | Predict or supervise on a blending-boundary map; train on synthetically blended positives | Best generalization for face-swap-style attacks; SBI needs no real forged data | Assumes a blending step exists — a weaker fit for reenactment, where the boundary is subtle or absent |
| Frequency-domain | F3-Net | Two-stream DCT-based decomposition + local frequency statistics | Best robustness to compression among single-cue methods | Frequency fingerprint of the *generator* can be scrambled by recapture — needs to be paired with a recapture-aware branch, not used alone |
| Spatial+frequency transformer | M2TR | Multi-scale patch transformer + frequency filters + fusion block | Concretely realizes the fusion idea the parent doc proposes | Heavier than a CNN; no explicit temporal/motion branch, so weaker on reenactment specifically |
| Segmentation-then-classification | Swin-Y-Net | Swin backbone + U-Net decoder producing a mask, then classifying | High reported in-dataset/cross-dataset numbers; built-in localization | Segmentation-mask ground truth is easy for face-swap, harder to define cleanly for reenactment's localized, subtle edits |
| Motion/temporal | LipForensics, FTCN | Learn semantic-level motion regularities (lipreading pretraining, full-video temporal convolutions) instead of pixel artifacts | Best generalization and best compression/recapture robustness for reenactment-style attacks | Needs a short video clip, not a single frame — costs latency, which matters against the sub-1–2 s budget |
| Foundation-model adaptation | LNCLIP-DF / GenD, DFD-FCG | Freeze most of a large pretrained vision-language encoder (CLIP/DINO), fine-tune a tiny fraction of parameters | Currently the strongest broad cross-dataset generalization reported in the literature | Backbone is large; needs distillation (per parent doc Section 8.3) before it fits a real-time attendance kiosk |

---

## 7. Feature/Cue Analysis

Putting Sections 4 and 6 together, the picture for this branch is fairly clean:

- **For face-swap (with or without recapture):** blending-boundary and identity-consistency cues do most of the work, backed up by frequency-domain analysis. Recapture doesn't remove these cues so much as it **shifts** them — a boundary that was pixel-sharp in the digital file becomes fuzzier but still statistically present after screen-camera resampling, and the frequency branch picks up a *second*, independent recapture-specific signature (moiré) it can learn to key on instead of or alongside the native GAN fingerprint.
- **For reenactment (with or without recapture):** boundary/blending cues are inherently weaker, so temporal/motion cues (mouth dynamics, frame-to-frame flicker) carry more of the detection burden. This is the empirical basis for LipForensics' result: a model trained purely on motion semantics, with no pixel-level artifact modeling at all, still generalizes across FaceForensics++'s Face2Face/NeuralTextures methods, Celeb-DF, DFDC, and FaceShifter. Motion cues are also comparatively robust to recapture, since they're computed from *relative* frame-to-frame change rather than absolute pixel statistics.
- **The open, under-studied interaction** is what happens when both effects are stacked at once — a subtle reenactment boundary artifact plus moiré plus a second compression pass. The DeepMoiréFake benchmark (Section 5) is the first work to actually measure this for existing detectors, and it shows large performance drops across the board; it does **not** yet propose a detector purpose-built for the combination, which is exactly the gap this scope should target.

---

## 8. Dataset Comparison

| Dataset | Year | Content | Real / fake counts (as released) | Manipulation methods included | Recapture/screen-replay variant? |
|---|---|---|---|---|---|
| **FaceForensics++** | 2019 | YouTube face videos, 3 compression levels (raw / c23 / c40) | 1,000 real videos; 1,000 fake videos per method | DeepFakes, Face2Face, FaceSwap, NeuralTextures (+ FaceShifter added later) — covers both attack rows directly | No |
| **Celeb-DF (v2)** | 2020 | Celebrity YouTube videos, higher visual quality than FF++ | 590 real, 5,639 fake | Improved face-swap synthesis pipeline | No |
| **DFDC** | 2020 | Paid, consenting actors under varied lighting/pose/background | ~128,154 clips total (~119k in the main training split; roughly 100k fake / 19k real) from 3,426 actors | Eight manipulation techniques, including classic encoder-decoder deepfakes, GAN-based swaps, and non-learned methods | No |
| **DFDC Preview** | 2019 | Earlier, smaller release ahead of the full DFDC | ~5,214 clips | Two facial-modification methods | No |
| **FaceShifter** | 2020 (added to FF++) | Two-stage high-fidelity face-swap method | 1,000 fake videos | Face-swap | No |
| **DeeperForensics-1.0** | 2020 | Large-scale real-world forgery dataset with diverse perturbations | Large-scale (perturbation-focused) | Face-swap style, plus synthetic real-world distortions (not screen recapture specifically) | Perturbations yes, but not physical screen recapture |
| **DeepMoiréFake (DMF)** | 2025 | Built by physically photographing deepfake playback off real screens with smartphones | 12,832 videos, 35.64 hours total, from 802 source videos × 4 screens × 2 phones × 2 conditions | Draws source deepfakes from Celeb-DF, DFD, DFDC, UADFV, FF++ | **Yes — this is the only dataset purpose-built for exactly this scope's combined attack case** |

**Gap for the group's contribution:** there is no dataset combining (a) both face-swap *and* reenactment methods, (b) at multiple compression levels, *and* (c) recaptured off a real screen — DMF is the closest but is a benchmark/robustness-testing set built by recapturing existing deepfakes rather than a training-ready dataset spanning both attack types with full manipulation-method labels. This matches the parent doc's Section 8.3 "Unified benchmark dataset" gap almost exactly, scoped down to this branch.

---

## 9. Current SOTA / Strong Models

Three active paradigms, roughly in order of how much attention they're currently getting:

1. **Synthetic-blending training (backbone-agnostic).** Self-Blended Images and its extensions (e.g., frequency-enhanced variants) remain the strongest, cheapest way to get cross-dataset generalization: training data is synthesized from real faces alone, so the model never gets the chance to overfit to one generator's fingerprint. This is a *training strategy*, not an architecture, so it composes with anything else on this list.
2. **Foundation-model adaptation.** The newest wave (2025) freezes a large pretrained vision-language encoder (CLIP, DINO) and fine-tunes a very small fraction of parameters, reporting the best published average cross-dataset numbers across the widest set of benchmarks assembled so far (13–14 datasets spanning 2019–2025). This is where a from-scratch CNN/ViT spatial branch is likely to lose ground over the next couple of years.
3. **Motion/temporal-consistency methods.** LipForensics and FTCN remain the reference points for compression- and manipulation-agnostic detection of reenactment-style attacks specifically, because they target semantic motion rather than pixel statistics.

None of the three, as published, report results under the screen-recapture condition this scope cares about — the DeepMoiréFake benchmark evaluated existing top detectors (including several in this family) *after the fact* and found large, consistent degradation, which is the strongest available evidence that "SOTA on FF++/Celeb-DF/DFDC" does not imply "SOTA under the attendance-system's actual recapture conditions."

---

## 10. Limitations of Existing Work

- **Almost nothing is evaluated on recaptured (screen-replay) deepfakes.** The handful of works that do (DeepMoiréFake and related moiré-impact studies) treat it purely as a robustness *stress test* on existing detectors, not as a training condition — there is no detector in the literature purpose-built to fuse "native forgery cue" and "recapture cue" together.
- **Blending-boundary methods implicitly assume a hard blending step.** This is a strong assumption for face-swap but a poor fit for reenactment (Face2Face/NeuralTextures), where the edit is localized and the boundary is weak — explaining why the field needed a separate motion-based line of work (LipForensics) for reenactment-style forgeries in the first place.
- **Frequency-domain cues are exactly the artifact family recapture is most likely to destroy or overwrite.** A detector leaning heavily on GAN-upsampling frequency fingerprints (as F3-Net and similar methods do) risks losing its main signal precisely in the screen-replay scenario this project targets, while simultaneously being handed a *new*, learnable frequency signature (moiré) it isn't designed to use.
- **Cross-dataset generalization is still the field's largest unsolved problem in general** — reported in-dataset accuracy routinely reaches 95–99%, while naively-trained cross-dataset AUC can fall to the 60–80% range; recent synthetic-blending and foundation-model work narrows this gap but doesn't close it, and none of the closing work has been re-tested under recapture.
- **No unified dataset spans both attack types plus compression plus recapture together**, as noted in Section 8 — every existing benchmark isolates one or two of these axes at a time.

---

## 11. Open Research Questions (this branch)

1. **How do native forgery artifacts and screen-replay artifacts actually interact?** Do blending-boundary and moiré/banding signals compound (making detection easier — two independent tells) or partially cancel (recapture blur smoothing over the boundary), and does the answer differ for face-swap vs. reenactment given how different their native artifact strength already is?
2. **Can Self-Blended-Images-style synthetic training be extended to synthesize plausible recapture artifacts** (moiré, banding, gamma shift) alongside blending artifacts, so a detector can generalize to the combined attack without needing an expensive physically-recaptured training set the size of DeepMoiréFake?
3. **Do motion/temporal cues generalize better than spatial/frequency cues specifically under the recapture condition**, given that LipForensics-style motion features are already known to be more compression-robust — does that robustness extend to screen-replay, which is a different kind of degradation, or does frame-rate loss during recapture undermine motion cues in a way compression doesn't?
4. **What is the right latency/accuracy trade-off for a spatial+frequency+lightweight-motion fusion under a sub-1–2 second budget**, and can a foundation-model spatial branch (per Section 9's current SOTA direction) be distilled down far enough to fit that budget without losing the generalization that makes it attractive in the first place?
5. **Is a single shared model that jointly labels face-swap vs. reenactment vs. "screen-replay of either" actually better than two specialist branches gated by a screen/no-screen classifier** — i.e., is the combined attack best solved as one 4-way classification problem, or as a cheap upstream "is this a recapture?" check followed by attack-type-specific detection?

---

## 12. Reading List (papers referenced above, with direct links)

1. Rössler, Cozzolino, Verdoliva, Riess, Thies, Nießner — "FaceForensics++: Learning to Detect Manipulated Facial Images," ICCV 2019 — [arXiv:1901.08971](https://arxiv.org/abs/1901.08971)
2. Li, Bao, Zhang, Yang, Chen, Wen, Guo — "Face X-ray for More General Face Forgery Detection," CVPR 2020 — [arXiv:1912.13458](https://arxiv.org/abs/1912.13458)
3. Qian, Yin, Sheng, Chen, Shao — "Thinking in Frequency: Face Forgery Detection by Mining Frequency-aware Clues" (F3-Net), ECCV 2020 — [arXiv:2007.09355](https://arxiv.org/abs/2007.09355)
4. Haliassos, Vougioukas, Petridis, Pantic — "Lips Don't Lie: A Generalisable and Robust Approach to Face Forgery Detection" (LipForensics), CVPR 2021 — [arXiv:2012.07657](https://arxiv.org/abs/2012.07657)
5. Zheng, Bao, Chen, Zeng, Wen — "Exploring Temporal Coherence for More General Video Face Forgery Detection" (FTCN), ICCV 2021 — [arXiv:2108.06693](https://arxiv.org/abs/2108.06693)
6. Shiohara, Yamasaki — "Detecting Deepfakes with Self-Blended Images" (SBI), CVPR 2022 — [arXiv:2204.08376](https://arxiv.org/abs/2204.08376)
7. Wang, Guo, Zhu, et al. — "M2TR: Multi-modal Multi-scale Transformers for Deepfake Detection," ICMR 2022 — [arXiv:2104.09770](https://arxiv.org/abs/2104.09770)
8. Khalid, Akbar, Gul — "SWYNT: Swin Y-Net Transformers for Deepfake Detection," ICRAI 2023 — [IEEE Xplore](https://ieeexplore.ieee.org/document/10089585/) *(no arXiv preprint found)*
9. Authors (NeurIPS D&B 2025) — "Through the Lens: Benchmarking Deepfake Detectors Against Moiré-Induced Distortions" (DeepMoiréFake), NeurIPS 2025 Datasets & Benchmarks Track — [arXiv:2510.23225](https://arxiv.org/abs/2510.23225)
10. Yermakov et al. — "Deepfake Detection that Generalizes Across Benchmarks" (LNCLIP-DF / GenD), 2025 — [arXiv:2508.06248](https://arxiv.org/abs/2508.06248)
11. Li, Yang, Sun, Qi, Lyu — "Celeb-DF: A Large-scale Challenging Dataset for Deepfake Forensics," CVPR 2020 — [arXiv:1909.12962](https://arxiv.org/abs/1909.12962)
12. Dolhansky, Bitton, Pflaum, Lu, Howes, Wang, Canton Ferrer — "The DeepFake Detection Challenge (DFDC) Dataset," 2020 — [arXiv:2006.07397](https://arxiv.org/abs/2006.07397)

---

## 13. How This Plugs Back Into the Team's Architecture

Mapping this document onto the parent proposal's three-branch design (Section 6/8.1):

- **Spatial branch** → covered here via blending-boundary and identity-consistency features (Face X-ray, SBI-style training).
- **Frequency branch** → covered here via DCT/FFT-based features (F3-Net-style), extended to also target the recapture-specific moiré/banding signature, not just the GAN-upsampling fingerprint.
- **Lightweight temporal/motion cue** → a scoped-down stand-in for the parent doc's rPPG branch, since the physiological signal itself is out of scope here; motion/flicker features (LipForensics-style) are what this branch contributes instead, specifically to cover reenactment.
- **Not covered here:** rPPG/physiological liveness, real-time virtual-camera injection, and physical-only presentation attacks (print/mask/plain screen-replay of a real photo) — those remain separate workstreams in the parent project.
