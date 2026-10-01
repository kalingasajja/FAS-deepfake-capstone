# Week 2 — Implementation, Experiments & Results

## Goals
- Build the full pipeline: data → features/model → evaluation → report
- Run experiments proving (or disproving) the screen-replay hypothesis
- Produce documented, reproducible results

## What I did
- Built a Colab pipeline (`notebooks/deepfake_imageproc_colab.ipynb`):
  1. Downloaded FaceForensics++ (c23) via Kaggle, using the official FF++ train/val/test splits
  2. Face-cropped 8-frame clips from each video (2 clips/video)
  3. Built a screen-replay simulator modeling an LCD sub-pixel grid + camera recapture (rotation, blur, moiré, banding, gamma shift, noise, JPEG) — with a separate "unseen" profile to test generalization
  4. Extracted hand-crafted features: frequency (FFT/DCT), boundary/blending (gradient/noise-residual/Laplacian rings), temporal (optical flow, flicker)
  5. Trained classical models (gradient-boosted trees) and ran 5 experiments:
     - E1: trained on clean data only → tested on clean/recaptured (shows the drop)
     - E2: trained on clean + simulated recapture → same test (shows the recovery)
     - E3: feature-group ablation (frequency vs boundary vs temporal vs combined)
     - E4: two-stage pipeline — "is this a screen recapture?" gate → specialist model
     - E5: unseen-manipulation test (FaceShifter, never trained on)
  6. Trained an EfficientNet-B0 deep baseline for comparison, same clean/aug split
  7. Auto-generated a results report (tables + figures) from the run

## Output
- `results/all_results.csv` — full metrics table (AUC, EER, HTER, accuracy) across all models/conditions
- `results/results_summary.md` — auto-generated markdown report
- `results/fig1..fig8*.png` — recapture examples, spectra, AUC heatmaps, feature-group ablation, ROC curves, top-feature importance
- `results/dataset_summary.csv` — exact clip/video counts used per split

## Key finding
Training on clean deepfakes only achieves strong detection (AUC 0.75–0.86 across classical and CNN models), but performance collapses to near-chance (AUC ~0.48–0.58) when the same models are tested against simulated screen-replay recapture — confirming the central hypothesis that screen-recapture is a distinct, unaddressed failure mode.

Training with recapture-augmented data partially recovers performance: the classical feature-based model (frequency + boundary + temporal) improved from 0.506 to 0.573 AUC on unseen recapture conditions, while the CNN baseline showed negligible improvement (0.524 → 0.525), suggesting hand-crafted features generalize to novel recapture conditions better than an end-to-end CNN trained on the same limited augmented data.

The strongest result came from the two-stage "screen detector → specialist model" pipeline (E4), which reached 0.569–0.601 AUC under unseen recapture — better than any single end-to-end model — though the gate's own clean-vs-recapture detection (AUC 1.000) reflects how separable our own simulator's artifacts are, not necessarily real-world recapture, since both classes come from the same synthetic pipeline.

As predicted by the literature review, reenactment attacks (Face2Face, NeuralTextures) degraded more under recapture than face-swap attacks (Deepfakes, FaceSwap) — e.g. under the pipeline model, FaceSwap-group held 0.601 AUC vs Reenactment-group's 0.536 — consistent with reenactment's weaker native forgery signal being more fragile under further degradation.

Permutation feature importance showed the single most informative feature was a temporal one (center-vs-ring optical-flow acceleration ratio), not a frequency-domain one, suggesting motion/flicker inconsistency carries more signal than frequency artifacts in this setup.

**Caveats:** EER/HTER remained near 0.45–0.52 under recapture even where AUC showed a ranking signal, meaning a fixed-threshold classifier is still close to unreliable in practice. Results are from a Colab-scale subset (60/15/30 video pairs per split) and should be read as indicative trends, not final benchmark numbers.

## Next steps (stretch, optional)
- Film real screen recaptures to validate the simulation against real footage (Stage 9 of the notebook)
- Awaiting approval for the DeepMoiréFake dataset for real-recapture evaluation