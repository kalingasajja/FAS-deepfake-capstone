# Week 3 — Face Anti-Spoofing on SiW (Spoof in the Wild)

## What this week covers

After finishing the FaceForensics++ deepfake-detection pipeline, I extended the
image-processing work to a related but different problem: **liveness / face
anti-spoofing**. Instead of detecting GAN-generated deepfakes, this trains and
tests a model that tells a **real live camera feed** apart from a
**presentation attack** — someone holding up a printed photo or replaying a
video on a screen to fool the camera.

Dataset: [SiW (Spoof in the Wild)](https://www.kaggle.com/datasets/kokomi3012/siw-dataset)
— 165 subjects, live videos plus print and replay spoof attacks.

## What I did

- **Reused the existing feature-extraction pipeline** (frequency, boundary,
  and temporal hand-crafted features) built for the FF++ work, since these
  are generic pixel-statistics cues, not deepfake-specific — they transfer
  directly to spotting print/replay spoofing.
- **Indexed and labeled the dataset** from SiW's filename convention
  (`SubjectID_SensorID_TypeID_MediumID_SessionID`), mapping TypeID to
  live vs. spoof.
- **Built subject-disjoint train/val/test splits** — no subject's face
  appears in more than one split, so the model can't just memorize faces;
  it has to actually learn spoof cues.
- **Face-cropped video clips** (8 frames per clip, 2 clips per video) the
  same way as the FF++ pipeline.
- **Trained two models:**
  - A classical model (Histogram Gradient Boosting) on the hand-crafted
    features, including a feature-group ablation (frequency-only,
    boundary-only, temporal-only, and combined).
  - A deep learning model (EfficientNet-B0, fine-tuned) on the raw face
    crops.
- **Evaluated both** on a held-out test set using AUC, EER, HTER, and
  accuracy, plus permutation importance to see which features mattered most.

## Results

| Model | AUC | EER | HTER | Accuracy |
|---|---|---|---|---|
| HGB — frequency only | 0.914 | 0.163 | 0.182 | 0.861 |
| HGB — boundary only | 0.849 | 0.240 | 0.247 | 0.704 |
| HGB — temporal only | 0.566 | 0.451 | 0.457 | 0.602 |
| HGB — frequency + boundary | 0.936 | 0.140 | 0.138 | 0.879 |
| HGB — all features | 0.937 | 0.147 | 0.145 | 0.858 |
| **CNN (EfficientNet-B0)** | **0.997** | **0.033** | **0.033** | **0.973** |

*(AUC/accuracy: higher is better. EER/HTER: lower is better.)*

## Key takeaways

- Unlike the FF++ deepfake task — where screen-replay badly broke
  detection (AUC collapsing to ~0.50) — **live-vs-spoof detection on SiW
  is a much easier, more separable problem**. Even the classical model hits
  AUC 0.94, and the CNN nearly saturates at AUC 0.997.
- **Frequency-domain features are by far the strongest signal** (AUC 0.91
  alone), while temporal motion cues are the weakest (AUC 0.57, barely
  above chance) — makes sense, since a printed photo or a screen replay
  has obvious frequency/texture artifacts (moiré, re-compression, flat
  print texture) that a live face doesn't.
- Combining feature groups helps the classical model close most of the
  gap to the CNN, but the CNN still wins outright by directly learning
  from pixels rather than hand-crafted statistics.
- This result is a useful contrast to the FF++ experiment: it shows the
  same feature/model pipeline generalizes well to a *real* spoof-attack
  dataset (not simulated), while the *simulated* screen-replay attack in
  the FF++ work remains the harder, unsolved case.

## Files

- `notebooks/siw_liveness_colab.ipynb` — full pipeline notebook (Colab, GPU)
- `results/siw/results_summary.md` — auto-generated results summary
- `results/siw/classical_results.csv` — classical model results + ablation
- `results/siw/cnn_results.csv` — CNN results
- `results/siw/all_results.csv` — combined results table
- `results/siw/dataset_summary.csv` — dataset split/clip counts
