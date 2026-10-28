# rPPG branch

Scores are P(fake) in [0,1] per video, from pulse-signal features:
MediaPipe face patches -> POS pulse -> spectral features -> random forest.

Pipeline: extract_signals.py -> features.py -> baseline.py (train->val) -> predict_test.py -> rppg_report.py

- Model fit on train only. Threshold tuned on val. Test is not used for fitting.
- Val AUC 0.755 (95% CI 0.65 to 0.85), so this is a moderate branch.
- 2 test videos had too few face frames and were median-imputed (low confidence).
- All real videos are Celeb-DF, so compression differences may leak into the score.
- Paths: edit REPO, DATA and SIG_DIR at the top of each script.