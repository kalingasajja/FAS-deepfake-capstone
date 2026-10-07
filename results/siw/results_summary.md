# SiW Live vs. Spoof — Results summary (auto-generated)

Config: {"T": 8, "SIZE": 224, "CLIPS_PER_VIDEO": 2, "SEED": 42, "TRAIN_FRAC": 0.6, "VAL_FRAC": 0.15}

## Dataset actually used (clips / videos per split, Live vs Spoof)

| method | split | clips | videos |
|---|---|---|---|
| Live | test | 1615 | 1615 |
| Live | train | 3621 | 3621 |
| Live | val | 829 | 829 |
| Spoof | test | 625 | 625 |
| Spoof | train | 749 | 749 |
| Spoof | val | 128 | 128 |


## Test-set results — all models (AUC, EER, HTER, accuracy)

| model | AUC | EER | HTER | ACC | AUC_video |
|---|---|---|---|---|---|
| HGB[F] | 0.914 | 0.163 | 0.182 | 0.861 | 0.914 |
| HGB[B] | 0.849 | 0.240 | 0.247 | 0.704 | 0.849 |
| HGB[T] | 0.566 | 0.451 | 0.457 | 0.602 | 0.566 |
| HGB[F+B] | 0.936 | 0.140 | 0.138 | 0.879 | 0.936 |
| HGB[F+B+T] | 0.937 | 0.147 | 0.145 | 0.858 | 0.937 |
| CNN[EffNet-B0] | 0.997 | 0.033 | 0.033 | 0.973 | 0.997 |


## Feature-group ablation (classical model only)

| model | AUC | HTER | ACC |
|---|---|---|---|
| HGB[F] | 0.914 | 0.182 | 0.861 |
| HGB[B] | 0.849 | 0.247 | 0.704 |
| HGB[T] | 0.566 | 0.457 | 0.602 |
| HGB[F+B] | 0.936 | 0.138 | 0.879 |
| HGB[F+B+T] | 0.937 | 0.145 | 0.858 |


## Top 15 features by permutation importance (full-feature classical model)

| feature | perm_importance |
|---|---|
| f_cb_rad0 | 0.046 |
| f_cr_rad0 | 0.042 |
| f_hf | 0.028 |
| f_cb_rad3 | 0.019 |
| f_rad01 | 0.014 |
| f_band_col | 0.011 |
| f_pk_max | 0.010 |
| f_pk_p999 | 0.010 |
| b_dY | 0.007 |
| f_cr_rad3 | 0.007 |
| b_gm00 | 0.007 |
| f_slope | 0.006 |
| f_dct01 | 0.006 |
| b_gm01 | 0.005 |
| b_gm05 | 0.004 |

