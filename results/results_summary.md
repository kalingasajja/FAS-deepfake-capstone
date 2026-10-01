# Results summary (auto-generated)

Config: {"PAIRS": {"train": 60, "val": 15, "test": 30}, "T": 8, "SIZE": 224, "CLIPS_PER_VIDEO": 2, "SEED": 42}

## Dataset actually used (clips / videos per split)

| method | clips / test | clips / train | clips / val | videos / test | videos / train | videos / val |
|---|---|---|---|---|---|---|
| Deepfakes | 120 | 240 | 60 | 60 | 120 | 30 |
| Face2Face | 120 | 240 | 60 | 60 | 120 | 30 |
| FaceShifter | 120 | 0 | 0 | 60 | 0 | 0 |
| FaceSwap | 120 | 240 | 60 | 60 | 120 | 30 |
| NeuralTextures | 120 | 240 | 60 | 60 | 120 | 30 |
| Real | 120 | 240 | 60 | 60 | 120 | 30 |


## AUC — main models

| model | clean / ALL-seen | clean / FaceShifter(unseen) | clean / FaceSwap-group | clean / Reenact-group | rc_seen / ALL-seen | rc_seen / FaceShifter(unseen) | rc_seen / FaceSwap-group | rc_seen / Reenact-group | rc_unseen / ALL-seen | rc_unseen / FaceShifter(unseen) | rc_unseen / FaceSwap-group | rc_unseen / Reenact-group |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HGB[F+B+T] train=clean | 0.765 | 0.658 | 0.826 | 0.705 | 0.478 | 0.525 | 0.482 | 0.474 | 0.506 | 0.493 | 0.489 | 0.522 |
| HGB[F+B+T] train=aug | 0.756 | 0.682 | 0.817 | 0.694 | 0.538 | 0.580 | 0.571 | 0.505 | 0.573 | 0.546 | 0.591 | 0.554 |
| PIPELINE gate+specialists | 0.765 | 0.658 | 0.826 | 0.705 | 0.546 | 0.567 | 0.576 | 0.517 | 0.569 | 0.557 | 0.601 | 0.536 |
| CNN[EffNet-B0] train=clean | 0.842 | 0.661 | 0.818 | 0.865 | 0.496 | 0.483 | 0.485 | 0.508 | 0.524 | 0.556 | 0.522 | 0.527 |
| CNN[EffNet-B0] train=aug | 0.836 | 0.603 | 0.840 | 0.832 | 0.575 | 0.522 | 0.589 | 0.560 | 0.525 | 0.521 | 0.545 | 0.504 |


## HTER at validation-fixed threshold — main models

| model | clean / ALL-seen | clean / FaceShifter(unseen) | clean / FaceSwap-group | clean / Reenact-group | rc_seen / ALL-seen | rc_seen / FaceShifter(unseen) | rc_seen / FaceSwap-group | rc_seen / Reenact-group | rc_unseen / ALL-seen | rc_unseen / FaceShifter(unseen) | rc_unseen / FaceSwap-group | rc_unseen / Reenact-group |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HGB[F+B+T] train=clean | 0.341 | 0.442 | 0.279 | 0.402 | 0.503 | 0.504 | 0.504 | 0.502 | 0.498 | 0.496 | 0.506 | 0.490 |
| HGB[F+B+T] train=aug | 0.328 | 0.388 | 0.285 | 0.371 | 0.480 | 0.442 | 0.456 | 0.504 | 0.446 | 0.467 | 0.423 | 0.469 |
| PIPELINE gate+specialists | 0.341 | 0.446 | 0.279 | 0.402 | 0.481 | 0.475 | 0.448 | 0.515 | 0.472 | 0.475 | 0.454 | 0.490 |
| CNN[EffNet-B0] train=clean | 0.236 | 0.379 | 0.250 | 0.223 | 0.492 | 0.512 | 0.502 | 0.481 | 0.489 | 0.438 | 0.490 | 0.488 |
| CNN[EffNet-B0] train=aug | 0.297 | 0.475 | 0.304 | 0.290 | 0.486 | 0.483 | 0.481 | 0.492 | 0.484 | 0.475 | 0.483 | 0.485 |


## EER — main models

| model | clean / ALL-seen | clean / FaceShifter(unseen) | clean / FaceSwap-group | clean / Reenact-group | rc_seen / ALL-seen | rc_seen / FaceShifter(unseen) | rc_seen / FaceSwap-group | rc_seen / Reenact-group | rc_unseen / ALL-seen | rc_unseen / FaceShifter(unseen) | rc_unseen / FaceSwap-group | rc_unseen / Reenact-group |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HGB[F+B+T] train=clean | 0.302 | 0.375 | 0.250 | 0.342 | 0.517 | 0.483 | 0.500 | 0.521 | 0.501 | 0.492 | 0.517 | 0.492 |
| HGB[F+B+T] train=aug | 0.312 | 0.375 | 0.265 | 0.367 | 0.467 | 0.471 | 0.450 | 0.508 | 0.463 | 0.471 | 0.419 | 0.463 |
| PIPELINE gate+specialists | 0.302 | 0.375 | 0.250 | 0.342 | 0.468 | 0.450 | 0.463 | 0.483 | 0.426 | 0.442 | 0.417 | 0.460 |
| CNN[EffNet-B0] train=clean | 0.217 | 0.362 | 0.248 | 0.206 | 0.502 | 0.508 | 0.508 | 0.500 | 0.483 | 0.479 | 0.506 | 0.473 |
| CNN[EffNet-B0] train=aug | 0.240 | 0.425 | 0.235 | 0.240 | 0.435 | 0.458 | 0.440 | 0.450 | 0.499 | 0.483 | 0.469 | 0.517 |


## Video-level AUC (mean over clips) — main models

| model | clean / ALL-seen | clean / FaceShifter(unseen) | clean / FaceSwap-group | clean / Reenact-group | rc_seen / ALL-seen | rc_seen / FaceShifter(unseen) | rc_seen / FaceSwap-group | rc_seen / Reenact-group | rc_unseen / ALL-seen | rc_unseen / FaceShifter(unseen) | rc_unseen / FaceSwap-group | rc_unseen / Reenact-group |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HGB[F+B+T] train=clean | 0.813 | 0.703 | 0.874 | 0.752 | 0.467 | 0.527 | 0.475 | 0.458 | 0.511 | 0.503 | 0.490 | 0.533 |
| HGB[F+B+T] train=aug | 0.792 | 0.720 | 0.855 | 0.729 | 0.546 | 0.604 | 0.585 | 0.507 | 0.582 | 0.547 | 0.611 | 0.553 |
| PIPELINE gate+specialists | 0.813 | 0.703 | 0.874 | 0.752 | 0.562 | 0.587 | 0.597 | 0.528 | 0.586 | 0.571 | 0.628 | 0.545 |
| CNN[EffNet-B0] train=clean | 0.860 | 0.674 | 0.838 | 0.883 | 0.483 | 0.428 | 0.453 | 0.513 | 0.533 | 0.574 | 0.532 | 0.533 |
| CNN[EffNet-B0] train=aug | 0.852 | 0.612 | 0.857 | 0.848 | 0.587 | 0.503 | 0.603 | 0.571 | 0.541 | 0.517 | 0.563 | 0.519 |


## Feature-group ablation (AUC, ALL-seen methods)

| model | clean | rc_seen | rc_unseen |
|---|---|---|---|
| HGB[B] train=aug | 0.653 | 0.540 | 0.516 |
| HGB[B] train=clean | 0.653 | 0.513 | 0.521 |
| HGB[F+B+T] train=aug | 0.756 | 0.538 | 0.573 |
| HGB[F+B+T] train=clean | 0.765 | 0.478 | 0.506 |
| HGB[F+B] train=aug | 0.726 | 0.546 | 0.578 |
| HGB[F+B] train=clean | 0.751 | 0.506 | 0.470 |
| HGB[F] train=aug | 0.689 | 0.538 | 0.588 |
| HGB[F] train=clean | 0.716 | 0.561 | 0.508 |
| HGB[T] train=aug | 0.659 | 0.533 | 0.490 |
| HGB[T] train=clean | 0.670 | 0.498 | 0.501 |


## Recapture gate (E4)

| test_set | rate_flagged_as_screen | AUC_clean_vs_screen |
|---|---|---|
| clean (should be 'no screen') | 0.000 | nan |
| rc_seen | 1.000 | 1.000 |
| rc_unseen | 1.000 | 1.000 |


## Top features

|  | perm_importance |
|---|---|
| t_acc_ratio | 0.039 |
| b_lp05 | 0.020 |
| f_rad05 | 0.015 |
| t_acc_c | 0.014 |
| t_diff_r_sd | 0.010 |
| t_mag_c_sd | 0.006 |
| f_rad12 | 0.006 |
| f_pk_p999 | 0.006 |
| f_blk_v | 0.005 |
| t_rel | 0.005 |
| f_dct_zero | 0.005 |
| f_cr_rad1 | 0.004 |
| f_slope | 0.004 |
| f_cr_rad2 | 0.004 |
| f_cb_rad2 | 0.003 |
| f_rad09 | 0.003 |
| t_mag_ratio | 0.003 |
| b_lp00 | 0.003 |
| t_mag_r_sd | 0.003 |
| f_rad02 | 0.003 |

