#!/usr/bin/env python
"""
run_all.py  ─  Single entry-point that does everything:

    python run_all.py                            # remote HF fetch (default)
    python run_all.py --dataset-dir data/nfi_dataset  # local directory

Produces:
    data/master_manifest.json
    data/splits.json  +  data/locked_splits.json
    data/splits_logo.json
    data/manifests/{spatial,frequency,rppg}_manifest.json
    data/predictions/{spatial,frequency,rppg}_test_preds.json  (mock – until real preds arrive)
    results/  (if --run-fusion flag is passed)
"""

import os, re, sys, json, random, argparse, urllib.request
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

HF_ID      = "NetherlandsForensicInstitute/rppg-deepfake-detection"
HF_API_URL = f"https://huggingface.co/api/datasets/{HF_ID}"
HF_ANN_URL = f"https://huggingface.co/datasets/{HF_ID}/raw/main/annotations.json"
SEED       = 42


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 – Fetch / scan
# ──────────────────────────────────────────────────────────────────────────────

def _hf_get(url: str) -> dict | list:
    req = urllib.request.Request(url, headers={"User-Agent": "Deepfake-Capstone/1.0"})
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.loads(r.read().decode())


def fetch_file_list_remote() -> list[str]:
    data = _hf_get(HF_API_URL)
    return [s["rfilename"] for s in data.get("siblings", [])
            if s["rfilename"].lower().endswith((".mp4", ".avi"))]


def fetch_file_list_local(root: str) -> list[str]:
    out = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".mp4", ".avi")):
                rel = os.path.relpath(os.path.join(dirpath, f), root).replace("\\", "/")
                out.append(rel)
    return sorted(out)


def fetch_annotations_remote() -> dict:
    try:
        req = urllib.request.Request(HF_ANN_URL, headers={"User-Agent": "Deepfake-Capstone/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        print(f"  [WARN] Could not fetch annotations.json: {e}")
        return {}


def fetch_annotations_local(root: str) -> dict:
    p = os.path.join(root, "annotations.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 – Parse each video path into a manifest entry
# ──────────────────────────────────────────────────────────────────────────────

def _entry(rel: str, anns: dict) -> dict:
    norm  = rel.replace("\\", "/")
    parts = norm.split("/")
    fname = parts[-1]
    stem  = os.path.splitext(fname)[0]

    if norm.startswith("real/"):
        cat, gen, label = "real", parts[1] if len(parts) > 2 else "CelebDF", 0
        m   = re.match(r"^(id\d+)_\d+", fname)
        grp = f"real_subject_{m.group(1)}" if m else f"real_{stem}"
        meta = {}

    elif norm.startswith("fake/"):
        label = 1
        sub   = parts[1] if len(parts) > 2 else "unknown"

        if "_FS" in sub:                          # face-swap
            cat, gen = "face_swap", sub.replace("_FS", "")
            m   = re.match(r"^(\d+)_", fname)
            grp = f"target_{m.group(1)}" if m else stem
            meta = {}

        elif "_SYN" in sub:                       # fully synthetic
            cat, gen = "synthetic", sub.replace("_SYN", "")
            m   = re.search(r"(\d+)", fname)
            grp = f"prompt_{m.group(1)}" if m else stem
            raw = anns.get(f"{gen},{fname}", {})
            meta = {k: raw.get(k) for k in ("gender", "skin_color", "head_movement", "lighting")}

        else:
            cat, gen, grp, meta = "unknown_fake", sub, stem, {}

    else:
        cat, gen, label, grp, meta = "unknown", "unknown", 0, stem, {}

    return dict(
        video_id      = f"{cat}_{gen}_{stem}",
        relative_path = norm,
        filename      = fname,
        label         = label,
        label_name    = "real" if label == 0 else "fake",
        category      = cat,
        generator     = gen,
        split_group   = grp,
        metadata      = meta,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 – Leakage-safe splits
# ──────────────────────────────────────────────────────────────────────────────

def _partition(groups: list[str], t=0.70, v=0.15, seed=SEED) -> dict:
    rng = random.Random(seed)
    sh  = sorted(set(groups)); rng.shuffle(sh)
    n   = len(sh)
    nt  = max(1, round(n * t))
    nv  = max(1, round(n * v))
    return {"train": sh[:nt], "val": sh[nt:nt+nv], "test": sh[nt+nv:]}


def build_splits(manifest: list[dict]) -> dict:
    by_cat: dict[str, dict[str, list]] = {}
    for item in manifest:
        by_cat.setdefault(item["category"], {}).setdefault(item["split_group"], []).append(item)

    asgn: dict[str, list] = {"train": [], "val": [], "test": []}
    for cat, gd in by_cat.items():
        p = _partition(list(gd), seed=SEED + hash(cat) % 10_000)
        for sp in ("train", "val", "test"):
            for g in p[sp]:
                asgn[sp].extend(gd[g])

    def _stats(lst: list) -> dict:
        r  = sum(1 for x in lst if x["label"] == 0)
        gc = {}
        for x in lst:
            gc[x["generator"]] = gc.get(x["generator"], 0) + 1
        return {"total": len(lst), "real": r, "fake": len(lst) - r, "generators": gc}

    return {
        "metadata": {
            "description": "Deterministic leakage-safe splits — seed=42, ratios 70/15/15",
            "seed": SEED,
            "ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
            "leakage_rules": {
                "real":      "Grouped by subject_id (id0..id61). Same person never spans splits.",
                "face_swap": "Grouped by target_video_id (001,009..). Same scene never spans splits.",
                "synthetic": "Grouped by prompt_index (prompt_1..prompt_26). Same prompt never spans splits.",
            },
            "stats": {s: _stats(asgn[s]) for s in ("train", "val", "test")},
        },
        **{f"{s}_video_ids": [x["video_id"] for x in asgn[s]] for s in ("train", "val", "test")},
        **{f"{s}_samples":   asgn[s]                           for s in ("train", "val", "test")},
    }


def build_logo(manifest: list[dict]) -> dict:
    real_grps: dict[str, list] = {}
    fake_gens: dict[str, list] = {}
    for item in manifest:
        if item["label"] == 0:
            real_grps.setdefault(item["split_group"], []).append(item)
        else:
            fake_gens.setdefault(item["generator"], []).append(item)

    rp        = _partition(list(real_grps), t=0.70, v=0.15)
    real_tr   = [x for g in rp["train"] for x in real_grps[g]]
    real_v    = [x for g in rp["val"]   for x in real_grps[g]]
    real_te   = [x for g in rp["test"]  for x in real_grps[g]]

    folds = {}
    for ho, ho_fakes in fake_gens.items():
        tr_f, v_f = [], []
        for g, fl in fake_gens.items():
            if g == ho: continue
            rng = random.Random(SEED + hash(g) % 1000)
            sh  = list(fl); rng.shuffle(sh)
            nv  = max(1, int(len(sh) * 0.15))
            v_f.extend(sh[:nv]); tr_f.extend(sh[nv:])
        folds[ho] = {
            "unseen_generator": ho,
            "train_video_ids": [x["video_id"] for x in real_tr + tr_f],
            "val_video_ids":   [x["video_id"] for x in real_v  + v_f],
            "test_video_ids":  [x["video_id"] for x in real_te + ho_fakes],
            "test_summary": {
                "total":         len(real_te) + len(ho_fakes),
                "unseen_fakes":  len(ho_fakes),
                "real":          len(real_te),
            },
        }
    return folds


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 – Branch manifests
# ──────────────────────────────────────────────────────────────────────────────

def export_manifests(splits: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for branch in ("spatial", "frequency", "rppg"):
        obj = {
            "branch": branch,
            "description": (
                f"Manifest for the {branch} branch. "
                "Use 'train' to fit your model, 'val' to tune thresholds, "
                "'test' ONLY for final evaluation. "
                "Export predictions for all 'test' entries as data/predictions/"
                f"{branch}_test_preds.json."
            ),
            **{sp: [{"video_id": x["video_id"],
                     "path":     x["relative_path"],
                     "label":    x["label"]}
                    for x in splits[f"{sp}_samples"]]
               for sp in ("train", "val", "test")},
        }
        with open(os.path.join(out_dir, f"{branch}_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
# Step 5 – Mock predictions (simulates degraded rPPG under bad lighting/motion)
# ──────────────────────────────────────────────────────────────────────────────

def generate_mock(splits: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(SEED)
    sp_preds, fq_preds, rp_preds = [], [], []

    for s in splits["test_samples"]:
        vid   = s["video_id"]
        label = s["label"]
        meta  = s.get("metadata", {})
        bad   = meta.get("lighting") == "bad" or meta.get("head_movement") == "yes"

        mu_s = 0.82 if label else 0.18
        mu_f = 0.80 if label else 0.20
        mu_r, sig_r = (0.50, 0.25) if bad else ((0.85, 0.10) if label else (0.15, 0.10))

        clip = lambda x: round(float(np.clip(x, 0.01, 0.99)), 4)
        sp_preds.append({"video_id": vid, "score": clip(rng.normal(mu_s, 0.12))})
        fq_preds.append({"video_id": vid, "score": clip(rng.normal(mu_f, 0.14))})
        rp_preds.append({"video_id": vid, "score": clip(rng.normal(mu_r, sig_r))})

    for name, preds in (("spatial", sp_preds), ("frequency", fq_preds), ("rppg", rp_preds)):
        with open(os.path.join(out_dir, f"{name}_test_preds.json"), "w", encoding="utf-8") as f:
            json.dump({"branch": name, "predictions": preds}, f, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def _save(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print(f"  ✔  {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full dataset + fusion pipeline.")
    parser.add_argument("--dataset-dir", default=None,
                        help="Local NFI dataset directory (omit to fetch from HuggingFace).")
    parser.add_argument("--run-fusion",  action="store_true",
                        help="Also run baseline fusion on mock predictions.")
    args = parser.parse_args()

    # ── 1. File list + annotations ───────────────────────────────────────────
    print("\n── Step 1 ─ Loading file list & annotations ──────────────────────")
    if args.dataset_dir:
        files = fetch_file_list_local(args.dataset_dir)
        anns  = fetch_annotations_local(args.dataset_dir)
        print(f"  Local scan: {len(files)} video files, {len(anns)} annotation records")
    else:
        print("  Querying HuggingFace API …")
        files = fetch_file_list_remote()
        anns  = fetch_annotations_remote()
        print(f"  Remote: {len(files)} video files, {len(anns)} annotation records")

    # ── 2. Master manifest ───────────────────────────────────────────────────
    print("\n── Step 2 ─ Building master_manifest.json ────────────────────────")
    manifest = [_entry(f, anns) for f in sorted(files)]
    _save("data/master_manifest.json", manifest)

    cats = {}
    for e in manifest:
        cats[e["category"]] = cats.get(e["category"], 0) + 1
    for c, n in sorted(cats.items()):
        print(f"      {c:15s}: {n:3d}")

    # ── 3. Leakage-safe splits ───────────────────────────────────────────────
    print("\n── Step 3 ─ Creating splits (seed=42, 70/15/15) ──────────────────")
    splits = build_splits(manifest)
    _save("data/splits.json",        splits)
    _save("data/locked_splits.json", splits)   # immutable backup

    st = splits["metadata"]["stats"]
    for sp in ("train", "val", "test"):
        s = st[sp]
        print(f"  [{sp.upper():5}]  total={s['total']:3d}  real={s['real']:3d}  fake={s['fake']:3d}")

    # ── 4. LOGO generalization splits ────────────────────────────────────────
    print("\n── Step 4 ─ Creating LOGO generalization splits ──────────────────")
    logo = build_logo(manifest)
    _save("data/splits_logo.json", logo)
    for gen, fold in logo.items():
        ts = fold["test_summary"]
        print(f"  leave-out {gen:15s}: test={ts['total']:3d}  (fakes={ts['unseen_fakes']}  real={ts['real']})")

    # ── 5. Branch manifests ──────────────────────────────────────────────────
    print("\n── Step 5 ─ Exporting per-branch manifests ───────────────────────")
    export_manifests(splits, "data/manifests")
    for branch in ("spatial", "frequency", "rppg"):
        p = f"data/manifests/{branch}_manifest.json"
        with open(p, encoding="utf-8") as f:
            bm = json.load(f)
        print(f"  {branch}_manifest.json  "
              f"train={len(bm['train'])}  val={len(bm['val'])}  test={len(bm['test'])}")

    # ── 6. Mock predictions ──────────────────────────────────────────────────
    print("\n── Step 6 ─ Generating mock branch predictions ───────────────────")
    generate_mock(splits, "data/predictions")
    for br in ("spatial", "frequency", "rppg"):
        p = f"data/predictions/{br}_test_preds.json"
        with open(p, encoding="utf-8") as f:
            preds = json.load(f)
        print(f"  {br}_test_preds.json  ({len(preds['predictions'])} entries)")

    # ── 7. Optional fusion run ───────────────────────────────────────────────
    if args.run_fusion:
        print("\n── Step 7 ─ Running baseline fusion pipeline ─────────────────────")
        os.system(
            "python run_fusion_pipeline.py "
            "--manifest data/master_manifest.json "
            "--splits data/splits.json "
            "--spatial-preds  data/predictions/spatial_test_preds.json "
            "--frequency-preds data/predictions/frequency_test_preds.json "
            "--rppg-preds     data/predictions/rppg_test_preds.json "
            "--output-dir results"
        )

    print("\n" + "═" * 60)
    print(" ALL STEPS COMPLETE")
    print("═" * 60)
    print(" Files ready:")
    for fp in [
        "data/master_manifest.json",
        "data/splits.json",
        "data/locked_splits.json",
        "data/splits_logo.json",
        "data/manifests/spatial_manifest.json",
        "data/manifests/frequency_manifest.json",
        "data/manifests/rppg_manifest.json",
        "data/predictions/spatial_test_preds.json",
        "data/predictions/frequency_test_preds.json",
        "data/predictions/rppg_test_preds.json",
    ]:
        exists = "✔" if os.path.exists(fp) else "✘ MISSING"
        print(f"  {exists}  {fp}")
    print()


if __name__ == "__main__":
    main()
