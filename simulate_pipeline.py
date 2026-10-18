"""
simulate_pipeline.py
Runs the full pipeline in-process to produce all output files:
  1. data/master_manifest.json
  2. data/splits.json
  3. data/locked_splits.json
  4. data/splits_logo.json
  5. data/manifests/spatial_manifest.json
  6. data/manifests/frequency_manifest.json
  7. data/manifests/rppg_manifest.json
  8. data/predictions/spatial_test_preds.json
  9. data/predictions/frequency_test_preds.json
  10. data/predictions/rppg_test_preds.json
"""

import os, sys, json, re, random, urllib.request
import numpy as np

# ── helpers ──────────────────────────────────────────────────────────────────

HF_DATASET_NAME = "NetherlandsForensicInstitute/rppg-deepfake-detection"
HF_API_URL      = f"https://huggingface.co/api/datasets/{HF_DATASET_NAME}"
HF_ANN_URL      = f"https://huggingface.co/datasets/{HF_DATASET_NAME}/raw/main/annotations.json"


def _fetch(url, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": "Deepfake-Fusion-Capstone/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def fetch_file_list():
    data = _fetch(HF_API_URL)
    return [s["rfilename"] for s in data.get("siblings", [])
            if s["rfilename"].endswith((".mp4", ".avi"))]


def fetch_annotations():
    try:
        req = urllib.request.Request(HF_ANN_URL, headers={"User-Agent": "Deepfake-Fusion-Capstone/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        print(f"[WARN] annotations.json fetch failed: {e}")
        return {}


def extract_entry(rel_path, annotations):
    norm = rel_path.replace("\\", "/")
    parts = norm.split("/")
    fname = parts[-1]

    if norm.startswith("real/"):
        cat, gen, label = "real", parts[1] if len(parts) > 2 else "CelebDF", 0
        m = re.match(r"^(id\d+)_\d+", fname)
        grp = f"real_subject_{m.group(1)}" if m else f"real_{fname}"
        meta = {}
    elif norm.startswith("fake/"):
        sub = parts[1] if len(parts) > 2 else "unknown"
        label = 1
        if "_FS" in sub:
            cat, gen = "face_swap", sub.replace("_FS", "")
            m = re.match(r"^(\d+)_", fname)
            grp = f"target_{m.group(1)}" if m else fname
            meta = {}
        elif "_SYN" in sub:
            cat, gen = "synthetic", sub.replace("_SYN", "")
            m = re.search(r"(\d+)", fname)
            pidx = m.group(1) if m else fname
            grp = f"prompt_{pidx}"
            ann_key = f"{gen},{fname}"
            raw = annotations.get(ann_key, {})
            meta = {k: raw.get(k) for k in ("gender","skin_color","head_movement","lighting")}
        else:
            cat, gen, grp, meta = "unknown_fake", sub, fname, {}
    else:
        cat, gen, label, grp, meta = "unknown","unknown",0,fname,{}

    stem = os.path.splitext(fname)[0]
    vid  = f"{cat}_{gen}_{stem}"
    return dict(video_id=vid, relative_path=norm, filename=fname,
                label=label, label_name="real" if label==0 else "fake",
                category=cat, generator=gen, split_group=grp, metadata=meta)


def partition(groups, t_r=0.70, v_r=0.15, seed=42):
    rng = random.Random(seed)
    sh  = sorted(list(set(groups))); rng.shuffle(sh)
    n   = len(sh)
    nt  = max(1, int(round(n * t_r)))
    nv  = max(1, int(round(n * v_r)))
    return {"train": sh[:nt], "val": sh[nt:nt+nv], "test": sh[nt+nv:]}


def create_splits(manifest, seed=42):
    by_cat = {}
    for item in manifest:
        c, g = item["category"], item["split_group"]
        by_cat.setdefault(c, {}).setdefault(g, []).append(item)

    assigned = {"train": [], "val": [], "test": []}
    for cat, gd in by_cat.items():
        parts = partition(list(gd.keys()), seed=seed + hash(cat) % 10000)
        for sp in ("train","val","test"):
            for g in parts[sp]:
                assigned[sp].extend(gd[g])

    def stats(lst):
        r = sum(1 for x in lst if x["label"]==0)
        f = len(lst)-r
        gc = {}
        for x in lst: gc[x["generator"]] = gc.get(x["generator"],0)+1
        return {"total":len(lst),"real":r,"fake":f,"generators":gc}

    return {
        "metadata": {
            "description": "Deterministic leakage-safe group split for NFI dataset",
            "seed": seed,
            "ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
            "stats": {s: stats(assigned[s]) for s in ("train","val","test")}
        },
        **{f"{s}_video_ids": [x["video_id"] for x in assigned[s]] for s in ("train","val","test")},
        **{f"{s}_samples": assigned[s] for s in ("train","val","test")}
    }


def create_logo(manifest, seed=42):
    real_grps = {}
    fake_gens  = {}
    for item in manifest:
        if item["label"] == 0:
            real_grps.setdefault(item["split_group"], []).append(item)
        else:
            fake_gens.setdefault(item["generator"], []).append(item)

    rp = partition(list(real_grps.keys()), seed=seed)
    real_train = [x for g in rp["train"] for x in real_grps[g]]
    real_val   = [x for g in rp["val"]   for x in real_grps[g]]
    real_test  = [x for g in rp["test"]  for x in real_grps[g]]

    folds = {}
    for ho_gen, ho_fakes in fake_gens.items():
        tr_fakes, v_fakes = [], []
        for g, fl in fake_gens.items():
            if g == ho_gen: continue
            rng = random.Random(seed + hash(g) % 1000)
            sh  = list(fl); rng.shuffle(sh)
            nv  = max(1, int(len(sh)*0.15))
            v_fakes.extend(sh[:nv]); tr_fakes.extend(sh[nv:])
        folds[ho_gen] = {
            "unseen_generator": ho_gen,
            "train_video_ids": [x["video_id"] for x in real_train + tr_fakes],
            "val_video_ids":   [x["video_id"] for x in real_val   + v_fakes],
            "test_video_ids":  [x["video_id"] for x in real_test  + ho_fakes],
            "test_summary": {"total": len(real_test)+len(ho_fakes),
                             "unseen_fakes": len(ho_fakes), "real": len(real_test)}
        }
    return folds


def make_branch_manifests(splits, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for branch in ("spatial","frequency","rppg"):
        obj = {
            "branch": branch,
            "description": f"Manifest for the {branch} branch",
            **{sp: [{"video_id": x["video_id"], "path": x["relative_path"], "label": x["label"]}
                    for x in splits[f"{sp}_samples"]]
               for sp in ("train","val","test")}
        }
        with open(os.path.join(out_dir, f"{branch}_manifest.json"), "w") as f:
            json.dump(obj, f, indent=2)
    print(f"[OK] Branch manifests → {out_dir}/")


def generate_mock(splits, out_dir, seed=42):
    os.makedirs(out_dir, exist_ok=True)
    nprng = np.random.RandomState(seed)
    samples = splits["test_samples"]
    sp, fq, rp = [], [], []
    for s in samples:
        vid   = s["video_id"]
        label = s["label"]
        meta  = s.get("metadata", {})
        bad   = meta.get("lighting") == "bad" or meta.get("head_movement") == "yes"

        mu_s = 0.82 if label == 1 else 0.18
        mu_f = 0.80 if label == 1 else 0.20
        if bad:
            mu_r = 0.50
            sig_r = 0.25
        else:
            mu_r = 0.85 if label == 1 else 0.15
            sig_r = 0.10

        sp.append({"video_id": vid, "score": round(float(np.clip(nprng.normal(mu_s, 0.12), 0.01, 0.99)), 4)})
        fq.append({"video_id": vid, "score": round(float(np.clip(nprng.normal(mu_f, 0.14), 0.01, 0.99)), 4)})
        rp.append({"video_id": vid, "score": round(float(np.clip(nprng.normal(mu_r, sig_r), 0.01, 0.99)), 4)})

    for name, preds in (("spatial", sp), ("frequency", fq), ("rppg", rp)):
        path = os.path.join(out_dir, f"{name}_test_preds.json")
        with open(path, "w") as f:
            json.dump({"branch": name, "predictions": preds}, f, indent=2)
    print(f"[OK] Mock predictions ({len(samples)} test samples) → {out_dir}/")


# ── main ─────────────────────────────────────────────────────────────────────
os.makedirs("data/manifests", exist_ok=True)
os.makedirs("data/predictions", exist_ok=True)

# Step 1 – manifest
print("=" * 55)
print("STEP 1 — Fetching file list from Hugging Face...")
print("=" * 55)
file_list   = fetch_file_list()
annotations = fetch_annotations()
print(f"  → Found {len(file_list)} video files, {len(annotations)} annotation records")

manifest = [extract_entry(f, annotations) for f in sorted(file_list)]
with open("data/master_manifest.json", "w") as fout:
    json.dump(manifest, fout, indent=2)

cats = {}
for e in manifest:
    cats[e["category"]] = cats.get(e["category"], 0) + 1
print(f"  → master_manifest.json  ({len(manifest)} videos)")
for c, n in sorted(cats.items()):
    print(f"      {c:15s}: {n}")

# Step 2 – splits
print("\n" + "=" * 55)
print("STEP 2 — Creating leakage-safe splits (seed=42)...")
print("=" * 55)
splits = create_splits(manifest, seed=42)
for path in ("data/splits.json", "data/locked_splits.json"):
    with open(path, "w") as fout:
        json.dump(splits, fout, indent=2)
st = splits["metadata"]["stats"]
for sp in ("train", "val", "test"):
    s = st[sp]
    print(f"  [{sp.upper():5}]  total={s['total']:3d}  real={s['real']:3d}  fake={s['fake']:3d}")

# Step 3 – LOGO
print("\n" + "=" * 55)
print("STEP 3 — Creating Leave-One-Generator-Out (LOGO) splits...")
print("=" * 55)
logo = create_logo(manifest, seed=42)
with open("data/splits_logo.json", "w") as fout:
    json.dump(logo, fout, indent=2)
print(f"  → {len(logo)} LOGO folds")
for gen, fold in logo.items():
    ts = fold["test_summary"]
    print(f"      leave-out {gen:15s}: test={ts['total']} (fakes={ts['unseen_fakes']} real={ts['real']})")

# Step 4 – branch manifests
print("\n" + "=" * 55)
print("STEP 4 — Exporting per-branch manifests...")
print("=" * 55)
make_branch_manifests(splits, "data/manifests")
for branch in ("spatial", "frequency", "rppg"):
    p = f"data/manifests/{branch}_manifest.json"
    with open(p) as f:
        bm = json.load(f)
    print(f"  {branch}_manifest.json  train={len(bm['train'])}  val={len(bm['val'])}  test={len(bm['test'])}")

# Step 5 – mock predictions
print("\n" + "=" * 55)
print("STEP 5 — Generating mock branch predictions...")
print("=" * 55)
generate_mock(splits, "data/predictions", seed=42)

print("\n" + "=" * 55)
print(" DONE — all output files ready.")
print("=" * 55)
print("  data/master_manifest.json")
print("  data/splits.json")
print("  data/locked_splits.json")
print("  data/splits_logo.json")
print("  data/manifests/spatial_manifest.json")
print("  data/manifests/frequency_manifest.json")
print("  data/manifests/rppg_manifest.json")
print("  data/predictions/spatial_test_preds.json")
print("  data/predictions/frequency_test_preds.json")
print("  data/predictions/rppg_test_preds.json")
