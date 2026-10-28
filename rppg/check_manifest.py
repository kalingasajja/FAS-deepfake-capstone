import json, os, pandas as pd

REPO = r"C:\dev\FAS-deepfake-capstone"
DATA = r"C:\datasets\nfi"

m = json.load(open(os.path.join(REPO, "data", "manifests", "rppg_manifest.json")))
rows = [dict(split=s, video_id=e["video_id"], path=e["path"], label=e["label"])
        for s in ["train", "val", "test"] for e in m[s]]
df = pd.DataFrame(rows)
df["full"] = df.path.apply(lambda p: os.path.join(DATA, *p.split("/")))
df["exists"] = df.full.apply(os.path.exists)

print(df.groupby(["split", "label"]).size().unstack())
print("Missing files:", (~df.exists).sum())