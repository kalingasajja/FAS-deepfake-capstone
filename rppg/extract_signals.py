import argparse, json, os, time, urllib.request
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

REPO = r"C:\dev\FAS-deepfake-capstone"
DATA = r"C:\datasets\nfi"
OUT = r"C:\datasets\nfi_signals"
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "face_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/latest/face_landmarker.task")
SECONDS = 8.0
WIDTH = 480

ROIS = {
    "forehead": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323],
    "left_cheek": [205, 206, 207, 213, 192, 147, 187],
    "right_cheek": [425, 426, 427, 436, 416, 376, 411],
}


def ensure_model():
    if not os.path.exists(MODEL):
        print("Downloading face model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL)


def manifest_rows():
    m = json.load(open(os.path.join(REPO, "data", "manifests", "rppg_manifest.json")))
    rows = []
    for split in ["train", "val", "test"]:
        for e in m[split]:
            rows.append(dict(split=split, video_id=e["video_id"], path=e["path"],
                             label=e["label"], full=os.path.join(DATA, *e["path"].split("/"))))
    return rows


def extract(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0:
        raise RuntimeError(f"bad fps: {fps}")
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL),
        running_mode=vision.RunningMode.VIDEO, num_faces=1)
    landmarker = vision.FaceLandmarker.create_from_options(options)
    t_list, rgb_list, det_list = [], [], []
    i = 0
    try:
        while True:
            t = i / fps
            if t >= SECONDS:
                break
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if w != WIDTH:
                frame = cv2.resize(frame, (WIDTH, int(round(h * WIDTH / w))))
                h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect_for_video(img, int(round(t * 1000)))
            vals = np.full((len(ROIS), 3), np.nan, dtype=np.float32)
            found = bool(res.face_landmarks)
            if found:
                lms = res.face_landmarks[0]
                for r, idxs in enumerate(ROIS.values()):
                    pts = np.array([[lms[k].x * w, lms[k].y * h] for k in idxs], dtype=np.float32)
                    hull = cv2.convexHull(pts).astype(np.int32)
                    mask = np.zeros((h, w), np.uint8)
                    cv2.fillConvexPoly(mask, hull, 255)
                    vals[r] = cv2.mean(rgb, mask=mask)[:3]
            t_list.append(t)
            rgb_list.append(vals)
            det_list.append(found)
            i += 1
    finally:
        landmarker.close()
        cap.release()
    return fps, np.array(t_list, np.float32), np.stack(rgb_list), np.array(det_list, bool)


def kind_of(row):
    if row["label"] == 0:
        return "real"
    return "SYN" if "_SYN" in row["path"] else "FS"


def write_summary(rows):
    recs = []
    for r in rows:
        f = os.path.join(OUT, r["video_id"] + ".npz")
        if not os.path.exists(f):
            continue
        d = np.load(f)
        recs.append(dict(video_id=r["video_id"], split=r["split"], label=r["label"],
                         kind=kind_of(r), fps=float(d["fps"]), frames=len(d["t"]),
                         face_rate=float(d["det"].mean())))
    s = pd.DataFrame(recs)
    s.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    if len(s):
        print("\nSummary (done so far):")
        print(s.groupby("kind").agg(n=("video_id", "count"), fps=("fps", "median"),
                                    face_rate=("face_rate", "mean")).round(3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="process only 3 videos")
    args = ap.parse_args()
    ensure_model()
    os.makedirs(OUT, exist_ok=True)
    all_rows = manifest_rows()
    rows = all_rows
    if args.test:
        rows = [next(r for r in all_rows if r["label"] == 0),
                next(r for r in all_rows if r["label"] == 1 and "_FS" in r["path"]),
                next(r for r in all_rows if r["label"] == 1 and "_SYN" in r["path"])]
    total = len(rows)
    for k, r in enumerate(rows, 1):
        out = os.path.join(OUT, r["video_id"] + ".npz")
        if os.path.exists(out):
            continue
        t0 = time.time()
        try:
            fps, t, rgb, det = extract(r["full"])
            np.savez_compressed(out, t=t, rgb=rgb, det=det, fps=fps,
                                label=r["label"], split=r["split"])
            print(f"[{k}/{total}] {r['video_id']}: fps={fps:.1f} frames={len(t)} "
                  f"face={det.mean():.2f} {time.time() - t0:.1f}s", flush=True)
        except Exception as e:
            print(f"[{k}/{total}] FAILED {r['video_id']}: {e}", flush=True)
    write_summary(all_rows)


if __name__ == "__main__":
    main()