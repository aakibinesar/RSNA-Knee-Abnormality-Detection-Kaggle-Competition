SHARD, NSHARDS, LIMIT, CACHE_PX = 5, 6, 0, 384
"""Series selection and DICOM -> small uint8 volume conversion for the RSNA knee competition.

Used by both the preprocessing kernel and the inference kernel so train/test pipelines stay identical.
"""
import os
import numpy as np
import pandas as pd
import pydicom
import cv2

PLANES = ("Sagittal", "Coronal", "Axial")
SIZE = 224   # output height/width
DEPTH = 32   # output slices per series


def choose_series(series_df: pd.DataFrame) -> dict:
    """Pick one series per plane for one study. Prefer fluid-sensitive + fat-suppressed, then fluid-sensitive.

    series_df: rows of train_series.csv / test_series.csv for a single study.
    Returns {plane: SeriesInstanceUID}; planes without any series are omitted.
    """
    out = {}
    for plane in PLANES:
        g = series_df[series_df.Anatomical_Plane == plane]
        if g.empty:
            continue
        g = g.assign(_k=-(g.Fluid_Sensitive * 2 + g.Fat_Suppression)).sort_values(["_k", "SeriesInstanceUID"])
        out[plane] = g.SeriesInstanceUID.iloc[0]
    return out


def _position_key(ds):
    """Position of the slice along its own normal; falls back to InstanceNumber."""
    try:
        iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
        ipp = np.asarray(ds.ImagePositionPatient, dtype=float)
        normal = np.cross(iop[:3], iop[3:])
        return float(np.dot(ipp, normal))
    except Exception:
        return float(getattr(ds, "InstanceNumber", 0))


def load_series(series_dir: str, size: int = SIZE, depth: int = DEPTH) -> np.ndarray:
    """Read one series, order slices spatially, normalise, centre-crop square, resize, resample to `depth`.

    Returns uint8 array [depth, size, size].
    """
    items = []
    for f in os.listdir(series_dir):
        ds = pydicom.dcmread(os.path.join(series_dir, f))
        items.append((_position_key(ds), int(getattr(ds, "InstanceNumber", 0)), ds.pixel_array.astype(np.float32)))
    items.sort(key=lambda t: (t[0], t[1]))
    vol = np.stack([it[2] for it in items])  # [N, H, W]

    lo, hi = np.percentile(vol, [0.5, 99.5])
    vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)

    n, h, w = vol.shape
    s = min(h, w)
    t, l = (h - s) // 2, (w - s) // 2
    vol = vol[:, t:t + s, l:l + s]
    vol = np.stack([cv2.resize(sl, (size, size), interpolation=cv2.INTER_AREA) for sl in vol])

    idx = np.linspace(0, n - 1, depth).round().astype(int)
    return (vol[idx] * 255).astype(np.uint8)


def load_study(root: str, split: str, study_uid: str, series_df: pd.DataFrame,
               size: int = SIZE, depth: int = DEPTH):
    """Returns (volumes [3, depth, size, size] uint8 in PLANES order, present mask [3] bool)."""
    chosen = choose_series(series_df)
    vols = np.zeros((len(PLANES), depth, size, size), np.uint8)
    present = np.zeros(len(PLANES), bool)
    for i, plane in enumerate(PLANES):
        if plane in chosen:
            try:
                vols[i] = load_series(f"{root}/{split}_series/{study_uid}/{chosen[plane]}", size, depth)
                present[i] = True
            except Exception as e:  # a corrupt/unreadable series must not crash the whole run
                print(f"WARN {study_uid[-8:]} {plane}: {type(e).__name__}: {str(e)[:80]}", flush=True)
    return vols, present

"""v2 preprocessing: physical-scale crop, 336 px, 2.5D adjacent-slice triplets (shared by training and inference).

Per series: order slices by physical position (headers only), pick `n_groups` centres spread over the stack, take each
centre with its two neighbours as the three channels of one image, crop the central `fov_mm` square, resize to `size`.
Returns uint8 [n_groups, 3, size, size].
"""
import os
import numpy as np
import pydicom
import cv2


SIZE2 = 336
FOV_MM = 150.0
N_GROUPS = 11


def _header(path):
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    try:
        sp = float(ds.PixelSpacing[0])
    except Exception:
        sp = 0.3125
    return _position_key(ds), int(getattr(ds, "InstanceNumber", 0)), sp


def _crop_resize(px, spacing, size, fov_mm):
    h, w = px.shape
    side = int(min(h, w, max(round(fov_mm / spacing), 8)))
    t, l = (h - side) // 2, (w - side) // 2
    c = px[t:t + side, l:l + side].astype(np.float32)
    return cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA if side > size else cv2.INTER_LINEAR)


def load_series_v2(series_dir, size=SIZE2, fov_mm=FOV_MM, n_groups=N_GROUPS):
    files = sorted(os.listdir(series_dir))
    hdr = [(*_header(os.path.join(series_dir, f)), f) for f in files]
    hdr.sort(key=lambda t: (t[0], t[1]))
    n = len(hdr)
    centres = np.linspace(1, n - 2, n_groups).round().astype(int) if n >= 3 else np.zeros(n_groups, int)
    idx = np.clip(centres[:, None] + np.array([-1, 0, 1])[None, :], 0, n - 1)   # [n_groups, 3]
    cache = {}
    for i in np.unique(idx):
        _, _, sp, f = hdr[i]
        cache[i] = _crop_resize(pydicom.dcmread(os.path.join(series_dir, f)).pixel_array, sp, size, fov_mm)
    stack = np.stack([cache[i] for i in sorted(cache)])
    lo, hi = np.percentile(stack, [0.5, 99.5])
    scale = lambda a: (np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    return np.stack([[scale(cache[i]) for i in row] for row in idx])


def load_study_v2(root, split, study_uid, series_df, size=SIZE2, fov_mm=FOV_MM, n_groups=N_GROUPS):
    """Returns (vols uint8 [3 planes, n_groups, 3, size, size], present bool [3])."""
    chosen = choose_series(series_df)
    vols = np.zeros((len(PLANES), n_groups, 3, size, size), np.uint8)
    present = np.zeros(len(PLANES), bool)
    for i, plane in enumerate(PLANES):
        if plane in chosen:
            try:
                vols[i] = load_series_v2(f"{root}/{split}_series/{study_uid}/{chosen[plane]}", size, fov_mm, n_groups)
                present[i] = True
            except Exception as e:
                print(f"WARN {study_uid[-8:]} {plane}: {type(e).__name__}: {str(e)[:80]}", flush=True)
    return vols, present

# ---------------- v2 image cache: 2.5D triplets, compressed, sharded ----------------
import time, json
from concurrent.futures import ProcessPoolExecutor

ROOT = "/kaggle/input/competitions/rsna-knee-abnormality-detection"
if not os.path.exists(f"{ROOT}/train_series.csv"):            # diagnostic + fallback: the competition mount was missing on some sessions
    print("ROOT missing; /kaggle/input =", os.listdir("/kaggle/input") if os.path.exists("/kaggle/input") else "absent", flush=True)
    for r, ds, fs in os.walk("/kaggle/input"):
        if r.count("/") > 4: ds[:] = []; continue
        print("  ", r, ds[:6], fs[:6], flush=True)
        if "train_series.csv" in fs: ROOT = r; print("found train_series.csv in", r, flush=True); break
OUT = f"/kaggle/working/shard{SHARD}"
os.makedirs(OUT, exist_ok=True)
SIZE_C = CACHE_PX                                   # pixels per side of every cached slice (set by build.sh; 280 for the existing caches)


def work(uid):
    try:
        vols, present = load_study_v2(ROOT, "train", uid, BY_STUDY[uid], size=SIZE_C)
        np.savez_compressed(f"{OUT}/{uid}.npz", vols=vols, present=present)
        return uid, present.tolist(), None
    except Exception as e:
        return uid, None, f"{type(e).__name__}: {str(e)[:100]}"


SERIES = pd.read_csv(f"{ROOT}/train_series.csv")
BY_STUDY = {k: g for k, g in SERIES.groupby("StudyInstanceUID")}

if __name__ == "__main__":
    studies = sorted(BY_STUDY)[SHARD::NSHARDS]
    if LIMIT: studies = studies[:LIMIT]
    print(f"shard {SHARD}/{NSHARDS}: {len(studies)} studies @ {SIZE_C}px", flush=True)
    t0, done, errs, manifest = time.time(), 0, [], {}
    with ProcessPoolExecutor(4) as ex:
        for uid, present, err in ex.map(work, studies, chunksize=4):
            done += 1
            if err: errs.append((uid, err))
            else: manifest[uid] = present
            if done % 100 == 0 or done == len(studies):
                gb = sum(os.path.getsize(f"{OUT}/{f}") for f in os.listdir(OUT)) / 1e9
                print(f"{done}/{len(studies)} | {time.time()-t0:.0f}s | out {gb:.2f} GB | errors {len(errs)}", flush=True)
                if gb > 17.5:
                    print("STOP: output near the 20 GB limit"); break
    json.dump(manifest, open(f"{OUT}/manifest.json", "w"))
    print("errors:", errs[:5], "| planes present:", np.array(list(manifest.values())).sum(0).tolist(), "of", len(manifest))
