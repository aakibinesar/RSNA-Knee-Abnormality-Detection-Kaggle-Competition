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
