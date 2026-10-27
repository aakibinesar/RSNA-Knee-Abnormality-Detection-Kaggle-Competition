"""v2 preprocessing: physical-scale crop, 336 px, 2.5D adjacent-slice triplets (shared by training and inference).

Per series: order slices by physical position (headers only), pick `n_groups` centres spread over the stack, take each
centre with its two neighbours as the three channels of one image, crop the central `fov_mm` square, resize to `size`.
Returns uint8 [n_groups, 3, size, size].
"""
import os
import numpy as np
import pydicom
import cv2

from knee_prep import choose_series, PLANES, _position_key  # noqa: F401  (same series choice + slice ordering as v1)

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
