ARCH, RES, TTA, CACHE_PX = "resnet50", 224, 0, 280
ARCH_MAP = {"rsna-knee-18-cnn3-cnx0": "convnext_tiny","rsna-knee-18-cnn3-cnx1": "convnext_tiny","rsna-knee-18-cnn3-cnx2": "convnext_tiny","rsna-knee-18-cnn3-cnx3": "convnext_tiny","rsna-knee-18-cnn3-cnx4": "convnext_tiny","rsna-knee-18-cnn3-cxs0": "convnext_small","rsna-knee-18-cnn3-cxs1": "convnext_small","rsna-knee-18-cnn3-cxs2": "convnext_small","rsna-knee-18-cnn3-cxs3": "convnext_small","rsna-knee-18-cnn3-cxs4": "convnext_small","rsna-knee-18-cnn3-t384f0": "convnext_tiny","rsna-knee-18-cnn3-t384f1": "convnext_tiny","rsna-knee-18-cnn3-mir0": "convnext_tiny","rsna-knee-18-cnn3-mir1": "convnext_tiny"}
import os
import numpy as np
import pandas as pd
"""Series selection and DICOM -> small uint8 volume conversion for the RSNA knee competition.

Used by both the preprocessing kernel and the inference kernel so train/test pipelines stay identical.
"""
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

"""Slice-attention head over frozen DINOv2 slice features (shared by training and inference)."""
import torch
import torch.nn as nn


class Head(nn.Module):
    def __init__(s, d_in=768, d=256, n_out=12, depth=32, drop=0.2, per_finding=False):
        super().__init__()
        s.per_finding = per_finding; s.n_out = n_out
        s.proj = nn.Sequential(nn.LayerNorm(d_in), nn.Dropout(drop), nn.Linear(d_in, d))
        s.pos = nn.Parameter(torch.zeros(1, 1, depth, d)); s.plane = nn.Parameter(torch.zeros(1, 3, 1, d))
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, d * 2, drop, batch_first=True, norm_first=True), 2)
        if per_finding:      # one attention over the slice groups per finding (a tear may show on 2 slices, OA on many), then a per-finding read-out
            s.attn = nn.Linear(d, n_out)
            s.fuse = nn.Sequential(nn.LayerNorm(3 * d), nn.Dropout(drop), nn.Linear(3 * d, d), nn.GELU())
            s.out_w = nn.Parameter(torch.randn(n_out, d) * d ** -0.5); s.out_b = nn.Parameter(torch.zeros(n_out))
        else:
            s.attn = nn.Linear(d, 1)
            s.fuse = nn.Sequential(nn.LayerNorm(3 * d), nn.Dropout(drop), nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, n_out))

    def forward(s, x, present):
        B, Pn, D, _ = x.shape
        h = s.proj(x.float()) + s.pos + s.plane
        h = s.enc(h.view(B * Pn, D, -1))
        if s.per_finding:
            w = torch.softmax(s.attn(h), 1)                                        # [B*Pn, D, K]
            v = torch.einsum("bdk,bde->bke", w, h).view(B, Pn, s.n_out, -1)        # [B, Pn, K, d]
            v = v * present.view(B, Pn, 1, 1)
            z = s.fuse(v.permute(0, 2, 1, 3).reshape(B * s.n_out, -1)).view(B, s.n_out, -1)   # the fuse MLP is shared across findings
            return (z * s.out_w).sum(-1) + s.out_b
        w = torch.softmax(s.attn(h).squeeze(-1), 1)
        v = (h * w.unsqueeze(-1)).sum(1).view(B, Pn, -1) * present.unsqueeze(-1)
        return s.fuse(v.reshape(B, -1))

# ---------------- torchvision backbone factory shared by the training and inference scripts ----------------
# Supports resnet*, convnext_*, efficientnet_* / efficientnet_v2_*; weights are the unmodified torchvision .pth files from Kaggle model sources.
import torch
import torch.nn as nn


def strip_head(bb):
    """Replace the classifier with an identity so the backbone returns pooled features; returns the feature dimension."""
    if hasattr(bb, "fc"):                                         # resnet
        d = bb.fc.in_features; bb.fc = nn.Identity()
    elif hasattr(bb, "classifier") and isinstance(bb.classifier[-1], nn.Linear) and isinstance(bb.classifier[0], nn.Dropout):   # efficientnet
        d = bb.classifier[-1].in_features; bb.classifier = nn.Identity()
    elif hasattr(bb, "classifier"):                               # convnext: [LayerNorm2d, Flatten, Linear] -> keep norm + flatten
        d = bb.classifier[-1].in_features; bb.classifier[-1] = nn.Identity()
    else:
        raise ValueError("unsupported architecture")
    return d


def find_weights(arch):
    ck = None
    for r, dirs, fs in os.walk("/kaggle/input"):                  # pruned: never descend into the 820k-file image tree or the npz shards
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series") and not d.startswith("shard")]
        hit = [f for f in fs if f.startswith(arch) and f.endswith(".pth") and f[len(arch)] in "-_"]
        if hit: ck = os.path.join(r, hit[0]); break
    assert ck, f"no {arch}*.pth weights found"
    return ck


def build_backbone(arch, pretrained):
    import torchvision
    bb = torchvision.models.get_model(arch, weights=None)
    if pretrained: bb.load_state_dict(torch.load(find_weights(arch), map_location="cpu", weights_only=True))
    return bb, strip_head(bb)

# ---------------- inference for the fine-tuned model(s): average every ft_fold*.pt found under /kaggle/input ----------------
import glob, time, json
from concurrent.futures import ProcessPoolExecutor

ROOT = next((r for r in ("/kaggle/input/competitions/rsna-knee-abnormality-detection", "/kaggle/input/rsna-knee-abnormality-detection") if os.path.exists(r + "/sample_submission.csv")),
            "/kaggle/input/competitions/rsna-knee-abnormality-detection")   # Kaggle mounts the competition at either path depending on the session
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def find(name):
    for r, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        if name in files:
            return os.path.join(r, name)
    raise FileNotFoundError(name)


class CNNModel(nn.Module):                      # identical to the training definition, minus the pretrained-weight load (the checkpoint holds all weights)
    def __init__(s, arch, S_, G_, RES_, per_finding=False):
        super().__init__()
        s.S, s.G, s.RES = S_, G_, RES_
        s.bb, d = build_backbone(arch, pretrained=False)
        s.head = Head(d_in=d, depth=G_, per_finding=per_finding)
        s.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        s.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(s, v, present):
        B = v.shape[0]
        x = v.view(B * 3 * s.G, 3, s.S, s.S).float() / 255
        if s.RES != s.S: x = nn.functional.interpolate(x, size=(s.RES, s.RES), mode="bilinear", antialias=True, align_corners=False)
        f = s.bb((x - s.mean) / s.std)
        return s.head(f.view(B, 3, s.G, -1), present)


def spec_for(ckpt, state=None):
    """How to run one checkpoint. A config_fold*.json next to it (written by the training script) wins; otherwise the build-time defaults
    (ARCH_MAP / ARCH / RES / CACHE_PX) are used, which reproduces the older checkpoints. per_finding is detected from the weights themselves."""
    d = os.path.dirname(ckpt); cfgs = sorted(glob.glob(os.path.join(d, "config_fold*.json")))
    cfg = json.load(open(cfgs[0])) if cfgs else {}
    arch = cfg.get("arch") or next((a for k, a in ARCH_MAP.items() if k in ckpt.replace("\\", "/")), ARCH)
    pf = cfg["per_finding"] if "per_finding" in cfg else (bool(state is not None and "head.out_w" in state))
    return dict(path=ckpt, arch=arch, S=int(cfg.get("S", CACHE_PX)), G=int(cfg.get("G", 11)), RES=int(cfg.get("RES", RES)), pf=pf, from_cfg=bool(cfg))


def prep(uid):
    out = {}
    for S_ in SIZES:                             # one load per distinct cache size (usually just one)
        try: out[S_] = load_study_v2(ROOT, "test", uid, SERIES[SERIES.StudyInstanceUID == uid], size=S_)
        except Exception as e:
            print(f"WARN study {uid[-8:]} @ {S_}px failed entirely: {type(e).__name__}: {str(e)[:80]}", flush=True)
            out[S_] = (np.zeros((3, 11, 3, S_, S_), np.uint8), np.zeros(3, bool))
    return out


if __name__ == "__main__":
    t0 = time.time(); dev = "cuda"
    sub = pd.read_csv(f"{ROOT}/sample_submission.csv"); SERIES = pd.read_csv(f"{ROOT}/test_series.csv")
    uids = sub.StudyInstanceUID.tolist()
    ckpts = []                                   # pruned walk: never descend into the 820k-file competition image tree
    for r, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        ckpts += [os.path.join(r, f) for f in files if f.startswith("ft_fold") and f.endswith(".pt")]
    ckpts.sort()
    print(f"{len(uids)} test studies, {len(SERIES)} series | checkpoints: {[os.path.basename(os.path.dirname(c)) + '/' + os.path.basename(c) for c in ckpts]}", flush=True)
    assert ckpts, "no ft_fold*.pt found"
    specs, models = [], []
    for c in ckpts:
        state = torch.load(c, map_location="cpu"); sp = spec_for(c, state)
        m = CNNModel(sp["arch"], sp["S"], sp["G"], sp["RES"], sp["pf"]); m.load_state_dict(state, strict=True); models.append(m.to(dev).eval()); specs.append(sp)
        print(f"loaded {sp['arch']} S={sp['S']} G={sp['G']} RES={sp['RES']} per_finding={sp['pf']} ({'config' if sp['from_cfg'] else 'defaults'}) <- {os.path.basename(os.path.dirname(c))}/{os.path.basename(c)}", flush=True)
    SIZES = sorted({sp["S"] for sp in specs})
    print("distinct cache sizes needed:", SIZES, flush=True)

    preds = np.full((len(uids), 12), 0.5, np.float32); BS = 2
    with ProcessPoolExecutor(4) as ex:
        it = ex.map(prep, uids, chunksize=2)
        for s in range(0, len(uids), BS):
            batch = [next(it) for _ in range(min(BS, len(uids) - s))]
            vs = {S_: torch.from_numpy(np.stack([b[S_][0] for b in batch])).to(dev) for S_ in SIZES}
            p = torch.from_numpy(np.stack([b[SIZES[0]][1] for b in batch])).to(dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                outs = [torch.sigmoid(m(vs[sp["S"]], p).float()) for m, sp in zip(models, specs)]
                if TTA:                                   # flip coronal+axial left-right (still a plausible knee); sagittal is NOT flipped (would swap anterior/posterior)
                    for m, sp in zip(models, specs):
                        vf = vs[sp["S"]].clone(); vf[:, 1:] = vf[:, 1:].flip(-1); outs.append(torch.sigmoid(m(vf, p).float()))
                out = torch.stack(outs).mean(0).cpu().numpy()
            out[~p.any(1).cpu().numpy()] = 0.5
            preds[s:s + len(batch)] = out
            if (s // BS) % 100 == 0: print(f"{s + len(batch)}/{len(uids)} | {time.time() - t0:.0f}s", flush=True)
    sub[T] = preds
    assert list(sub.columns) == ["StudyInstanceUID"] + T and len(sub) == len(uids) and not sub.isna().any().any()
    sub.to_csv("/kaggle/working/submission.csv", index=False)
    print("wrote submission.csv", sub.shape, f"| {time.time() - t0:.0f}s"); print(sub[T].describe().loc[["mean", "min", "max"]].round(3).to_string())
