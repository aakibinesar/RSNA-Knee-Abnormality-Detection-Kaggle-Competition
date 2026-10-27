"""Does the way triplet-group features are pooled matter? mean (baseline) vs max vs mean||max, on the 3-series choice and on all-series-per-plane."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from frozen_ensemble import probe, macro
from knee_prep import choose_series, PLANES

F3 = np.load("kernels/14_allseries_merged/feats_all.npy").astype(np.float32)               # [24371, 11 groups, 1152]
I = pd.read_csv("kernels/14_allseries_merged/series_index.csv")
uids = sorted(I.StudyInstanceUID.unique()); pos = {s: k for k, s in enumerate(I.SeriesInstanceUID)}; by = {k: g for k, g in I.groupby("StudyInstanceUID")}
pools = {"mean": F3.mean(1), "max": F3.max(1), "mean||max": np.concatenate([F3.mean(1), F3.max(1)], 1)}
for pname, Fp in pools.items():
    D = Fp.shape[1]; XA = np.zeros((len(uids), 3 * D), np.float32); XB = np.zeros_like(XA)
    for n, u in enumerate(uids):
        g = by[u]; ch = choose_series(g)
        for p, plane in enumerate(PLANES):
            gp = g[g.Anatomical_Plane == plane]
            if plane in ch: XA[n, p * D:(p + 1) * D] = Fp[pos[ch[plane]]]
            if len(gp): XB[n, p * D:(p + 1) * D] = Fp[[pos[s] for s in gp.SeriesInstanceUID]].max(0) if pname == "max" else Fp[[pos[s] for s in gp.SeriesInstanceUID]].mean(0)
    for vname, X in (("3-series", XA), ("all-series", XB)):
        best = None
        for a in (1000, 3000, 10000, 30000):
            o, g, yb, yg = probe(pname, X, uids, a); s = macro(yb, o)
            if best is None or s > best[0]: best = (s, macro(yg, g), a)
        print(f"group-pool {pname:10s} | {vname:10s} dim={X.shape[1]:5d} | pseudo {best[0]:.4f} | gold {best[1]:.4f} | alpha {best[2]}", flush=True)
