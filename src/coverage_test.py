"""Does using ALL series per study beat the 3-series (one per plane) choice? Ridge probe on frozen DINOv2-S 280px features.
A = same 3 series the earlier pipeline chose (control, same resolution);  B = mean over all series in each plane;
C = mean per (plane x fluid-sensitive flag) with presence indicators."""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from frozen_ensemble import probe, macro, T, gold
from knee_prep import choose_series, PLANES

F = np.load("kernels/14_allseries_merged/feats_all.npy").astype(np.float32).mean(1)        # [24371, 1152]  mean over triplet groups
I = pd.read_csv("kernels/14_allseries_merged/series_index.csv")
uids = sorted(I.StudyInstanceUID.unique()); pos = {s: k for k, s in enumerate(I.SeriesInstanceUID)}
by = {k: g for k, g in I.groupby("StudyInstanceUID")}
D = F.shape[1]
print("flag agreement Fluid==Fat:", float((I.Fluid_Sensitive == I.Fat_Suppression).mean()), "| fluid share:", round(float(I.Fluid_Sensitive.mean()), 3))
XA = np.zeros((len(uids), 3 * D), np.float32); XB = np.zeros_like(XA); XC = np.zeros((len(uids), 6 * D + 6), np.float32)
for n, u in enumerate(uids):
    g = by[u]; ch = choose_series(g)
    for p, plane in enumerate(PLANES):
        gp = g[g.Anatomical_Plane == plane]
        if plane in ch: XA[n, p * D:(p + 1) * D] = F[pos[ch[plane]]]
        if len(gp): XB[n, p * D:(p + 1) * D] = F[[pos[s] for s in gp.SeriesInstanceUID]].mean(0)
        for fl in (1, 0):
            gs = gp[gp.Fluid_Sensitive == fl]; c = p * 2 + (0 if fl == 1 else 1)
            if len(gs): XC[n, c * D:(c + 1) * D] = F[[pos[s] for s in gs.SeriesInstanceUID]].mean(0); XC[n, 6 * D + c] = 1
print("series per study (mean):", round(len(I) / len(uids), 2), "| studies with a non-fluid series in >=1 plane:", round(float((XC[:, 6 * D + 1] + XC[:, 6 * D + 3] + XC[:, 6 * D + 5] > 0).mean()), 3))
res = {}
for name, X, alphas in (("A 3-series (control)", XA, (1000, 3000, 10000)), ("B all series, mean/plane", XB, (1000, 3000, 10000)), ("C plane x fluid", XC, (3000, 10000, 30000))):
    best = None
    for a in alphas:
        o, g, yb, yg = probe(name, X, uids, a); s = macro(yb, o); gs = macro(yg, g)
        print(f"{name:26s} dim={X.shape[1]:6d} alpha={a:6d} | vs pseudo {s:.4f} | gold {gs:.4f}", flush=True)
        if best is None or s > best[0]: best = (s, gs, a)
    res[name] = best
print("\nBEST per variant (alpha chosen by pseudo agreement):")
for k, (s, gs, a) in res.items(): print(f"  {k:26s} pseudo {s:.4f} | gold {gs:.4f} | alpha {a}")
