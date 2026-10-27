"""Evaluate frozen-feature ridge probes per backbone and their rank-average.
Train on pseudo-labelled studies only (5-fold); gold-58 are never trained on and are scored by the mean of the fold models.
Usage: python src/frozen_ensemble.py [name=path:present:uids ...]   (paths relative to rsna-knee/)"""
import json, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
train = pd.read_csv("data/train.csv"); gold = train.dropna(subset=T).set_index("StudyInstanceUID")[T]
lab = pd.read_csv("data/public_labels/stevenleehans__rsna-knee-llm-report-labels/llm_labels_v4_blend.csv").set_index("StudyInstanceUID")[T]
ALPHAS = (300, 1000, 3000, 10000)


def macro(y, p): return float(np.mean([roc_auc_score(y[:, j], p[:, j]) for j in range(12) if 0 < y[:, j].sum() < len(y)]))
def rank01(p): return np.column_stack([rankdata(p[:, j]) / len(p) for j in range(p.shape[1])])


def pooled(path, pres_path):
    f = np.load(path).astype(np.float32); pres = np.load(pres_path)
    return (f.mean(2) * pres[:, :, None]).reshape(len(f), -1)             # mean over triplet groups, concat planes


def probe(name, X, uids, alpha):
    ig = np.array([u in gold.index for u in uids]); Y = lab.reindex(uids).values; Yb = (Y > 0.5).astype(int)
    pool, gi = np.where(~ig)[0], np.where(ig)[0]
    oof, gp = np.zeros((len(uids), 12)), np.zeros((len(gi), 12))
    for tr, va in KFold(5, shuffle=True, random_state=0).split(pool):
        tr, va = pool[tr], pool[va]
        sc = StandardScaler().fit(X[tr]); r = Ridge(alpha=alpha).fit(sc.transform(X[tr]), Y[tr])
        oof[va] = r.predict(sc.transform(X[va])); gp += r.predict(sc.transform(X[gi])) / 5
    return oof[pool], gp, Yb[pool], gold.reindex([uids[i] for i in gi]).values.astype(int)


def main(specs):
    results = {}
    for name, (fp, pp, up) in specs.items():
        uids = json.load(open(up)); X = pooled(fp, pp); best = None
        for a in ALPHAS:
            o, g, yb, yg = probe(name, X, uids, a)
            s = macro(yb, o); print(f"{name:12s} dim={X.shape[1]:5d} alpha={a:6d} | vs pseudo {s:.4f} | gold {macro(yg, g):.4f}", flush=True)
            if best is None or s > best[0]: best = (s, a, o, g, yb, yg)
        results[name] = best; print(f"  -> {name}: chose alpha={best[1]} by pseudo agreement\n", flush=True)
    names = list(results); yb, yg = results[names[0]][4], results[names[0]][5]
    if len(names) > 1:
        eo = np.mean([rank01(results[n][2]) for n in names], 0); eg = np.mean([rank01(results[n][3]) for n in names], 0)
        print(f"ENSEMBLE of {names}: vs pseudo {macro(yb, eo):.4f} | gold {macro(yg, eg):.4f}")
        print("  per-target gold:", {t: round(float(roc_auc_score(yg[:, j], eg[:, j])), 3) for j, t in enumerate(T) if 0 < yg[:, j].sum() < len(yg)})


if __name__ == "__main__":
    specs = {}
    for a in sys.argv[1:]:
        n, rest = a.split("="); fp, pp, up = rest.split(":"); specs[n] = (fp, pp, up)
    main(specs)
