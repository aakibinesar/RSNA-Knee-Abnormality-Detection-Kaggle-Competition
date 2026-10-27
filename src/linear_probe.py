"""Diagnostic: how much signal is in the frozen features? Ridge on slice-averaged features, plane-concatenated.
Train on pseudo-labelled studies only; gold studies (never trained on) scored by the average of the fold models."""
import json, sys
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
train = pd.read_csv("data/train.csv"); gold = train.dropna(subset=T).set_index("StudyInstanceUID")[T]
lab = pd.read_csv("data/public_labels/stevenleehans__rsna-knee-llm-report-labels/llm_labels_v4_blend.csv").set_index("StudyInstanceUID")[T]


def macro(y, p):
    return float(np.mean([roc_auc_score(y[:, j], p[:, j]) for j in range(12) if 0 < y[:, j].sum() < len(y)]))


def run(name, feat_path, present_path, uids_path, alpha_list=(100, 1000, 10000)):
    f = np.load(feat_path).astype(np.float32); pres = np.load(present_path); uids = json.load(open(uids_path))
    m = pres[:, :, None, None] if f.ndim == 4 else None
    pooled = (f * m).sum(2) / np.maximum(1, 1) if False else f.mean(2)                  # mean over slices/groups -> [N,3,D]
    X = (pooled * pres[:, :, None]).reshape(len(uids), -1)
    ig = np.array([u in gold.index for u in uids]); Y = lab.reindex(uids).values; Yb = (Y > 0.5).astype(int)
    yg = gold.reindex([u for u in uids if u in gold.index]).values.astype(int)
    pool = np.where(~ig)[0]; gi = np.where(ig)[0]
    for a in alpha_list:
        oof = np.zeros((len(uids), 12)); gp = np.zeros((len(gi), 12))
        for tr, va in KFold(5, shuffle=True, random_state=0).split(pool):
            tr, va = pool[tr], pool[va]
            sc = StandardScaler().fit(X[tr]); r = Ridge(alpha=a).fit(sc.transform(X[tr]), Y[tr])
            oof[va] = r.predict(sc.transform(X[va])); gp += r.predict(sc.transform(X[gi])) / 5
        print(f"{name:10s} dim={X.shape[1]:5d} alpha={a:6d} | vs pseudo (OOF) {macro(Yb[pool], oof[pool]):.4f} | gold(58) {macro(yg, gp):.4f}", flush=True)


run("v1 224px", "kernels/04_features_full/out/features.npy", "kernels/04_features_full/out/present.npy", "kernels/04_features_full/out/uids.json")
run("v2 336px", "kernels/07_features2/out_full/features2.npy", "kernels/07_features2/out_full/present2.npy", "kernels/07_features2/out_full/uids2.json")
