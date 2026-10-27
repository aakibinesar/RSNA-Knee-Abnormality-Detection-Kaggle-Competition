"""CPU-only analysis of the five ResNet-50 folds + teacher-student label builder.

1. pools the held-out (OOF) predictions of folds 0-4, scores them per finding against the LLM pseudo-labels and the gold-58
2. checks whether OOF predictions agree with an INDEPENDENT label source better than the primary labels do (denoising evidence)
3. writes blended teacher-student label tables  data/ts_labels_a{ALPHA}.csv  = alpha * OOF + (1-alpha) * primary LLM labels
Run from the project root:  python src/oof_analysis.py
"""
import json, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
FOLD_DIRS = ["kernels/15_cnn_r50lr1/out", "kernels/15_cnn_f1/out", "kernels/15_cnn_f2/out", "kernels/15_cnn_f3/out", "kernels/15_cnn_f4/out"]
PRIMARY = "data/public_labels/stevenleehans__rsna-knee-llm-report-labels/llm_labels_v4_blend.csv"
INDEP = {"qwen3-8b": "data/public_labels/laymond__rsna-knee-abnormality-qwen3-8b-weak-labels/qwen_knee_weak_labels.csv",
         "pilkwang": "data/public_labels/pilkwang__rsna-knee-llm-labels/report_labels_v2.csv",
         "gpt56sol": "data/public_labels/lixin73__rsna-knee-llm-report-labels-sol56/labels_llm_gpt56sol.csv"}


def auc(y, p):
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan


def per_target(Y, P): return np.array([auc(Y[:, j], P[:, j]) for j in range(12)])


def load_labels(path):
    df = pd.read_csv(path)
    if "StudyInstanceUID" not in df.columns or not all(t in df.columns for t in T): return None
    return df.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")[T].apply(pd.to_numeric, errors="coerce")


def read_json(p): return json.load(open(p, encoding="utf-8-sig"))


train = pd.read_csv("data/train.csv"); gold = train.dropna(subset=T).set_index("StudyInstanceUID")[T]
prim = load_labels(PRIMARY)

# ---- pool OOF predictions --------------------------------------------------------------------------------------------
uids, preds, gold_preds = [], [], []
for k, d in enumerate(FOLD_DIRS):
    m = read_json(f"{d}/ft_metrics_fold{k}.json")
    uids += m["val_uids"]; preds.append(np.load(f"{d}/val_pred_fold{k}.npy")); gold_preds.append(np.load(f"{d}/gold_pred_fold{k}.npy"))
oof = pd.DataFrame(np.vstack(preds), index=uids, columns=T)
assert oof.index.is_unique, "a study appears in two folds"
print(f"pooled OOF studies: {len(oof)}  (gold excluded: {oof.index.isin(gold.index).sum()} gold rows in OOF - expected 0)")

Yb = (prim.reindex(oof.index).values > 0.5).astype(int)
a_oof = per_target(Yb, oof.values)
print("\nOOF vs binarised primary pseudo-labels (per finding; prevalence in brackets):")
for t, a, pv in zip(T, a_oof, Yb.mean(0)): print(f"  {t:17s} {a:.3f}  [{pv:.3f}]")
print(f"  macro {np.nanmean(a_oof):.4f}")

# ---- gold-58: each fold's model, and the mean ------------------------------------------------------------------------
gy = gold.values.astype(int); gm = np.mean(gold_preds, 0)
print("\ngold-58 per finding:  mean of 5 folds | best single | worst single | prevalence in gold")
singles = np.array([per_target(gy, g) for g in gold_preds])
a_gm = per_target(gy, gm)
for j, t in enumerate(T): print(f"  {t:17s} {a_gm[j]:.3f} | {np.nanmax(singles[:, j]):.3f} | {np.nanmin(singles[:, j]):.3f} | n_pos={gy[:, j].sum()}")
print(f"  macro mean-of-5 {np.nanmean(a_gm):.4f} | singles {np.round(np.nanmean(singles, 1), 3)}")

# ---- does the OOF model carry signal beyond its training labels? -----------------------------------------------------
print("\nagreement with an independent label source B (AUC vs B>0.5), macro over findings with both classes:")
for name, path in INDEP.items():
    if not os.path.exists(path): continue
    B = load_labels(path)
    if B is None: print(f"  {name}: schema mismatch"); continue
    common = oof.index.intersection(B.index)
    if len(common) < 500: print(f"  {name}: only {len(common)} common studies"); continue
    Bb = (B.reindex(common).fillna(0).values > 0.5).astype(int)
    a_model = per_target(Bb, oof.reindex(common).values)
    a_prim = per_target(Bb, prim.reindex(common).values)
    print(f"  {name:9s} n={len(common)} | primary labels vs B {np.nanmean(a_prim):.4f} | OOF model vs B {np.nanmean(a_model):.4f}")
    for j, t in enumerate(T):
        if not np.isnan(a_model[j]): print(f"      {t:17s} primary {a_prim[j]:.3f}  model {a_model[j]:.3f}")

# ---- build blended labels --------------------------------------------------------------------------------------------
os.makedirs("data", exist_ok=True)
for alpha in (0.3, 0.5, 0.7):
    out = prim.copy()
    idx = oof.index
    out.loc[idx, T] = alpha * oof.loc[idx, T].values + (1 - alpha) * prim.loc[idx, T].values
    out.reset_index().to_csv(f"data/ts_labels_a{int(alpha*100)}.csv", index=False)
    print(f"wrote data/ts_labels_a{int(alpha*100)}.csv  rows={len(out)}  replaced={len(idx)}  mean abs change {np.abs(out.loc[idx].values - prim.loc[idx].values).mean():.3f}")
