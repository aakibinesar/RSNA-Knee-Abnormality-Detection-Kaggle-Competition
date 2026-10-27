"""Score every candidate public label table (csv/parquet with StudyInstanceUID + the 12 targets) against the 58 gold studies."""
import glob, os, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
train = pd.read_csv("data/train.csv"); gold = train.dropna(subset=T).set_index("StudyInstanceUID")[T]


def load(path):
    try:
        df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    except Exception:
        return None
    if "StudyInstanceUID" not in df.columns:
        for c in ("study_id", "uid", "StudyUID", "study_uid"):
            if c in df.columns: df = df.rename(columns={c: "StudyInstanceUID"}); break
    if "StudyInstanceUID" not in df.columns or not all(t in df.columns for t in T):
        return None
    return df.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")[T].apply(pd.to_numeric, errors="coerce")


def macro(y, p): return float(np.mean([roc_auc_score(y[:, j], p[:, j]) for j in range(12) if 0 < y[:, j].sum() < len(y)]))


rows, tables = [], {}
for p in sorted(glob.glob("data/public_labels/*/*.csv") + glob.glob("data/public_labels/*/*.parquet")):
    df = load(p)
    if df is None: continue
    g = df.reindex(gold.index)
    g = g.fillna(g.mean()).fillna(0.5)          # tables may miss some gold ids or whole columns: fill by column mean, then neutral 0.5; coverage is reported
    cov = df.reindex(gold.index).notna().all(axis=1).mean()
    name = p.replace("data/public_labels/", "").replace("\\", "/")
    rows.append((macro(gold.values.astype(int), g.values), name, len(df), cov)); tables[name] = df
rows.sort(reverse=True)
print(f"gold studies: {len(gold)}")
for s, n, ln, cov in rows: print(f"{s:.4f}  rows={ln:5d}  gold-coverage={cov:.2f}  {n}")
