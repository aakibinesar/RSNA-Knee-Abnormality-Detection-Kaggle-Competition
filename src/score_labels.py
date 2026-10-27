"""Score candidate report-derived label files against the 58 gold-labelled training studies."""
import glob, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
train = pd.read_csv("data/train.csv")
gold = train.dropna(subset=TARGETS).set_index("StudyInstanceUID")[TARGETS]


def score(path):
    df = pd.read_csv(path).set_index("StudyInstanceUID")
    if not all(t in df.columns for t in TARGETS):
        return None
    cand = df.reindex(gold.index)[TARGETS]
    if cand.isna().any().any():
        return None
    per = {t: roc_auc_score(gold[t], cand[t]) for t in TARGETS}
    return float(np.mean(list(per.values()))), per, len(df)


rows = []
for p in sorted(glob.glob("data/public_labels/*/*.csv")):
    r = score(p)
    if r is None:
        continue
    macro, per, n = r
    rows.append((macro, p.replace("data/public_labels/", ""), n, per))
rows.sort(reverse=True)
print(f"gold studies: {len(gold)}")
for macro, p, n, per in rows:
    print(f"{macro:.4f}  n={n}  {p}")
print("\nper-target AUC of best:", {k: round(v, 3) for k, v in rows[0][3].items()})
