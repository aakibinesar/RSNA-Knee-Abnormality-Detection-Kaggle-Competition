"""Is a nonlinear head on frozen pooled features better than ridge? PCA(256) -> MLP, same folds/yardsticks as the ridge probe."""
import json, sys, warnings
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from frozen_ensemble import macro, lab, gold, T, pooled
from sklearn.model_selection import KFold

fp, pp, up = "kernels/07_features2/out_full/features2.npy", "kernels/07_features2/out_full/present2.npy", "kernels/07_features2/out_full/uids2.json"
uids = json.load(open(up)); X = pooled(fp, pp)
ig = np.array([u in gold.index for u in uids]); Y = lab.reindex(uids).values; Yb = (Y > 0.5).astype(int)
yg = gold.reindex([uids[i] for i in np.where(ig)[0]]).values.astype(int); pool, gi = np.where(~ig)[0], np.where(ig)[0]
for ncomp, hid, alpha in ((256, 512, 1e-2), (256, 512, 1.0), (128, 256, 1.0)):
    oof, gp = np.zeros((len(uids), 12)), np.zeros((len(gi), 12))
    for tr, va in KFold(5, shuffle=True, random_state=0).split(pool):
        tr, va = pool[tr], pool[va]
        sc = StandardScaler().fit(X[tr]); pca = PCA(ncomp, random_state=0).fit(sc.transform(X[tr]))
        Z = lambda idx: pca.transform(sc.transform(X[idx])) / np.sqrt(pca.explained_variance_[:1])
        m = MLPRegressor(hidden_layer_sizes=(hid,), alpha=alpha, early_stopping=True, validation_fraction=0.1, n_iter_no_change=8, max_iter=200, random_state=0).fit(Z(tr), Y[tr])
        oof[va] = m.predict(Z(va)); gp += m.predict(Z(gi)) / 5
    print(f"MLP pca={ncomp} hidden={hid} alpha={alpha:g} | pseudo {macro(Yb[pool], oof[pool]):.4f} | gold {macro(yg, gp):.4f}", flush=True)
