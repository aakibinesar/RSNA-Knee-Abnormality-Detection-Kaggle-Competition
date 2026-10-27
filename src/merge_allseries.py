"""Merge the two interleaved all-series feature shards and verify they cover train_series.csv exactly once."""
import os
import numpy as np, pandas as pd

parts = [f"kernels/14_allseries_h{i}/out" for i in (0, 1)]
feats = [np.load(f"{p}/feats_all.npy") for p in parts]
idx = [pd.read_csv(f"{p}/series_index.csv") for p in parts]
for p, f, i in zip(parts, feats, idx):
    assert len(f) == len(i), (p, f.shape, len(i))
F = np.concatenate(feats); I = pd.concat(idx, ignore_index=True)
ref = pd.read_csv("data/train_series.csv")
assert I.SeriesInstanceUID.is_unique, "duplicate series across shards"
assert set(I.SeriesInstanceUID) == set(ref.SeriesInstanceUID) and len(I) == len(ref), "coverage mismatch"
assert I.ok.all(), f"{(~I.ok).sum()} series failed to read"
assert not np.isnan(F.astype(np.float32)).any()
# restore train_series.csv row order
pos = {s: k for k, s in enumerate(I.SeriesInstanceUID)}
order = [pos[s] for s in ref.SeriesInstanceUID]
F, I = F[order], I.iloc[order].reset_index(drop=True)
assert (I.SeriesInstanceUID.values == ref.SeriesInstanceUID.values).all()
os.makedirs("kernels/14_allseries_merged", exist_ok=True)
np.save("kernels/14_allseries_merged/feats_all.npy", F); I.to_csv("kernels/14_allseries_merged/series_index.csv", index=False)
print("merged OK:", F.shape, F.dtype, f"{F.nbytes/1e6:.0f} MB | studies {I.StudyInstanceUID.nunique()} | all ok | order matches train_series.csv")
print(I.Anatomical_Plane.value_counts().to_dict())
