
# ---------------- v2 image cache: 2.5D triplets, compressed, sharded ----------------
import time, json
from concurrent.futures import ProcessPoolExecutor

ROOT = "/kaggle/input/competitions/rsna-knee-abnormality-detection"
if not os.path.exists(f"{ROOT}/train_series.csv"):            # diagnostic + fallback: the competition mount was missing on some sessions
    print("ROOT missing; /kaggle/input =", os.listdir("/kaggle/input") if os.path.exists("/kaggle/input") else "absent", flush=True)
    for r, ds, fs in os.walk("/kaggle/input"):
        if r.count("/") > 4: ds[:] = []; continue
        print("  ", r, ds[:6], fs[:6], flush=True)
        if "train_series.csv" in fs: ROOT = r; print("found train_series.csv in", r, flush=True); break
OUT = f"/kaggle/working/shard{SHARD}"
os.makedirs(OUT, exist_ok=True)
SIZE_C = CACHE_PX                                   # pixels per side of every cached slice (set by build.sh; 280 for the existing caches)


def work(uid):
    try:
        vols, present = load_study_v2(ROOT, "train", uid, BY_STUDY[uid], size=SIZE_C)
        np.savez_compressed(f"{OUT}/{uid}.npz", vols=vols, present=present)
        return uid, present.tolist(), None
    except Exception as e:
        return uid, None, f"{type(e).__name__}: {str(e)[:100]}"


SERIES = pd.read_csv(f"{ROOT}/train_series.csv")
BY_STUDY = {k: g for k, g in SERIES.groupby("StudyInstanceUID")}

if __name__ == "__main__":
    studies = sorted(BY_STUDY)[SHARD::NSHARDS]
    if LIMIT: studies = studies[:LIMIT]
    print(f"shard {SHARD}/{NSHARDS}: {len(studies)} studies @ {SIZE_C}px", flush=True)
    t0, done, errs, manifest = time.time(), 0, [], {}
    with ProcessPoolExecutor(4) as ex:
        for uid, present, err in ex.map(work, studies, chunksize=4):
            done += 1
            if err: errs.append((uid, err))
            else: manifest[uid] = present
            if done % 100 == 0 or done == len(studies):
                gb = sum(os.path.getsize(f"{OUT}/{f}") for f in os.listdir(OUT)) / 1e9
                print(f"{done}/{len(studies)} | {time.time()-t0:.0f}s | out {gb:.2f} GB | errors {len(errs)}", flush=True)
                if gb > 17.5:
                    print("STOP: output near the 20 GB limit"); break
    json.dump(manifest, open(f"{OUT}/manifest.json", "w"))
    print("errors:", errs[:5], "| planes present:", np.array(list(manifest.values())).sum(0).tolist(), "of", len(manifest))
