
# ---------------- inference for the fine-tuned model(s): average every ft_fold*.pt found under /kaggle/input ----------------
import glob, time, json
from concurrent.futures import ProcessPoolExecutor

ROOT = next((r for r in ("/kaggle/input/competitions/rsna-knee-abnormality-detection", "/kaggle/input/rsna-knee-abnormality-detection") if os.path.exists(r + "/sample_submission.csv")),
            "/kaggle/input/competitions/rsna-knee-abnormality-detection")   # Kaggle mounts the competition at either path depending on the session
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def find(name):
    for r, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        if name in files:
            return os.path.join(r, name)
    raise FileNotFoundError(name)


class CNNModel(nn.Module):                      # identical to the training definition, minus the pretrained-weight load (the checkpoint holds all weights)
    def __init__(s, arch, S_, G_, RES_, per_finding=False):
        super().__init__()
        s.S, s.G, s.RES = S_, G_, RES_
        s.bb, d = build_backbone(arch, pretrained=False)
        s.head = Head(d_in=d, depth=G_, per_finding=per_finding)
        s.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        s.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(s, v, present):
        B = v.shape[0]
        x = v.view(B * 3 * s.G, 3, s.S, s.S).float() / 255
        if s.RES != s.S: x = nn.functional.interpolate(x, size=(s.RES, s.RES), mode="bilinear", antialias=True, align_corners=False)
        f = s.bb((x - s.mean) / s.std)
        return s.head(f.view(B, 3, s.G, -1), present)


def spec_for(ckpt, state=None):
    """How to run one checkpoint. A config_fold*.json next to it (written by the training script) wins; otherwise the build-time defaults
    (ARCH_MAP / ARCH / RES / CACHE_PX) are used, which reproduces the older checkpoints. per_finding is detected from the weights themselves."""
    d = os.path.dirname(ckpt); cfgs = sorted(glob.glob(os.path.join(d, "config_fold*.json")))
    cfg = json.load(open(cfgs[0])) if cfgs else {}
    arch = cfg.get("arch") or next((a for k, a in ARCH_MAP.items() if k in ckpt.replace("\\", "/")), ARCH)
    pf = cfg["per_finding"] if "per_finding" in cfg else (bool(state is not None and "head.out_w" in state))
    return dict(path=ckpt, arch=arch, S=int(cfg.get("S", CACHE_PX)), G=int(cfg.get("G", 11)), RES=int(cfg.get("RES", RES)), pf=pf, from_cfg=bool(cfg))


def prep(uid):
    out = {}
    for S_ in SIZES:                             # one load per distinct cache size (usually just one)
        try: out[S_] = load_study_v2(ROOT, "test", uid, SERIES[SERIES.StudyInstanceUID == uid], size=S_)
        except Exception as e:
            print(f"WARN study {uid[-8:]} @ {S_}px failed entirely: {type(e).__name__}: {str(e)[:80]}", flush=True)
            out[S_] = (np.zeros((3, 11, 3, S_, S_), np.uint8), np.zeros(3, bool))
    return out


if __name__ == "__main__":
    t0 = time.time(); dev = "cuda"
    sub = pd.read_csv(f"{ROOT}/sample_submission.csv"); SERIES = pd.read_csv(f"{ROOT}/test_series.csv")
    uids = sub.StudyInstanceUID.tolist()
    ckpts = []                                   # pruned walk: never descend into the 820k-file competition image tree
    for r, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        ckpts += [os.path.join(r, f) for f in files if f.startswith("ft_fold") and f.endswith(".pt")]
    ckpts.sort()
    print(f"{len(uids)} test studies, {len(SERIES)} series | checkpoints: {[os.path.basename(os.path.dirname(c)) + '/' + os.path.basename(c) for c in ckpts]}", flush=True)
    assert ckpts, "no ft_fold*.pt found"
    specs, models = [], []
    for c in ckpts:
        state = torch.load(c, map_location="cpu"); sp = spec_for(c, state)
        m = CNNModel(sp["arch"], sp["S"], sp["G"], sp["RES"], sp["pf"]); m.load_state_dict(state, strict=True); models.append(m.to(dev).eval()); specs.append(sp)
        print(f"loaded {sp['arch']} S={sp['S']} G={sp['G']} RES={sp['RES']} per_finding={sp['pf']} ({'config' if sp['from_cfg'] else 'defaults'}) <- {os.path.basename(os.path.dirname(c))}/{os.path.basename(c)}", flush=True)
    SIZES = sorted({sp["S"] for sp in specs})
    print("distinct cache sizes needed:", SIZES, flush=True)

    preds = np.full((len(uids), 12), 0.5, np.float32); BS = 2
    with ProcessPoolExecutor(4) as ex:
        it = ex.map(prep, uids, chunksize=2)
        for s in range(0, len(uids), BS):
            batch = [next(it) for _ in range(min(BS, len(uids) - s))]
            vs = {S_: torch.from_numpy(np.stack([b[S_][0] for b in batch])).to(dev) for S_ in SIZES}
            p = torch.from_numpy(np.stack([b[SIZES[0]][1] for b in batch])).to(dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                outs = [torch.sigmoid(m(vs[sp["S"]], p).float()) for m, sp in zip(models, specs)]
                if TTA:                                   # flip coronal+axial left-right (still a plausible knee); sagittal is NOT flipped (would swap anterior/posterior)
                    for m, sp in zip(models, specs):
                        vf = vs[sp["S"]].clone(); vf[:, 1:] = vf[:, 1:].flip(-1); outs.append(torch.sigmoid(m(vf, p).float()))
                out = torch.stack(outs).mean(0).cpu().numpy()
            out[~p.any(1).cpu().numpy()] = 0.5
            preds[s:s + len(batch)] = out
            if (s // BS) % 100 == 0: print(f"{s + len(batch)}/{len(uids)} | {time.time() - t0:.0f}s", flush=True)
    sub[T] = preds
    assert list(sub.columns) == ["StudyInstanceUID"] + T and len(sub) == len(uids) and not sub.isna().any().any()
    sub.to_csv("/kaggle/working/submission.csv", index=False)
    print("wrote submission.csv", sub.shape, f"| {time.time() - t0:.0f}s"); print(sub[T].describe().loc[["mean", "min", "max"]].round(3).to_string())
