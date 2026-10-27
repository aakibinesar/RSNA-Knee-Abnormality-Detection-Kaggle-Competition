
# ---------------- fine-tune DINOv2-S (last K blocks) + slice-attention head on the 2.5D cache ----------------
# FOLD selects one of 5 folds of the pseudo-labelled pool as validation; gold studies are never trained on and are scored
# every epoch, so each fold run gives an honest gold-58 number.
import glob, json, time, math
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
dev = "cuda"


def find(name):
    for r, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series") and not d.startswith("shard")]
        if name in files:
            return os.path.join(r, name)
    raise FileNotFoundError(name)


paths = {}
for r, dirs, files in os.walk("/kaggle/input"):
    dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
    for f in files:
        if f.endswith(".npz"):
            paths[f[:-4]] = os.path.join(r, f)
uids = sorted(paths)
if LIMIT: uids = uids[:LIMIT]
_z = np.load(paths[uids[0]])["vols"]; G, S = int(_z.shape[1]), int(_z.shape[-1])          # groups per plane and pixels per side are read from the cache, never hard-coded
print(f"cache slice size S={S}, groups G={G}, network input RES={RES}", flush=True)
json.dump(dict(arch=ARCH, S=S, G=G, RES=RES, per_finding=bool(PF), fold=FOLD, epochs=EPOCHS, bb_lr=BB_LR, head_lr=HEAD_LR, label_file=LABEL_FILE), open(f"/kaggle/working/config_fold{FOLD}.json", "w"))   # read by the inference script: the checkpoint says how it must be run
train = pd.read_csv(find("train.csv"))
lab = pd.read_csv(find(LABEL_FILE)).set_index("StudyInstanceUID")[T]
gold = train.dropna(subset=T).set_index("StudyInstanceUID")[T]
N = len(uids); SMOKE = N < 200
is_gold = np.array([u in gold.index for u in uids])
Y = lab.reindex(uids).values.astype(np.float32)
pool = np.where(~is_gold)[0]; gi = np.where(is_gold)[0]
tr_idx, va_idx = list(KFold(5, shuffle=True, random_state=0).split(pool))[FOLD]
tr_idx, va_idx = pool[tr_idx], pool[va_idx]
if SMOKE: EPOCHS = 1
print(f"cache studies={N} gold={len(gi)} train={len(tr_idx)} val={len(va_idx)} fold={FOLD} epochs={EPOCHS} K={K_UNFREEZE}", flush=True)


class DS(Dataset):
    def __init__(s, idx): s.idx = idx
    def __len__(s): return len(s.idx)
    def __getitem__(s, i):
        j = s.idx[i]; z = np.load(paths[uids[j]])
        return torch.from_numpy(z["vols"]), torch.from_numpy(z["present"]), torch.from_numpy(Y[j])


class CNNModel(nn.Module):
    """torchvision CNN (ImageNet weights, loaded offline) applied to every 2.5D triplet; slice-attention head on top."""
    def __init__(s, arch):
        super().__init__()
        s.bb, d = build_backbone(arch, pretrained=True)
        s.head = Head(d_in=d, depth=G, per_finding=bool(PF))
        s.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        s.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(s, v, present):                                   # v uint8 [B,3,G,3,S,S]
        B = v.shape[0]
        x = v.view(B * 3 * G, 3, S, S).float() / 255
        if RES != S: x = nn.functional.interpolate(x, size=(RES, RES), mode="bilinear", antialias=True, align_corners=False)
        f = s.bb((x - s.mean) / s.std)
        return s.head(f.view(B, 3, G, -1), present)


def mirror_batch(v, sel):
    """Mirror the knee (left<->right) for the studies in `sel`. v: [B, 3 planes (sag, cor, ax), G, 3 slices, S, S].
    Coronal and axial images are flipped horizontally; sagittal images stay as they are but the medial<->lateral order of the slice groups
    (and of the 3 neighbouring slices inside each group) is reversed. Label-preserving: a mirrored left knee is a plausible right knee."""
    if not bool(sel.any()): return v
    v = v.clone(); w = v[sel]
    w[:, 1:] = w[:, 1:].flip(-1)
    w[:, 0] = w[:, 0].flip(1).flip(2)
    v[sel] = w
    return v


def macro_auc(y, p):
    a = [roc_auc_score(y[:, j], p[:, j]) for j in range(y.shape[1]) if 0 < y[:, j].sum() < len(y)]
    return float(np.mean(a)) if a else float("nan")


@torch.no_grad()
def predict(m, idx):
    m.eval(); out = []
    for v, p, _ in DataLoader(DS(idx), batch_size=2, num_workers=4):
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(torch.sigmoid(m(v.to(dev), p.to(dev)).float()).cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 12))


torch.manual_seed(FOLD); np.random.seed(FOLD)
m = CNNModel(ARCH).to(dev)
bb_params = [p for p in m.bb.parameters() if p.requires_grad]
print(f"trainable backbone params {sum(p.numel() for p in bb_params)/1e6:.1f}M | head {sum(p.numel() for p in m.head.parameters())/1e6:.2f}M", flush=True)
opt = torch.optim.AdamW([{"params": bb_params, "lr": BB_LR}, {"params": m.head.parameters(), "lr": HEAD_LR}], weight_decay=1e-2)
bpe = len(tr_idx) // BS                                   # micro-batches per epoch (drop_last)
spe = max(1, bpe // ACC)                                  # optimizer steps per epoch
total = EPOCHS * spe; w_head = max(1, int(0.05 * total)); w_bb = WARM_EPOCHS * spe; ramp = spe   # backbone idle, then 1-epoch linear ramp, then cosine


def f_head(t): return (t + 1) / w_head if t < w_head else 0.5 * (1 + math.cos(math.pi * (t - w_head) / max(1, total - w_head)))
def f_bb(t):
    if t < w_bb: return 0.0
    if t < w_bb + ramp: return (t - w_bb + 1) / ramp
    return 0.5 * (1 + math.cos(math.pi * (t - w_bb - ramp) / max(1, total - w_bb - ramp)))


sched = torch.optim.lr_scheduler.LambdaLR(opt, [f_bb, f_head])
scaler = torch.amp.GradScaler()
gold_y = gold.reindex([uids[i] for i in gi]).values.astype(int) if len(gi) else None
best, history, t_start, top = -1, [], time.time(), []
for ep in range(EPOCHS):
    m.train(); t0 = time.time(); run = 0.0; gn = torch.tensor(0.0)
    for bi, (v, p, y) in enumerate(DataLoader(DS(tr_idx), batch_size=BS, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)):
        v, p, y = v.to(dev, non_blocking=True), p.to(dev), y.to(dev)
        v = (v.float() * (1 + 0.1 * torch.randn(v.shape[0], 3, 1, 1, 1, 1, device=dev)) + 8 * torch.randn(v.shape[0], 3, 1, 1, 1, 1, device=dev)).clamp(0, 255)  # brightness/contrast jitter per plane
        if MIRROR: v = mirror_batch(v, torch.rand(v.shape[0], device=v.device) < 0.5)        # left<->right mirror of half the studies
        with torch.autocast("cuda", dtype=torch.float16):
            loss = nn.functional.binary_cross_entropy_with_logits(m(v, p).float(), y)
        scaler.scale(loss / ACC).backward()
        if (bi + 1) % ACC == 0:
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_([q for g in opt.param_groups for q in g["params"]], 1.0)
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(set_to_none=True)
        run += loss.item()
        if bi % 200 == 0 or (SMOKE and bi % 4 == 0):
            print(f"ep{ep} step {bi}/{bpe} | loss {run/(bi+1):.4f} | gradnorm {float(gn):.3f} | scale {scaler.get_scale():.0f} | lr_bb {opt.param_groups[0]['lr']:.2e} lr_head {opt.param_groups[1]['lr']:.2e} | peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB | {time.time()-t0:.0f}s", flush=True)
    pv = predict(m, va_idx)
    if not np.isfinite(pv).all():
        print('NaN/inf in validation predictions - training diverged, stopping', flush=True); break
    s_val = macro_auc((Y[va_idx] > 0.5).astype(int), pv)
    s_gold = macro_auc(gold_y, predict(m, gi)) if len(gi) else float("nan")
    history.append(dict(epoch=ep, train_loss=round(run / max(bi + 1, 1), 4), val_vs_pseudo=round(s_val, 4), gold_auc=round(s_gold, 4), epoch_secs=round(time.time() - t0)))
    print(json.dumps(history[-1]), flush=True)
    top.append((s_val, ep, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()})); top.sort(key=lambda t: -t[0]); top = top[:3]
    if s_val > best:
        best = s_val; torch.save(m.state_dict(), f"/kaggle/working/ft_fold{FOLD}.pt"); np.save(f"/kaggle/working/val_pred_fold{FOLD}.npy", pv)
        if len(gi): np.save(f"/kaggle/working/gold_pred_fold{FOLD}.npy", predict(m, gi))
    if ep >= WARM_EPOCHS + 1 and s_val < best - 0.06:
        print(f"COLLAPSE GUARD: val {s_val:.4f} fell more than 0.06 below best {best:.4f} after the backbone started moving - stopping", flush=True); break
    if time.time() - t_start > 7.5 * 3600:
        print("time budget reached, stopping"); break
avg_info = {}
if len(top) >= 2:                                                  # weight average of the best (up to) 3 epochs by val: free at inference time
    sds = [t[2] for t in top]
    avg_sd = {k: (sum(sd[k].float() for sd in sds) / len(sds)).to(sds[0][k].dtype) if sds[0][k].is_floating_point() else sds[0][k] for k in sds[0]}
    m.load_state_dict(avg_sd); pa = predict(m, va_idx)
    a_val = macro_auc((Y[va_idx] > 0.5).astype(int), pa); ga = predict(m, gi) if len(gi) else None
    a_gold = macro_auc(gold_y, ga) if len(gi) else float("nan")
    avg_info = dict(epochs=[t[1] for t in top], val_vs_pseudo=round(a_val, 4), gold_auc=round(a_gold, 4), best_single=round(best, 4))
    print("WEIGHT AVERAGE of best epochs", json.dumps(avg_info), flush=True)
    torch.save(avg_sd, f"/kaggle/working/avg_fold{FOLD}.pt"); np.save(f"/kaggle/working/val_pred_avg_fold{FOLD}.npy", pa)
    if ga is not None: np.save(f"/kaggle/working/gold_pred_avg_fold{FOLD}.npy", ga)
    if a_val > best: torch.save(avg_sd, f"/kaggle/working/ft_fold{FOLD}.pt"); np.save(f"/kaggle/working/val_pred_fold{FOLD}.npy", pa); print("averaged weights are better -> saved as ft_fold", flush=True); best = a_val
json.dump(dict(fold=FOLD, history=history, best_val=best, avg=avg_info, val_uids=[uids[i] for i in va_idx]), open(f"/kaggle/working/ft_metrics_fold{FOLD}.json", "w"), indent=1)
print("done | best val vs pseudo", round(best, 4))
