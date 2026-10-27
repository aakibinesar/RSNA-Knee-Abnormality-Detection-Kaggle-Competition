# ---------------- torchvision backbone factory shared by the training and inference scripts ----------------
# Supports resnet*, convnext_*, efficientnet_* / efficientnet_v2_*; weights are the unmodified torchvision .pth files from Kaggle model sources.
import torch
import torch.nn as nn


def strip_head(bb):
    """Replace the classifier with an identity so the backbone returns pooled features; returns the feature dimension."""
    if hasattr(bb, "fc"):                                         # resnet
        d = bb.fc.in_features; bb.fc = nn.Identity()
    elif hasattr(bb, "classifier") and isinstance(bb.classifier[-1], nn.Linear) and isinstance(bb.classifier[0], nn.Dropout):   # efficientnet
        d = bb.classifier[-1].in_features; bb.classifier = nn.Identity()
    elif hasattr(bb, "classifier"):                               # convnext: [LayerNorm2d, Flatten, Linear] -> keep norm + flatten
        d = bb.classifier[-1].in_features; bb.classifier[-1] = nn.Identity()
    else:
        raise ValueError("unsupported architecture")
    return d


def find_weights(arch):
    ck = None
    for r, dirs, fs in os.walk("/kaggle/input"):                  # pruned: never descend into the 820k-file image tree or the npz shards
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series") and not d.startswith("shard")]
        hit = [f for f in fs if f.startswith(arch) and f.endswith(".pth") and f[len(arch)] in "-_"]
        if hit: ck = os.path.join(r, hit[0]); break
    assert ck, f"no {arch}*.pth weights found"
    return ck


def build_backbone(arch, pretrained):
    import torchvision
    bb = torchvision.models.get_model(arch, weights=None)
    if pretrained: bb.load_state_dict(torch.load(find_weights(arch), map_location="cpu", weights_only=True))
    return bb, strip_head(bb)
