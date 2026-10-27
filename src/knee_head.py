"""Slice-attention head over frozen DINOv2 slice features (shared by training and inference)."""
import torch
import torch.nn as nn


class Head(nn.Module):
    def __init__(s, d_in=768, d=256, n_out=12, depth=32, drop=0.2, per_finding=False):
        super().__init__()
        s.per_finding = per_finding; s.n_out = n_out
        s.proj = nn.Sequential(nn.LayerNorm(d_in), nn.Dropout(drop), nn.Linear(d_in, d))
        s.pos = nn.Parameter(torch.zeros(1, 1, depth, d)); s.plane = nn.Parameter(torch.zeros(1, 3, 1, d))
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, d * 2, drop, batch_first=True, norm_first=True), 2)
        if per_finding:      # one attention over the slice groups per finding (a tear may show on 2 slices, OA on many), then a per-finding read-out
            s.attn = nn.Linear(d, n_out)
            s.fuse = nn.Sequential(nn.LayerNorm(3 * d), nn.Dropout(drop), nn.Linear(3 * d, d), nn.GELU())
            s.out_w = nn.Parameter(torch.randn(n_out, d) * d ** -0.5); s.out_b = nn.Parameter(torch.zeros(n_out))
        else:
            s.attn = nn.Linear(d, 1)
            s.fuse = nn.Sequential(nn.LayerNorm(3 * d), nn.Dropout(drop), nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, n_out))

    def forward(s, x, present):
        B, Pn, D, _ = x.shape
        h = s.proj(x.float()) + s.pos + s.plane
        h = s.enc(h.view(B * Pn, D, -1))
        if s.per_finding:
            w = torch.softmax(s.attn(h), 1)                                        # [B*Pn, D, K]
            v = torch.einsum("bdk,bde->bke", w, h).view(B, Pn, s.n_out, -1)        # [B, Pn, K, d]
            v = v * present.view(B, Pn, 1, 1)
            z = s.fuse(v.permute(0, 2, 1, 3).reshape(B * s.n_out, -1)).view(B, s.n_out, -1)   # the fuse MLP is shared across findings
            return (z * s.out_w).sum(-1) + s.out_b
        w = torch.softmax(s.attn(h).squeeze(-1), 1)
        v = (h * w.unsqueeze(-1)).sum(1).view(B, Pn, -1) * present.unsqueeze(-1)
        return s.fuse(v.reshape(B, -1))
