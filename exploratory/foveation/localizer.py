"""
Foveation Phase 1b.2: the learned query-conditioned localizer head.

Input  : frozen survey visual grid  vis (B, N, d_in)  [N = 16*16 cells]  + query emb q (B, d_in).
Output : per-cell relevance logits (B, N) -> heatmap; argmax/top-k cell -> region to foveate.
Only this head trains (the base VLM is frozen). A query token is prepended to the visual grid and
a small transformer contextualizes query<->cells; 2D positional embeddings give spatial structure
(needed for relative-position queries).
"""
import torch
import torch.nn as nn


class Localizer(nn.Module):
    def __init__(self, d_in, grid=16, h=512, layers=3, heads=8, p=0.1):
        super().__init__()
        self.grid = grid
        self.vis_proj = nn.Linear(d_in, h)
        self.q_proj = nn.Linear(d_in, h)
        self.pos = nn.Parameter(torch.randn(1, grid * grid, h) * 0.02)   # learned 2D pos (row-major)
        enc = nn.TransformerEncoderLayer(d_model=h, nhead=heads, dim_feedforward=h * 4,
                                         dropout=p, batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)
        self.norm = nn.LayerNorm(h)
        self.head = nn.Linear(h, 1)

    def forward(self, vis, q):                       # vis (B,N,d_in), q (B,d_in)
        v = self.vis_proj(vis) + self.pos            # (B,N,h)
        qt = self.q_proj(q).unsqueeze(1)             # (B,1,h)
        x = torch.cat([qt, v], dim=1)                # (B,N+1,h)  query token first
        x = self.enc(x)
        cells = self.norm(x[:, 1:])                  # (B,N,h)
        return self.head(cells).squeeze(-1)          # (B,N) per-cell logits


def topk_region(logits, grid=16, k=1, pad=0.35, minfrac=0.12):
    """Map per-cell logits -> a generous foveation box in FRACTIONAL coords (x0,y0,x1,y1) in [0,1].
    Takes the top-k cells, their enclosing box, then expands (coarse localization suffices)."""
    idx = torch.topk(logits, k).indices.tolist()
    rows = [i // grid for i in idx]; cols = [i % grid for i in idx]
    x0, x1 = min(cols) / grid, (max(cols) + 1) / grid
    y0, y1 = min(rows) / grid, (max(rows) + 1) / grid
    cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)
    w = max(w * (1 + 2 * pad), minfrac); h = max(h * (1 + 2 * pad), minfrac)
    return (max(0, cx - w / 2), max(0, cy - h / 2), min(1, cx + w / 2), min(1, cy + h / 2))
