"""Pointer-style actor-critic over a variable set of candidate targets (+ a learned STOP token).

Candidates are embedded by an MLP, the global row is added to every token, a small transformer encoder lets
candidates attend to each other (which object is already covered, which frontier is closer than another), and each
token gets a logit. Value = MLP on the mean-pooled tokens + global row.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .features import N_CAND_FEAT, N_GLOBAL_FEAT


class InspectionPolicy(nn.Module):
    def __init__(self, d: int = 128, heads: int = 4, layers: int = 2):
        super().__init__()
        self.cand = nn.Sequential(nn.Linear(N_CAND_FEAT, d), nn.ReLU(), nn.Linear(d, d))
        self.glob = nn.Sequential(nn.Linear(N_GLOBAL_FEAT, d), nn.ReLU(), nn.Linear(d, d))
        self.stop = nn.Parameter(torch.zeros(1, 1, d))
        # STOP starts unlikely (a random policy would otherwise end most episodes after a few targets)
        self.stop_bias = nn.Parameter(torch.tensor(-3.0))
        enc = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.logit = nn.Linear(d, 1)
        self.value = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, cand, mask, glob):
        """cand [B, N, F], mask [B, N] (True = real), glob [B, G] -> logits [B, N + 1] (last = STOP), value [B]."""
        B = cand.shape[0]
        g = self.glob(glob)[:, None]
        x = torch.cat([self.cand(cand) + g, self.stop.expand(B, 1, -1) + g], 1)
        m = torch.cat([mask, torch.ones(B, 1, dtype=torch.bool, device=mask.device)], 1)
        h = self.enc(x, src_key_padding_mask=~m)
        logits = self.logit(h).squeeze(-1)
        logits = torch.cat([logits[:, :-1], logits[:, -1:] + self.stop_bias], 1).masked_fill(~m, -1e9)
        pooled = (h * m[..., None]).sum(1) / m.sum(1, keepdim=True)
        v = self.value(torch.cat([pooled, g[:, 0]], -1)).squeeze(-1)
        return logits, v
