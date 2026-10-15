"""
parpi_rinalmo_giga_esmc head (trainable part only; both encoders frozen + cached).

    RNA repr  (B, L, 1280) --proj--> (B, L, H) --+
                                                 +--> Top-k cross attention (shared, DeepMutDTA)
    Prot repr (U, P, 1152) --proj--> (U, P, H) --+     RNA ctx  : Q = protein, K/V = RNA
                                                       Prot ctx : Q = RNA,     K/V = protein
                -> attention pooling (masked) -> concat (2H) -> 2-layer MLP -> raw logit

The top-k attention and attention pooling follow DeepMutDTA's ExactTopKAttention /
AttentionPooling. Two correctness additions for variable-length inputs:
  * padded key positions are masked out before the top-k selection;
  * attention pooling ignores padded positions (masked, numerically stable softmax).
Proteins are projected once per unique protein in the batch, then gathered.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExactTopKAttention(nn.Module):
    def __init__(self, hidden_size, topk=32, n_head=4, attention_dropout=0.1):
        super().__init__()
        assert hidden_size % n_head == 0, "hidden_size must be divisible by num_heads"
        self.topk = topk
        self.n_heads = n_head
        self.hidden_size = hidden_size
        self.f_q = nn.Linear(hidden_size, hidden_size)
        self.f_k = nn.Linear(hidden_size, hidden_size)
        self.f_v = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, key_mask=None):
        N, L, _ = queries.shape
        S = keys.shape[1]
        H, E = self.n_heads, self.hidden_size // self.n_heads
        q = self.f_q(queries).view(N, L, H, E)
        k = self.f_k(keys).view(N, S, H, E)
        v = self.f_v(values).view(N, S, H, E)

        qk = torch.einsum("nlhe,nshe->nhls", q, k).float()
        if key_mask is not None:
            qk = qk.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        kk = min(self.topk, S)
        kth = torch.topk(qk, kk, dim=-1, sorted=True).values[..., -1:]
        qk = qk.masked_fill(qk < kth, float("-inf"))           # keep top-k keys only
        attn = torch.softmax(qk / math.sqrt(E), dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn).to(v.dtype)
        out = torch.einsum("nhls,nshd->nlhd", attn, v).reshape(N, L, H * E)
        return out.contiguous(), attn


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.att_fc1 = nn.Linear(hidden_size, hidden_size)
        self.att_fc2 = nn.Linear(hidden_size, 1)
        for m in (self.att_fc1, self.att_fc2):
            m.weight.data.normal_(mean=0.0, std=0.2)
            m.bias.data.zero_()

    def forward(self, x, mask=None):
        a = self.att_fc2(torch.tanh(self.att_fc1(x))).squeeze(-1).float()   # (B, T)
        if mask is not None:
            a = a.masked_fill(~mask, float("-inf"))
        alpha = torch.nan_to_num(torch.softmax(a, dim=1), nan=0.0)
        pooled = torch.bmm(alpha.unsqueeze(1).to(x.dtype), x).squeeze(1)    # (B, D)
        return pooled, alpha


class Projector(nn.Module):
    """Trainable projection of frozen encoder features to hidden_size."""

    def __init__(self, in_dim, hidden_size, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden_size),
                                 nn.GELU(), nn.Dropout(dropout))

    def forward(self, x, mask):
        return self.net(x) * mask.unsqueeze(-1).to(x.dtype)


class RNAProteinTopK(nn.Module):
    def __init__(self, rna_dim=1280, prot_dim=1152, hidden_size=256, num_heads=4, topk=32,
                 dropout=0.1, l2norm="seq"):
        super().__init__()
        self.l2norm = l2norm
        self.rna_proj = Projector(rna_dim, hidden_size, dropout)
        self.prot_proj = Projector(prot_dim, hidden_size, dropout)
        self.fusion = ExactTopKAttention(hidden_size, topk=topk, n_head=num_heads,
                                         attention_dropout=dropout)
        self.pool = AttentionPooling(hidden_size)
        self.mlp = nn.Sequential(                       # 2-layer MLP -> raw logit
            nn.Linear(2 * hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, 1))

    def _norm(self, x):
        if self.l2norm == "seq":      # DeepMutDTA: F.normalize(..., dim=1)
            return F.normalize(x, p=2, dim=1)
        if self.l2norm == "feature":
            return F.normalize(x, p=2, dim=-1)
        return x

    def forward(self, rna, rna_mask, prot_u, prot_mask_u, prot_inv):
        """
        rna       : (B, L, Dr)  cached RiNALMo features (padded)
        rna_mask  : (B, L) bool
        prot_u    : (U, P, Dp)  cached ESMC features of the U unique proteins in batch
        prot_mask_u : (U, P) bool
        prot_inv  : (B,) long   index of each sample's protein in prot_u
        returns raw logits (B,)
        """
        r = self._norm(self.rna_proj(rna, rna_mask))
        p_u = self._norm(self.prot_proj(prot_u, prot_mask_u))
        p = p_u[prot_inv]
        p_mask = prot_mask_u[prot_inv]

        rna_ctx, _ = self.fusion(p, r, r, key_mask=rna_mask)     # protein queries RNA
        rna_vec, _ = self.pool(rna_ctx, p_mask)
        prot_ctx, _ = self.fusion(r, p, p, key_mask=p_mask)      # RNA queries protein
        prot_vec, _ = self.pool(prot_ctx, rna_mask)

        emb = F.normalize(torch.cat((rna_vec, prot_vec), dim=1), p=2, dim=1)
        return self.mlp(emb).squeeze(-1)
