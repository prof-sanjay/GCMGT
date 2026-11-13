"""
GC-MGT (Wang, Riccardi, Fiorentini & Losa, "A Granger causality-guided multi-graph transformer framework for
traffic flow forecasting", Discover Artificial Intelligence 6:1219, 2026) - RE-IMPLEMENTATION FROM THE PAPER TEXT.
No official repository exists; results must be reported as "re-implementation of GC-MGT", never as the
paper's published numbers.

Taken from the paper (section / equation):
  Granger prior        pairwise F-tests, Eq. 2-4: unrestricted = intercept + p own lags + p source lags; restricted
                       = intercept + p own lags; F = ((RSS_R - RSS_U)/p) / (RSS_U/(n_eff - k)), k = 2p + 1,
                       n_eff = w_G - d - p with first differencing d = 1. Estimation window w_G = 288 (one day),
                       lag p = 12, Benjamini-Hochberg FDR alpha = 0.05 within each estimation window (section 3.3.1.1).
  A_Granger            Eq. 5: per source row i keep at most K = 15 BH-significant targets with the largest F;
                       weights F_ij / (sum of retained F + eps). Rows = source i, columns = target j. The prior is
                       FIXED, estimated offline from the training data (section 3.3.1.2).
  Neural refinement    Eq. 6: A_causal(t) = A_Granger (.) sigmoid(f_theta(X_{t-T+1:t})), f_theta a "lightweight
                       encoder combining temporal convolution and cross-node attention" producing an N x N matrix.
  Graph regularisation Eq. 7: lambda1 ||A_causal||_1 + lambda2 ||A_causal(t) - A_causal(t-1)||_F^2,
                       lambda1 = 0.01, lambda2 = 0.1.
  Physical graph       Eq. 8: distance-decayed weights exp(-d^2/sigma^2) on connected / nearby sensor pairs.
  Semantic graph       Eq. 9-10: temporal mean pooling, multi-head attention over nodes (d_s = 128, h_s = 8 heads),
                       A_sem = average over heads of softmax(E_r E_r^T / sqrt(d_s/h_s)) over target sensors.
  Adaptive fusion      Eq. 11-12: [a_phy, a_sem, a_causal] = softmax(MLP(mean-pooled node embeddings)), MLP hidden
                       d_f = 64; A_fused = a_phy A_phy + a_sem A_sem + a_causal A_causal.
  Transformer          section 3.3.3: input projection + temporal positional encoding; per layer a multi-scale
                       Conv1D branch (kernels {1,3,5,7}, stride 1, same padding, concat + linear, Eq. 13-14) and a
                       per-time-step multi-head attention over the N sensors (d = 64, h = 8 heads) with the graph
                       bias beta_r * normalize(A_fused)^T added to the logits (Eq. 17, normalize = D^-1 (A + I));
                       sigmoid gate fusing the two branches (Eq. 18); 4 encoder layers, feed-forward d_ff = 256,
                       dropout 0.1 on the conv and attention branches (section 3.3.3.6).
  Decoder              Eq. 19: linear d -> H * F_out from the LAST observed time step; direct (non-autoregressive).
  Loss                 Eq. 20-21: MSE + Eq. 7 + lambda_smooth * mean ||Y_hat_{tau+1} - Y_hat_tau||^2, lambda_smooth
                       = 0.01.

Deviations (every one is deliberate and listed here; also written to config.json by the runner):
  D1 inputs    the paper uses speed/flow/occupancy; LargeST-GBA gives flow only, so the input features are the
               per-sensor standardised flow + 4 time encodings (F = 5), and F_out = 1.
  D2 A_phy     the LargeST road adjacency already stores distance-decayed weights in (0, 1] (the exact form of
               Eq. 8); it is symmetrised with max(W, W^T) and the diagonal is zeroed (Eq. 17 re-adds I).
  D3 prior     the paper fixes A_Granger offline but does not say how per-window tests are combined. We test
               `granger_windows` (default 8) day-long windows evenly spaced over the training split (restricted to
               fully input-ok rows), BH within each window as in the paper, then keep an edge when it is significant
               in >= half of the windows and score it by its median F. Top-K and row normalisation follow Eq. 5.
  D4 d = 1     the paper first-differences only ADF-non-stationary series; we difference ALL series uniformly
               (traffic flow in day windows is treated as difference-stationary) - mixing levels and differences
               across pairs is avoided and n_eff is the same for every pair.
  D5 lambda2   "consecutive input windows" are not available from the window loader (it yields exactly T rows), so
               A_causal(t-1) is approximated by the refinement of the same window WITHOUT its last step (the two
               windows end one step apart). Applied in training mode only.
  D6 norms     Eq. 7 sums over N^2 entries; at N = 2,056 (vs the paper's 170-325) the paper's lambdas would dwarf
               the MSE, so both norms are divided by N (per-source-row average). The prediction-smoothness term is
               a per-element mean. lambda1/lambda2/lambda_smooth keep the paper's values.
  D7 f_theta   dimensions unspecified in the paper; ours: Conv1d(F -> 32, kernel 3) per node, mean over time,
               one 4-head cross-node attention (residual), bilinear edge scorer (U e_i) . (V e_j) / sqrt(32).
  D8 layers    residual connections + LayerNorm + feed-forward per encoder layer are assumed from "follows the
               standard Transformer architecture" (section 3.3.3); positional encoding is the standard sinusoid.
  D9 memory    at N = 2,056 each layer's per-step N x N attention needs ~1.6 GB of activations per sample, so the
               spatial attention runs one time step at a time with gradient checkpointing (identical math, ~2x
               compute) and the default batch size is small (2). The loss is the paper's MSE, computed on standardised
               flow over the observed targets of the 12 output steps.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig

# ----------------------------------------------------------------------------------------------- model modules
class MultiScaleConv(nn.Module):
    """Eq. 13-14: parallel Conv1D over time per node, kernels {1,3,5,7}, concat + linear back to d."""

    def __init__(self, d: int, kernels=(1, 3, 5, 7), dropout: float = 0.1):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(d, d, k, padding=k // 2) for k in kernels])
        self.proj = nn.Linear(d * len(kernels), d)
        self.drop = nn.Dropout(dropout)

    def forward(self, E: torch.Tensor) -> torch.Tensor:           # [B, T, N, d]
        B, T, N, d = E.shape
        z = E.permute(0, 2, 3, 1).reshape(B * N, d, T)
        out = torch.cat([conv(z) for conv in self.convs], dim=1)   # [B*N, 4d, T]
        out = out.reshape(B, N, -1, T).permute(0, 3, 1, 2)         # [B, T, N, 4d]
        return self.drop(self.proj(out))


class GraphGuidedAttention(nn.Module):
    """
    Eq. 15-17: per-time-step multi-head attention over the N sensors with the logit bias beta_r * A_norm^T.
    One time step at a time, checkpointed in training (deviation D9; identical math).
    """

    def __init__(self, d: int, heads: int, dropout: float = 0.1, checkpoint_attention: bool = True):
        super().__init__()
        if d % heads:
            raise ValueError("d must be divisible by heads")
        self.h, self.dh = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.beta = nn.Parameter(torch.ones(heads))                # learnable bias scale per head (Eq. 17)
        self.attn_dropout = nn.Dropout(dropout)
        self.drop = nn.Dropout(dropout)
        self.checkpoint_attention = checkpoint_attention

    def _step(self, e: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:          # e [B, N, d], bias [B, N, N]
        B, N, d = e.shape
        q, k, v = self.qkv(e).reshape(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.dh)      # [B, h, N, N]
        logits = logits + self.beta.view(1, -1, 1, 1) * bias.unsqueeze(1)
        att = self.attn_dropout(torch.softmax(logits, dim=-1))
        o = (att @ v).transpose(1, 2).reshape(B, N, d)
        return self.out(o)

    def forward(self, E: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:        # E [B, T, N, d]
        use_ckpt = self.checkpoint_attention and self.training and torch.is_grad_enabled()
        outs = []
        for t in range(E.shape[1]):
            e = E[:, t]
            outs.append(checkpoint(self._step, e, bias, use_reentrant=False) if use_ckpt else self._step(e, bias))
        return self.drop(torch.stack(outs, dim=1))


class EncoderLayer(nn.Module):
    """One encoder layer: conv + graph-guided attention branches, sigmoid gate (Eq. 18), then LN/FFN (D8)."""

    def __init__(self, d: int, heads: int, dff: int, dropout: float, checkpoint_attention: bool):
        super().__init__()
        self.conv = MultiScaleConv(d, dropout=dropout)
        self.attn = GraphGuidedAttention(d, heads, dropout, checkpoint_attention)
        self.gate = nn.Linear(2 * d, d)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, dff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dff, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, E: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        Hc = self.conv(E)
        Ha = self.attn(E, bias)
        G = torch.sigmoid(self.gate(torch.cat([Hc, Ha], dim=-1)))
        Hf = G * Ha + (1.0 - G) * Hc                               # Eq. 18
        E = self.norm1(E + self.drop(Hf))
        return self.norm2(E + self.drop(self.ffn(E)))


class SemanticGraph(nn.Module):
    """Eq. 9-11: mean-pooled window, node-wise MHA (h_s heads), A_sem over heads, fusion weights from pooled Z."""

    def __init__(self, in_feat: int, ds: int = 128, hs: int = 8, df: int = 64):
        super().__init__()
        if ds % hs:
            raise ValueError("ds must be divisible by hs")
        self.hs, self.dsh = hs, ds // hs
        self.wq = nn.Linear(in_feat, ds)
        self.wk = nn.Linear(in_feat, ds)
        self.wv = nn.Linear(in_feat, ds)
        self.fusion = nn.Sequential(nn.Linear(ds, df), nn.ReLU(), nn.Linear(df, 3))

    def forward(self, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:   # feats [B, T, N, F]
        xbar = feats.mean(dim=1)                                   # temporal mean pooling [B, N, F]
        B, N, _ = xbar.shape
        shape = (B, N, self.hs, self.dsh)
        q = self.wq(xbar).reshape(shape).permute(0, 2, 1, 3)       # [B, hs, N, dsh]
        k = self.wk(xbar).reshape(shape).permute(0, 2, 1, 3)
        v = self.wv(xbar).reshape(shape).permute(0, 2, 1, 3)
        A_sem = feats.new_zeros(B, N, N)
        Z = []
        for r in range(self.hs):                                   # head loop caps the [B, N, N] transients
            E_r = torch.softmax(q[:, r] @ k[:, r].transpose(-1, -2) / math.sqrt(self.dsh), dim=-1) @ v[:, r]
            A_sem = A_sem + torch.softmax(E_r @ E_r.transpose(-1, -2) / math.sqrt(self.dsh), dim=-1)
            Z.append(E_r)
        A_sem = A_sem / self.hs                                    # Eq. 10
        Z = torch.cat(Z, dim=-1)                                   # [B, N, ds]
        alphas = torch.softmax(self.fusion(Z.mean(dim=1)), dim=-1) # Eq. 11, [B, 3]
        return A_sem, alphas


class GrangerRefiner(nn.Module):
    """f_theta of Eq. 6 (dimensions are ours, deviation D7): Conv1d + cross-node attention + bilinear scorer."""

    def __init__(self, in_feat: int, dr: int = 32, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dr = dr
        self.conv = nn.Conv1d(in_feat, dr, 3, padding=1)
        self.attn = nn.MultiheadAttention(dr, heads, dropout=dropout, batch_first=True)
        self.u = nn.Linear(dr, dr)
        self.v = nn.Linear(dr, dr)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:        # [B, T, N, F] -> edge logits [B, N, N]
        B, T, N, Fin = feats.shape
        z = feats.permute(0, 2, 3, 1).reshape(B * N, Fin, T)
        e = torch.relu(self.conv(z)).mean(dim=-1).reshape(B, N, self.dr)
        e = e + self.attn(e, e, e, need_weights=False)[0]
        return (self.u(e) @ self.v(e).transpose(1, 2)) / math.sqrt(self.dr)


class GCMGT(nn.Module):
    """
    The full model. forward(x [B, T, N], tfeat [B, T, 4]) -> (pred [B, out_steps, N], aux-loss scalar); the aux
    loss carries Eq. 7 (graph sparsity + temporal smoothness, training mode only for the latter) and Eq. 20
    (prediction smoothness), so the trainer adds it to the masked prediction loss (Eq. 21).
    """

    def __init__(self, num_nodes: int, a_granger: np.ndarray, a_phy: np.ndarray, *, in_feat: int = 5, d: int = 64,
                 layers: int = 4, heads: int = 8, dff: int = 256, ds: int = 128, hs: int = 8, df: int = 64,
                 dr: int = 32, out_steps: int = 12, dropout: float = 0.1, lambda1: float = 0.01,
                 lambda2: float = 0.1, lambda_smooth: float = 0.01, checkpoint_attention: bool = True,
                 max_len: int = 64):
        super().__init__()
        self.num_nodes, self.out_steps = num_nodes, out_steps
        self.lambda1, self.lambda2, self.lambda_smooth = lambda1, lambda2, lambda_smooth
        self.register_buffer("A_granger", torch.as_tensor(np.asarray(a_granger, dtype=np.float32)))
        self.register_buffer("A_phy", torch.as_tensor(np.asarray(a_phy, dtype=np.float32)))
        pos = torch.arange(max_len).unsqueeze(1) * torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe = torch.zeros(max_len, d)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos), torch.cos(pos)
        self.register_buffer("pe", pe)                             # sinusoidal positional encoding (D8)
        self.in_proj = nn.Linear(in_feat, d)
        self.refiner = GrangerRefiner(in_feat, dr=dr, dropout=dropout)
        self.semantic = SemanticGraph(in_feat, ds=ds, hs=hs, df=df)
        self.layers = nn.ModuleList([EncoderLayer(d, heads, dff, dropout, checkpoint_attention)
                                     for _ in range(layers)])
        self.decoder = nn.Linear(d, out_steps)                     # Eq. 19, F_out = 1

    def _causal(self, feats: torch.Tensor) -> torch.Tensor:
        return self.A_granger.unsqueeze(0) * torch.sigmoid(self.refiner(feats))    # Eq. 6

    def forward(self, x: torch.Tensor, tfeat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, N = x.shape
        feats = torch.cat([x.unsqueeze(-1), tfeat.unsqueeze(2).expand(-1, -1, N, -1)], dim=-1)
        A_causal = self._causal(feats)
        A_sem, alphas = self.semantic(feats)
        A_fused = (alphas[:, 0, None, None] * self.A_phy + alphas[:, 1, None, None] * A_sem
                   + alphas[:, 2, None, None] * A_causal)          # Eq. 12
        A_hat = A_fused + torch.eye(N, device=x.device, dtype=A_fused.dtype)
        A_norm = A_hat / A_hat.sum(dim=-1, keepdim=True).clamp_min(1e-8)           # D^-1 (A + I), Eq. 17
        bias = A_norm.transpose(1, 2)                              # transposed: row = target, column = source
        E = self.in_proj(feats) + self.pe[:T].view(1, T, 1, -1)
        for layer in self.layers:
            E = layer(E, bias)
        pred = self.decoder(E[:, -1]).permute(0, 2, 1)             # last observed step -> [B, out_steps, N]
        aux = self.lambda1 * A_causal.abs().sum(dim=(1, 2)).mean() / N             # Eq. 7, L1 (deviation D6)
        if self.training and self.lambda2 > 0:
            A_prev = self._causal(feats[:, :-1])                   # window one step earlier (deviation D5)
            aux = aux + self.lambda2 * ((A_causal - A_prev) ** 2).sum(dim=(1, 2)).mean() / N
        if self.lambda_smooth > 0 and self.out_steps > 1:
            aux = aux + self.lambda_smooth * ((pred[:, 1:] - pred[:, :-1]) ** 2).mean()        # Eq. 20
        return pred, aux


def build_model(num_nodes: int, a_granger: np.ndarray, a_phy: np.ndarray, cfg: ModelConfig, out_steps: int = 12) -> GCMGT:
    return GCMGT(num_nodes, a_granger, a_phy, in_feat=5, d=cfg.hidden, layers=cfg.layers, heads=cfg.heads, dff=cfg.ffn,
                 ds=cfg.sem_dim, hs=cfg.sem_heads, df=cfg.fusion_hidden, dr=cfg.refine_dim, out_steps=out_steps,
                 dropout=cfg.dropout, lambda1=cfg.lambda_sparse, lambda2=cfg.lambda_smooth_graph,
                 lambda_smooth=cfg.lambda_smooth_pred, checkpoint_attention=cfg.checkpoint_attention)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
