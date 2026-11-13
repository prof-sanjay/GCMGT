"""
The fixed Granger prior A_Granger of GC-MGT (paper section 3.3.1, Eq. 2-5) and the physical graph (Eq. 8).
Deviations D2 (physical graph), D3 (window aggregation) and D4 (uniform differencing) are documented in gcmgt/model.py.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .config import GrangerConfig

# ----------------------------------------------------------------------------------------------- Granger prior

def granger_f_matrix(window: np.ndarray, p: int = 12) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Pairwise Granger F-statistics on ONE estimation window (Eq. 2-4).

    window : [w_G, N] (levels; differenced here, deviation D4). Returns (F, pvalues, dof2) with F[i, j] the
    statistic for direction i -> j (diagonal 0, p-value 1). Float64 throughout: F is a difference of residual
    sums of squares and cancels catastrophically in float32.
    """
    from scipy import stats

    X = torch.as_tensor(np.asarray(window, dtype=np.float64))
    D = X[1:] - X[:-1]                                             # d = 1 differencing (deviation D4)
    m, N = D.shape
    n = m - p
    k = 2 * p + 1
    dof2 = n - k
    if dof2 <= 0:
        raise ValueError(f"window too short: n_eff - k = {dof2} <= 0 (w_g={len(window)}, p={p})")
    Y = D[p:]                                                      # [n, N]
    L = torch.stack([D[p - 1 - l: p - 1 - l + n] for l in range(p)], dim=2)        # [n, N, p], lag l+1
    Lf = L.reshape(n, N * p)
    ones = torch.ones(n, 1, dtype=torch.float64)
    Fmat = torch.zeros(N, N, dtype=torch.float64)
    eye = torch.eye(p, dtype=torch.float64)
    for j in range(N):
        Wj = torch.cat([ones, L[:, j, :]], dim=1)                  # restricted design: intercept + own lags
        Q, _ = torch.linalg.qr(Wj)                                 # orthonormal basis; projector QQ^T
        r = Y[:, j] - Q @ (Q.T @ Y[:, j])                          # restricted residuals
        rss_r = float(r @ r)
        Lres = Lf - Q @ (Q.T @ Lf)                                 # all source-lag blocks, own lags partialled out
        Lres = Lres.reshape(n, N, p)
        G = torch.einsum("nip,niq->ipq", Lres, Lres)               # [N, p, p]
        c = torch.einsum("nip,n->ip", Lres, r)                     # [N, p]
        ridge = 1e-10 * G.diagonal(dim1=1, dim2=2).mean().clamp_min(1e-30)
        sol = torch.linalg.solve(G + ridge * eye, c)
        explained = (c * sol).sum(dim=1).clamp(min=0.0, max=rss_r) # RSS_R - RSS_U >= 0
        rss_u = (rss_r - explained).clamp_min(1e-30)
        Fmat[:, j] = (explained / p) / (rss_u / dof2)
    Fmat.fill_diagonal_(0.0)
    Fnp = Fmat.numpy()
    pvals = stats.f.sf(Fnp, p, dof2)
    np.fill_diagonal(pvals, 1.0)
    return Fnp, pvals, dof2


def bh_significant(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg over the off-diagonal p-values of one window; returns a boolean [N, N] mask."""
    N = pvals.shape[0]
    off = ~np.eye(N, dtype=bool)
    flat = pvals[off]
    order = np.argsort(flat)
    m = flat.size
    below = flat[order] <= alpha * (np.arange(1, m + 1) / m)
    sig = np.zeros(m, dtype=bool)
    if below.any():
        cutoff = np.max(np.flatnonzero(below))
        sig[order[: cutoff + 1]] = True
    out = np.zeros((N, N), dtype=bool)
    out[off] = sig
    return out


def granger_prior(flow: np.ndarray, input_ok: np.ndarray, *, w_g: int = 288, p: int = 12, alpha: float = 0.05,
                  k_edges: int = 15, granger_windows: int = 8, eps: float = 1e-8, verbose: bool = True) -> dict:
    """
    The fixed A_Granger prior (Eq. 5) from the training split, aggregated over several estimation windows
    (deviation D3). Returns {"A", "F_med", "sig_frac", "window_starts"}; A[i, j] = weight of edge i -> j.
    """
    flow = np.asarray(flow)
    ok = np.asarray(input_ok, dtype=bool)
    rows = len(ok)
    bad = np.concatenate(([0], np.cumsum(~ok)))
    candidates = np.flatnonzero(bad[w_g:] - bad[:-w_g] == 0)       # starts with w_g consecutive input-ok rows
    if candidates.size == 0:
        raise ValueError(f"no run of {w_g} consecutive input-ok rows in {rows} training rows")
    take = np.unique(np.linspace(0, candidates.size - 1, min(granger_windows, candidates.size)).round().astype(int))
    starts = candidates[take]
    N = flow.shape[1]
    F_all = np.zeros((len(starts), N, N), dtype=np.float64)
    sig_all = np.zeros((len(starts), N, N), dtype=bool)
    for w, s in enumerate(starts):
        t0 = time.time()
        Fmat, pvals, _ = granger_f_matrix(flow[s: s + w_g], p)
        F_all[w] = Fmat
        sig_all[w] = bh_significant(pvals, alpha)
        if verbose:
            print(f"  granger window {w + 1}/{len(starts)} (start row {s}): "
                  f"{int(sig_all[w].sum()):,} BH-significant edges  {time.time() - t0:.0f}s", flush=True)
    sig_frac = sig_all.mean(axis=0)
    F_med = np.median(F_all, axis=0)
    keep = sig_frac >= 0.5                                         # majority vote (deviation D3)
    A = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        targets = np.flatnonzero(keep[i])
        if targets.size == 0:
            continue
        if targets.size > k_edges:
            targets = targets[np.argsort(F_med[i, targets])[::-1][:k_edges]]
        w = F_med[i, targets]
        A[i, targets] = (w / (w.sum() + eps)).astype(np.float32)   # Eq. 5
    return {"A": A, "F_med": F_med.astype(np.float32), "sig_frac": sig_frac.astype(np.float32),
            "window_starts": starts.astype(np.int64)}


def physical_graph(road_adjacency: np.ndarray) -> np.ndarray:
    """A_phy (Eq. 8): the LargeST weights are already distance-decayed (deviation D2); max-symmetrised, diag 0."""
    W = np.asarray(road_adjacency, dtype=np.float32)
    W = np.maximum(W, W.T)
    np.fill_diagonal(W, 0.0)
    return W


def load_or_build_prior(data, cfg: GrangerConfig, tag: str = "") -> dict:
    """The prior for the training split of `data`, cached in cfg.out_dir (it takes minutes at 2,056 sensors)."""
    params = dict(window=cfg.window, lag=cfg.lag, alpha=cfg.alpha, top_k=cfg.top_k, windows=cfg.windows,
                  train_rows=int(len(data.train.inputs)), tag=tag)
    digest = hashlib.sha256(json.dumps(params, sort_keys=True).encode() + np.asarray(data.sensors).tobytes()).hexdigest()[:16]
    path = Path(cfg.out_dir) / f"granger_{digest}.npz"
    if path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in ("A", "F_med", "sig_frac", "window_starts")} | {"cache": str(path)}
    print(f"estimating the Granger prior ({cfg.windows} windows of {cfg.window} rows, p={cfg.lag}, BH alpha={cfg.alpha}, "
          f"K={cfg.top_k}); cached at {path}", flush=True)
    prior = granger_prior(data.train.inputs, data.train.input_ok, w_g=cfg.window, p=cfg.lag, alpha=cfg.alpha,
                          k_edges=cfg.top_k, granger_windows=cfg.windows)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **prior, params=json.dumps(params))
    return prior | {"cache": str(path)}
