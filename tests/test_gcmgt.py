"""GC-MGT tests: the Granger statistics against direct OLS, Eq. 5 properties, model mechanics."""

import numpy as np
import pytest
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gcmgt import granger as gr
from gcmgt import model as gm

N, T, B, S = 8, 12, 3, 12


def _var_data(rows=300, n=5, seed=0, coupling=0.0, src=0, dst=1):
    """Random series; optionally x_dst is driven by lagged x_src with weight `coupling`."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((rows, n)).astype(np.float64)
    X = np.cumsum(X, axis=0) * 0.1 + rng.standard_normal((rows, n))            # mildly non-stationary levels
    if coupling:
        for t in range(2, rows):
            X[t, dst] += coupling * (X[t - 1, src] - X[t - 2, src])            # acts on the DIFFERENCED series
    return X


def _direct_f(window, p):
    """Reference: per-pair two-regression OLS via lstsq on the differenced window."""
    D = np.diff(np.asarray(window, dtype=np.float64), axis=0)
    m, n_series = D.shape
    n = m - p
    Y = D[p:]
    lags = {i: np.column_stack([D[p - 1 - l: p - 1 - l + n, i] for l in range(p)]) for i in range(n_series)}
    ones = np.ones((n, 1))
    F = np.zeros((n_series, n_series))
    for j in range(n_series):
        R = np.column_stack([ones, lags[j]])
        rss_r = np.sum((Y[:, j] - R @ np.linalg.lstsq(R, Y[:, j], rcond=None)[0]) ** 2)
        for i in range(n_series):
            if i == j:
                continue
            U = np.column_stack([ones, lags[j], lags[i]])
            rss_u = np.sum((Y[:, j] - U @ np.linalg.lstsq(U, Y[:, j], rcond=None)[0]) ** 2)
            F[i, j] = ((rss_r - rss_u) / p) / (rss_u / (n - (2 * p + 1)))
    return F


def test_granger_f_matches_direct_ols():
    window = _var_data(rows=200, n=5, seed=3)
    p = 4
    F, pvals, dof2 = gr.granger_f_matrix(window, p)
    assert dof2 == (len(window) - 1 - p) - (2 * p + 1)
    assert np.allclose(F, _direct_f(window, p), rtol=1e-6, atol=1e-6)
    assert np.all(np.diag(F) == 0) and np.all(np.diag(pvals) == 1)
    off = ~np.eye(5, dtype=bool)
    assert np.all((pvals[off] > 0) & (pvals[off] <= 1))


def test_known_structure_is_detected():
    window = _var_data(rows=288, n=5, seed=7, coupling=0.9, src=0, dst=1)
    F, pvals, _ = gr.granger_f_matrix(window, p=4)
    assert F[0, 1] > 10 * F[1, 0]                                  # driven direction dominates
    sig = gr.bh_significant(pvals, alpha=0.05)
    assert sig[0, 1] and not sig[1, 0]


def test_bh_threshold():
    p = np.array([[1.0, 0.001, 0.2], [0.012, 1.0, 0.9], [0.04, 0.5, 1.0]])
    # sorted off-diagonal: 0.001, 0.012, 0.04, 0.2, 0.5, 0.9 vs alpha*k/6 = 0.0083, 0.0167, 0.025, ...
    sig = gr.bh_significant(p, alpha=0.05)
    assert sig[0, 1] and sig[1, 0] and not sig[2, 0] and sig.sum() == 2


def test_prior_shape_topk_and_row_normalisation():
    rng = np.random.default_rng(1)
    flow = rng.standard_normal((2000, N)).astype(np.float32)
    for t in range(2, 2000):                                       # a few real edges so BH keeps something
        flow[t, 1] += 0.8 * (flow[t - 1, 0] - flow[t - 2, 0])
        flow[t, 3] += 0.8 * (flow[t - 1, 2] - flow[t - 2, 2])
    prior = gr.granger_prior(flow, np.ones(2000, bool), w_g=288, p=4, k_edges=2, granger_windows=3, verbose=False)
    A = prior["A"]
    assert A.shape == (N, N) and np.all(A >= 0) and np.all(np.diag(A) == 0)
    assert np.all((A > 0).sum(axis=1) <= 2)                        # at most K outgoing edges per source
    for i in range(N):
        if (A[i] > 0).any():
            assert 0.9 < A[i].sum() <= 1.0                         # F / (sum F + eps), Eq. 5
    assert A[0, 1] > 0 and A[2, 3] > 0
    assert prior["window_starts"].shape == (3,)


def test_physical_graph_symmetric_no_diagonal():
    W = np.array([[0.5, 0.2, 0.0], [0.0, 1.0, 0.7], [0.1, 0.0, 0.3]])
    P = gr.physical_graph(W)
    assert np.array_equal(P, P.T) and np.all(np.diag(P) == 0)
    assert np.isclose(P[0, 2], 0.1) and np.isclose(P[1, 2], 0.7)   # P is float32


def _tiny_model(**kw):
    rng = np.random.default_rng(0)
    A = np.zeros((N, N), np.float32)
    A[rng.integers(0, N, 10), rng.integers(0, N, 10)] = 0.5
    np.fill_diagonal(A, 0)
    phy = gr.physical_graph(rng.random((N, N)).astype(np.float32))
    defaults = dict(in_feat=5, d=16, layers=2, heads=4, dff=32, ds=16, hs=4, df=8, dr=8, out_steps=S)
    defaults.update(kw)
    return gm.GCMGT(N, A, phy, **defaults)


def test_forward_shapes_and_aux():
    torch.manual_seed(0)
    model = _tiny_model()
    x, tf = torch.randn(B, T, N), torch.randn(B, T, 4)
    model.train()
    pred, aux = model(x, tf)
    assert pred.shape == (B, S, N) and aux.ndim == 0
    assert torch.isfinite(pred).all() and torch.isfinite(aux) and aux > 0


def test_fusion_weights_and_semantic_rows_are_normalised():
    torch.manual_seed(0)
    sem = gm.SemanticGraph(5, ds=16, hs=4, df=8)
    A_sem, alphas = sem(torch.randn(B, T, N, 5))
    assert torch.allclose(A_sem.sum(-1), torch.ones(B, N), atol=1e-5)          # softmax over target sensors
    assert torch.allclose(alphas.sum(-1), torch.ones(B), atol=1e-6)            # simplex (Eq. 11)
    assert (alphas >= 0).all()


def test_checkpointing_changes_nothing():
    torch.manual_seed(0)
    m1 = _tiny_model(dropout=0.0, checkpoint_attention=True)
    m2 = _tiny_model(dropout=0.0, checkpoint_attention=False)
    m2.load_state_dict(m1.state_dict())
    x, tf = torch.randn(B, T, N), torch.randn(B, T, 4)
    m1.train(), m2.train()
    x1 = x.clone().requires_grad_(True)
    x2 = x.clone().requires_grad_(True)
    p1, a1 = m1(x1, tf)
    p2, a2 = m2(x2, tf)
    assert torch.allclose(p1, p2, atol=1e-6) and torch.allclose(a1, a2, atol=1e-6)
    (p1.sum() + a1).backward()
    (p2.sum() + a2).backward()
    assert torch.allclose(x1.grad, x2.grad, atol=1e-5)


def test_train_step_runs_and_grads_flow():
    torch.manual_seed(0)
    model = _tiny_model()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x, tf = torch.randn(B, T, N), torch.randn(B, T, 4)
    y = torch.randn(B, S, N)
    mask = torch.ones(B, S, dtype=torch.bool)
    pred, aux = model(x, tf)
    loss = ((pred - y) ** 2).mean() + aux
    loss.backward()
    beta_grad = model.layers[0].attn.beta.grad
    assert beta_grad is not None and torch.isfinite(beta_grad).all()
    assert model.refiner.u.weight.grad is not None                 # the refinement path is trained (via aux + bias)
    opt.step()
    assert torch.isfinite(loss)


def test_eval_is_deterministic_and_prediction_is_direct():
    torch.manual_seed(0)
    model = _tiny_model(dropout=0.0)
    model.eval()
    x, tf = torch.randn(B, T, N), torch.randn(B, T, 4)
    with torch.no_grad():
        p1, _ = model(x, tf)
        p2, _ = model(x, tf)
    assert torch.equal(p1, p2)                                     # no autoregressive feedback, no stochastic path
