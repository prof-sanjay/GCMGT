"""End-to-end check of the scripts on a small synthetic dataset: prepared file -> Granger prior -> training -> outputs."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def make_dataset(path: Path, n=10, days=14, seed=0):
    rng = np.random.default_rng(seed)
    T = days * 288
    road = np.zeros((n, n), dtype=np.float32)
    for i in range(n - 1):
        road[i, i + 1] = road[i + 1, i] = 0.8
    daily = 50 + 40 * np.sin(2 * np.pi * np.arange(T) / 288)[:, None]
    flow = daily + rng.standard_normal((T, n)) * 5
    for t in range(3, T):
        flow[t, 1:] += 0.3 * (flow[t - 3, :-1] - daily[t - 3, 0])
    flow = flow.astype(np.float32)
    flow[rng.random((T, n)) < 0.01] = np.nan
    inputs = np.where(np.isfinite(flow), flow, np.nanmean(flow, 0)).astype(np.float32)
    timestamps = 1_609_459_200_000_000_000 + np.arange(T, dtype=np.int64) * 300_000_000_000
    np.savez_compressed(path, flow=flow, inputs=inputs, timestamps=timestamps, sensors=np.arange(n), road=road)


def run(*args):
    proc = subprocess.run([PY, "-W", "ignore", *args], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    return proc.stdout


def test_pipeline_end_to_end(tmp_path):
    data = tmp_path / "toy.npz"
    make_dataset(data)
    out = run("scripts/build_prior.py", "--data", str(data), "--graphs", str(tmp_path / "graphs"), "--granger-windows", "2")
    assert "edges" in out
    run("scripts/train.py", "--data", str(data), "--graphs", str(tmp_path / "graphs"), "--results", str(tmp_path / "results"),
        "--granger-windows", "2", "--epochs", "2", "--max-windows", "24", "--hidden", "16", "--layers", "1",
        "--horizons", "1", "3", "12", "--device", "cpu")
    run_dir = tmp_path / "results" / "gcmgt__seed42"
    results = json.loads((run_dir / "results.json").read_text())
    assert set(results["test"]) >= {"1", "3", "12", "mean"}
    assert all(np.isfinite(results["test"][h]["mae"]) for h in ("1", "3", "12"))
    for h in (1, 3, 12):
        z = np.load(run_dir / f"test_predictions_h{h}.npz")
        assert {"targets", "predictions", "rows", "timestamps", "sensors", "input_fully_observed"} <= set(z.files)
        assert z["targets"].shape == z["predictions"].shape == (len(z["rows"]), 10)
    assert (run_dir / "history.csv").exists() and (run_dir / "checkpoints" / "best.pt").exists()
    # the cached prior is reused, not re-estimated
    again = run("scripts/build_prior.py", "--data", str(data), "--graphs", str(tmp_path / "graphs"), "--granger-windows", "2")
    assert "estimating" not in again


def test_gradient_accumulation_equals_a_real_batch(tmp_path):
    import torch

    sys.path.insert(0, str(ROOT))
    from gcmgt.config import DataConfig, GrangerConfig, ModelConfig, TrainConfig
    from gcmgt.data import load
    from gcmgt.train import Trainer

    path = tmp_path / "toy.npz"
    make_dataset(path, n=8, days=8)
    dc = DataConfig.from_file(str(path))
    gc = GrangerConfig(windows=2, out_dir=str(tmp_path / "graphs"))
    data = load(dc)

    def one_epoch(micro):
        cfg = TrainConfig(batch_size=4, micro_batch=micro, epochs=1, results_dir=str(tmp_path / f"r{micro}"), grad_clip=0.0,
                          model=ModelConfig(hidden=16, layers=1, heads=4, ffn=32, sem_dim=16, sem_heads=4, fusion_hidden=8,
                                            refine_dim=8, dropout=0.0))
        tr = Trainer(data, dc, gc, cfg, torch.device("cpu"), max_windows=8)    # includes windows with missing targets: the normalisation must be batch-wide
        tr.train_epoch(tr._train_loader(1), torch.optim.SGD(tr.model.parameters(), 1e-2), _Const())
        return [p.detach().clone() for p in tr.model.parameters()]

    a, b = one_epoch(4), one_epoch(2)
    assert max(float((x - y).abs().max()) for x, y in zip(a, b)) < 1e-6        # SGD: Adam would amplify float noise


class _Const:
    def step(self):
        pass


def test_loss_parts_sum_to_the_full_batch_loss():
    import torch

    sys.path.insert(0, str(ROOT))
    from gcmgt.train import masked_loss_part

    torch.manual_seed(0)
    pred, tgt = torch.randn(6, 12, 5) * 3, torch.randn(6, 12, 5) * 3
    mask = torch.rand(6, 12) > 0.3
    mask[:, 4] = False                                              # a step that no window of the batch observes
    counts = mask.sum(0).float()
    for kind in ("huber", "mse", "mae"):
        full = masked_loss_part(pred, tgt, mask, counts, kind)
        parts = sum(masked_loss_part(pred[i:i + 2], tgt[i:i + 2], mask[i:i + 2], counts, kind) for i in (0, 2, 4))
        assert torch.isclose(full, parts, atol=1e-6), kind
    # reference values: Huber(delta=1) is quadratic below 1 and linear above
    one = masked_loss_part(torch.tensor([[[0.5]]]), torch.tensor([[[0.0]]]), torch.tensor([[True]]), torch.tensor([1.0]))
    two = masked_loss_part(torch.tensor([[[3.0]]]), torch.tensor([[[0.0]]]), torch.tensor([[True]]), torch.tensor([1.0]))
    assert torch.isclose(one, torch.tensor(0.125)) and torch.isclose(two, torch.tensor(2.5))


def test_time_limit_stops_training_and_still_tests(tmp_path):
    data = tmp_path / "toy.npz"
    make_dataset(data)
    run("scripts/train.py", "--data", str(data), "--graphs", str(tmp_path / "graphs"), "--results", str(tmp_path / "results"),
        "--granger-windows", "2", "--epochs", "30", "--max-windows", "24", "--hidden", "16", "--layers", "1",
        "--time-limit-hours", "0.000001", "--device", "cpu")
    r = json.loads((tmp_path / "results" / "gcmgt__seed42" / "results.json").read_text())
    assert r["stop_reason"] == "time_limit" and r["epochs"] == 1                 # stopped after the first epoch
    assert np.isfinite(r["test"]["3"]["mae"])                                    # and the test metrics were still produced
