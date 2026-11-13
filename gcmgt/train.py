import csv
import itertools
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .config import DataConfig, GrangerConfig, TrainConfig
from .data import Data, Windows
from .granger import load_or_build_prior, physical_graph
from .metrics import horizon_metrics
from .model import build_model, count_parameters


def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_loss_part(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, step_counts: torch.Tensor,
                     kind: str = "huber", delta: float = 1.0) -> torch.Tensor:
    """
    The part of the masked training loss contributed by one micro-batch of a larger batch. pred, target [B, S, N]; mask [B, S].
    Element losses are divided by the observed-window counts `step_counts` [S] of the WHOLE batch and the output steps present in
    the batch are averaged with equal weight, so the parts of one batch sum to exactly the loss of the full batch (a batch that is
    not split passes its own mask counts).
    """
    e = (pred - torch.nan_to_num(target)).abs()
    if kind == "huber":
        e = torch.where(e <= delta, 0.5 * e ** 2, delta * (e - 0.5 * delta))
    elif kind == "mse":
        e = e ** 2
    elif kind != "mae":
        raise ValueError(f"unknown loss {kind!r}")
    err = (e * mask.to(pred.dtype).unsqueeze(-1)).sum((0, 2))
    present = (step_counts > 0).to(pred.dtype)
    return ((err / (step_counts * pred.shape[-1]).clamp_min(1.0)) * present).sum() / present.sum().clamp_min(1.0)


class Trainer:
    def __init__(self, data: Data, data_config: DataConfig, granger_config: GrangerConfig, config: TrainConfig,
                 device: torch.device | None = None, max_windows: int | None = None):
        self.data, self.dc, self.gc, self.config = data, data_config, granger_config, config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seed_everything(config.seed)
        self.prior = load_or_build_prior(data, granger_config)
        self.model = build_model(data.num_nodes, self.prior["A"], physical_graph(data.road), config.model,
                                 data_config.out_steps).to(self.device)
        steps = tuple(range(1, data_config.out_steps + 1))
        self.datasets, self.loaders = {}, {}
        for split in (data.train, data.val, data.test):
            ds = Windows(split, data_config.seq_len, steps)
            if max_windows:
                ds = Windows(split, data_config.seq_len, steps, ds.starts[:max_windows])
            self.datasets[split.name] = ds
        self.micro = config.micro_batch or config.batch_size
        if config.batch_size % self.micro:
            raise ValueError("batch_size must be a multiple of micro_batch")
        for name in ("val", "test"):
            self.loaders[name] = DataLoader(self.datasets[name], self.micro, shuffle=False, collate_fn=Windows.collate)
        self.run_dir = config.run_dir()
        (self.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    def _train_loader(self, epoch: int) -> DataLoader:
        ds, n = self.datasets["train"], self.config.epoch_windows
        if n and n < len(ds):
            pick = np.sort(np.random.default_rng(self.config.seed * 1000 + epoch).choice(len(ds), n, replace=False))
            ds = Subset(ds, pick.tolist())
        return DataLoader(ds, self.micro, shuffle=True, collate_fn=Windows.collate,
                          generator=torch.Generator().manual_seed(self.config.seed * 1000 + epoch))

    def _schedule(self, optimizer, steps_per_epoch: int):
        total, warm = self.config.epochs * steps_per_epoch, max(1, self.config.warmup_epochs * steps_per_epoch)
        if self.config.schedule == "constant":
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)

        def factor(step):
            if step < warm:
                return (step + 1) / warm
            return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)

    def train_epoch(self, loader, optimizer, scheduler) -> float:
        """
        One epoch. A batch of `batch_size` windows is processed as micro-batches whose gradients are summed, which gives exactly
        the update of one real batch: the MSE is normalised by the observed-target counts of the whole batch (not of each
        micro-batch), and the per-window auxiliary loss is weighted by the micro-batch's share of the batch. A short last batch
        is stepped as it is.
        """
        self.model.train()
        cfg, group_size = self.config, self.config.batch_size // self.micro
        total, count, it = 0.0, 0, iter(loader)
        while True:
            group = list(itertools.islice(it, group_size))
            if not group:
                break
            group = [tuple(t.to(self.device) for t in b[:4]) for b in group]
            step_counts = sum(b[3].sum(0) for b in group).to(torch.float32)          # observed windows per output step
            n = sum(len(b[0]) for b in group)
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0
            for x, tf, y, m in group:
                pred, aux = self.model(x, tf)
                loss = masked_loss_part(pred, y, m, step_counts, cfg.loss, cfg.huber_delta) + aux * len(x) / n
                loss.backward()
                batch_loss += float(loss.detach())
            total += batch_loss * n
            count += n
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
        return total / count

    @torch.no_grad()
    def predict(self, name: str):
        """Per scored horizon: targets and predictions in vehicles (observed windows only) and window starts."""
        self.model.eval()
        H = self.dc.horizons
        targets, preds, starts = ({h: [] for h in H} for _ in range(3))
        mean, std = (torch.as_tensor(v, device=self.device) for v in (self.data.mean, self.data.std))
        for x, tf, y, m, s in self.loaders[name]:
            x, tf, y = (t.to(self.device) for t in (x, tf, y))
            pred, _ = self.model(x, tf)
            pred, y = pred * std + mean, y * std + mean
            for h in H:
                keep = m[:, h - 1].numpy()
                if keep.any():
                    preds[h].append(pred[keep, h - 1].cpu().numpy())
                    targets[h].append(y[keep, h - 1].cpu().numpy())
                    starts[h].append(s.numpy()[keep])
        cat = lambda d: {h: np.concatenate(v) if v else np.zeros((0, self.data.num_nodes)) for h, v in d.items()}
        return cat(targets), cat(preds), cat(starts)

    def fit(self) -> dict:
        cfg = self.config
        opt_cls = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam}[cfg.optimizer]
        optimizer = opt_cls(self.model.parameters(), cfg.learning_rate, weight_decay=cfg.weight_decay)
        steps_per_epoch = math.ceil(min(cfg.epoch_windows or 10**12, len(self.datasets["train"])) / cfg.batch_size)
        scheduler = self._schedule(optimizer, steps_per_epoch)
        best, bad, t0, epoch = {"epoch": 0, "mae": float("inf")}, 0, time.time(), 0
        self._json("config.json", {"train": asdict(cfg), "data": asdict(self.dc), "granger": asdict(self.gc),
                                   "prior_edges": int((self.prior["A"] > 0).sum()), "prior_cache": self.prior["cache"],
                                   "parameters": count_parameters(self.model), "device": str(self.device)})
        with open(self.run_dir / "history.csv", "w", newline="") as f:
            writer = None
            for epoch in range(1, cfg.epochs + 1):
                t1 = time.time()
                train_loss = self.train_epoch(self._train_loader(epoch), optimizer, scheduler)
                targets, preds, _ = self.predict("val")
                val = horizon_metrics(targets, preds, self.dc.mape_min_flow)
                row = {"epoch": epoch, "train_loss": train_loss, "val_mae": val["mean"]["mae"], "val_rmse": val["mean"]["rmse"],
                       **{f"val_mae_h{h}": val[h]["mae"] for h in self.dc.horizons},
                       "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - t1}
                writer = writer or csv.DictWriter(f, fieldnames=list(row))
                if epoch == 1:
                    writer.writeheader()
                writer.writerow(row)
                f.flush()
                improved = val["mean"]["mae"] < best["mae"]
                if improved:
                    best, bad = {"epoch": epoch, "mae": val["mean"]["mae"], "val": val}, 0
                    torch.save(self.model.state_dict(), self.run_dir / "checkpoints" / "best.pt")
                else:
                    bad += 1
                print(f"epoch {epoch:3d}  loss {train_loss:.4f}  val MAE {val['mean']['mae']:.3f}  {row['seconds']:.0f}s"
                      f"{'  *' if improved else ''}", flush=True)
                if bad >= cfg.patience:
                    break

        self.model.load_state_dict(torch.load(self.run_dir / "checkpoints" / "best.pt", map_location=self.device))
        targets, preds, starts = self.predict("test")
        test = horizon_metrics(targets, preds, self.dc.mape_min_flow)
        self._save_predictions(targets, preds, starts)
        results = {"run": self.run_dir.name, "parameters": count_parameters(self.model), "best_epoch": best["epoch"],
                   "epochs": epoch, "validation": best["val"], "test": test,
                   "training_seconds": time.time() - t0, "device": str(self.device)}
        self._json("results.json", results)
        print("test  " + "  ".join(f"h{h}: RMSE {test[h]['rmse']:.3f} MAE {test[h]['mae']:.3f}" for h in self.dc.horizons))
        return results

    def _save_predictions(self, targets, preds, starts) -> None:
        split, T = self.data.test, self.dc.seq_len
        for h in self.dc.horizons:
            rows = starts[h] + T + h - 1
            np.savez_compressed(self.run_dir / f"test_predictions_h{h}.npz", targets=targets[h].astype(np.float32),
                                predictions=preds[h].astype(np.float32), rows=(split.offset + rows).astype(np.int64),
                                timestamps=split.timestamps[rows], sensors=self.data.sensors,
                                input_fully_observed=np.array([split.observed[s:s + T].all() for s in starts[h]]))

    def _json(self, name: str, obj) -> None:
        (self.run_dir / name).write_text(json.dumps(obj, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
