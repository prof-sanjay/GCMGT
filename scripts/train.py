"""
Estimate the Granger prior (cached) and train / evaluate GC-MGT.

    python scripts/train.py --seed 42                         # the HGP recipe (default): identical hyperparameters to HGP
    python scripts/train.py --seed 42 --paper                 # the GC-MGT paper's own training recipe (see README)
    python scripts/train.py --seed 42 --epochs 10 --epoch-windows 20000        # reduced budget
    python scripts/train.py --data data/gba_2021.npz --max-windows 48 --epochs 2 --granger-windows 3    # smoke test
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gcmgt.config import DataConfig, GrangerConfig, ModelConfig, TrainConfig
from gcmgt.data import load
from gcmgt.train import Trainer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data", default=DataConfig.path)
    ap.add_argument("--horizons", type=int, nargs="+", default=list(DataConfig.horizons), help="scored horizons, in 5-minute steps")
    ap.add_argument("--graphs", default=GrangerConfig.out_dir, help="where the Granger prior is cached")
    ap.add_argument("--results", default="results")
    ap.add_argument("--epochs", type=int, default=None, help="default 40; 100 with --paper (early stopping applies)")
    ap.add_argument("--loss", choices=["huber", "mse", "mae"], default=None, help="default huber; mse with --paper")
    ap.add_argument("--paper", action="store_true", help="the GC-MGT paper's recipe instead of the HGP one: Adam, lr 1e-3, wd 1e-4, batch 32, MSE, constant lr, no clipping")
    ap.add_argument("--batch-size", type=int, default=None, help="effective batch (default 16; 32 with --paper)")
    ap.add_argument("--micro-batch", type=int, default=None, help="windows per forward pass; the batch is accumulated from these")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epoch-windows", type=int, default=None, help="random training windows per epoch (default: all)")
    ap.add_argument("--hidden", type=int, default=ModelConfig.hidden)
    ap.add_argument("--layers", type=int, default=ModelConfig.layers)
    ap.add_argument("--dropout", type=float, default=ModelConfig.dropout)
    ap.add_argument("--granger-windows", type=int, default=GrangerConfig.windows, help="day-long windows combined into the prior")
    ap.add_argument("--top-k", type=int, default=GrangerConfig.top_k)
    ap.add_argument("--max-windows", type=int, default=None, help="limit windows per split (smoke tests)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    dc = DataConfig.from_file(args.data)
    dc.horizons = tuple(args.horizons)
    gc = GrangerConfig(windows=args.granger_windows, top_k=args.top_k, out_dir=args.graphs)
    overrides = {k: v for k, v in dict(epochs=args.epochs, batch_size=args.batch_size, micro_batch=args.micro_batch,
                                       learning_rate=args.lr, loss=args.loss).items() if v is not None}
    overrides.update(seed=args.seed, epoch_windows=args.epoch_windows, results_dir=args.results,
                     model=ModelConfig(hidden=args.hidden, layers=args.layers, dropout=args.dropout))
    tc = TrainConfig.paper(**overrides) if args.paper else TrainConfig(**overrides)
    data = load(dc)
    Trainer(data, dc, gc, tc, torch.device(args.device) if args.device else None, args.max_windows).fit()


if __name__ == "__main__":
    main()
