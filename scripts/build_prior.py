"""
Estimate and cache the fixed Granger prior only (the slow step: ~2 minutes per estimation window at 2,056 sensors on a
laptop CPU). scripts/train.py does the same on demand; run this first to separate the two.

    python scripts/build_prior.py --data data/gba_2021.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gcmgt.config import DataConfig, GrangerConfig
from gcmgt.data import load
from gcmgt.granger import load_or_build_prior


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", default=DataConfig.path)
    ap.add_argument("--graphs", default=GrangerConfig.out_dir)
    ap.add_argument("--granger-windows", type=int, default=GrangerConfig.windows)
    ap.add_argument("--top-k", type=int, default=GrangerConfig.top_k)
    args = ap.parse_args()
    data = load(DataConfig.from_file(args.data))
    prior = load_or_build_prior(data, GrangerConfig(windows=args.granger_windows, top_k=args.top_k, out_dir=args.graphs))
    A = prior["A"]
    print(f"{prior['cache']}: {int((A > 0).sum()):,} edges, {(A > 0).sum(1).mean():.1f} outgoing per source, "
          f"{int(((A > 0).sum(1) == 0).sum())} sources without any edge")


if __name__ == "__main__":
    main()
