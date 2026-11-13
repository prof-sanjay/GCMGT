from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Fixed per dataset: split, the smallest target counted in MAPE, and the smallest per-sensor standard deviation.
DATASETS = {
    "gba": {"train_rate": 0.8, "val_rate": 0.1, "mape_min_flow": 10.0, "min_std": 1.0},
    "pems_bay": {"train_rate": 0.7, "val_rate": 0.1, "mape_min_flow": 1.0, "min_std": 0.1},
}


@dataclass
class DataConfig:
    path: str = "data/gba_2021.npz"
    dataset: str = "gba"
    train_rate: float = 0.8
    val_rate: float = 0.1
    seq_len: int = 12
    out_steps: int = 12                       # native output length, trained jointly
    horizons: tuple[int, ...] = (3, 6, 12)    # horizons that are scored (5-minute steps)
    mape_min_flow: float = 10.0
    min_std: float = 1.0

    @classmethod
    def from_file(cls, path: str) -> "DataConfig":
        """The configuration of the dataset a prepared file belongs to (scripts/prepare_data.py records it)."""
        with np.load(path) as z:
            name = str(z["dataset"]) if "dataset" in z.files else "gba"
        return cls(path=path, dataset=name, **DATASETS[name])


@dataclass
class GrangerConfig:
    """The fixed Granger prior A_Granger (paper section 3.3.1.1, Table 4)."""
    window: int = 288          # w_G: one day of 5-minute rows
    lag: int = 12              # p
    alpha: float = 0.05        # Benjamini-Hochberg FDR level
    top_k: int = 15            # K: maximum retained targets per source
    windows: int = 8           # day-long windows combined into one prior (deviation D3)
    out_dir: str = "graphs"


@dataclass
class ModelConfig:
    hidden: int = 64           # d
    layers: int = 4
    heads: int = 8
    ffn: int = 256             # d_ff
    sem_dim: int = 128         # d_s
    sem_heads: int = 8         # h_s
    fusion_hidden: int = 64    # d_f
    refine_dim: int = 32       # width of the refinement encoder f_theta (deviation D7)
    dropout: float = 0.1
    lambda_sparse: float = 0.01   # lambda_1
    lambda_smooth_graph: float = 0.1   # lambda_2
    lambda_smooth_pred: float = 0.01   # lambda_smooth
    checkpoint_attention: bool = True  # recompute the per-step N x N attention in the backward pass (same maths)


@dataclass
class TrainConfig:
    """Defaults are the HGP training recipe, so that GC-MGT and HGP are compared under identical hyperparameters."""
    seed: int = 42                     # final runs: 42, 7, 123
    optimizer: str = "adamw"           # "adamw" | "adam" (Adam with L2 weight decay)
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 16               # effective batch: windows per optimizer step
    micro_batch: int | None = 2        # windows per forward pass (None = batch_size); the batch is accumulated from these
    epochs: int = 40
    schedule: str = "cosine"           # "cosine" (warm-up + cosine decay) | "constant"
    warmup_epochs: int = 1
    patience: int = 6                  # early stopping on the validation mean MAE
    grad_clip: float = 5.0             # 0 = off
    loss: str = "huber"                # "huber" | "mse" | "mae", on standardised flow over the observed targets
    huber_delta: float = 1.0
    epoch_windows: int | None = None   # random training windows per epoch (None = all)
    time_limit_hours: float | None = None   # stop before an epoch that would exceed this, then test the best checkpoint
    results_dir: str = "results"
    model: ModelConfig = field(default_factory=ModelConfig)

    @classmethod
    def paper(cls, **overrides) -> "TrainConfig":
        """The GC-MGT paper's own recipe (section 4.2 / Table 4): Adam, lr 1e-3, weight decay 1e-4, batch 32, MSE, dropout 0.1
        (ModelConfig default); no learning-rate schedule and no gradient clipping are mentioned, so none is used. The paper's
        70-epoch run is an upper bound here (epochs=100 with early stopping)."""
        base = dict(optimizer="adam", learning_rate=1e-3, weight_decay=1e-4, batch_size=32, micro_batch=2, epochs=100,
                    schedule="constant", warmup_epochs=0, grad_clip=0.0, loss="mse")
        base.update(overrides)
        return cls(**base)

    def run_dir(self) -> Path:
        return Path(self.results_dir) / f"gcmgt__seed{self.seed}"
