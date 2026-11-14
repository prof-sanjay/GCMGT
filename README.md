# GCMGT: Granger causality-guided multi-graph transformer (re-implementation)

A from-the-paper re-implementation of **GC-MGT** (Wang, Riccardi, Fiorentini & Losa, *A Granger causality-guided
multi-graph transformer framework for traffic flow forecasting*, Discover Artificial Intelligence 6:1219, 2026). The
authors released no code, so this is **not** the authors' implementation; report results as "re-implementation of
GC-MGT on <dataset>", never as the paper's published numbers.

## What is implemented

| Component | Paper | Code |
|---|---|---|
| Pairwise Granger F-tests, Benjamini–Hochberg FDR (α = 0.05), top-K = 15 per source, lag 12, day-long windows | Eq. 2–5 | `gcmgt/granger.py` |
| Fixed prior refined by a neural, state-dependent edge gate; sparsity and smoothness regularisers | Eq. 6–7 | `gcmgt/model.py` (`GrangerRefiner`, `GCMGT`) |
| Physical graph, semantic (multi-head attention) graph, adaptive three-way fusion | Eq. 8–12 | `gcmgt/granger.py` (`physical_graph`), `gcmgt/model.py` (`SemanticGraph`) |
| Multi-scale Conv1D {1,3,5,7}, per-step graph-biased attention over sensors, sigmoid gate, direct decoder | Eq. 13–19 | `gcmgt/model.py` |
| MSE + graph regularisers + prediction smoothness | Eq. 20–21 | `gcmgt/model.py`, `gcmgt/train.py` |

The SUMO intervention study (section 3.4) is not reproduced; it needs the FI-PI-LI corridor and a traffic simulator.
Hyperparameters follow the paper's Table 4.

## Deviations from the paper

Every departure is deliberate and listed as D1–D9 in the docstring of `gcmgt/model.py`. The ones that matter:

- **D1 inputs.** The paper uses speed, flow and occupancy; LargeST-GBA has flow only. Inputs are per-sensor standardised
  flow plus sin/cos of time of day and day of week.
- **D2 physical graph.** The LargeST road adjacency already holds distance-decayed weights in (0, 1], the form of Eq. 8;
  it is max-symmetrised and its diagonal is zeroed.
- **D3 prior aggregation.** The paper fixes `A_Granger` offline but does not say how per-window tests are combined. Eight
  day-long windows are spread over the training split; BH is applied within each window as in the paper, an edge is kept
  when significant in at least half of them, and scored by its median F.
- **D4 differencing.** All series are first-differenced (the paper differences only ADF-non-stationary ones), so every pair
  has the same effective sample size.
- **D5/D6 regularisers.** Eq. 7 sums over N² entries, which at 2,056 sensors (the paper: 170–325) would swamp the loss, so
  both norms are divided by N. "Consecutive windows" are approximated by the same window without its last step.
- **D9 memory.** The per-step N × N attention is run one step at a time with gradient checkpointing (identical maths,
  about twice the compute), and the default batch size is 2.

## Installation

```
pip install -r requirements.txt
```

Python ≥ 3.10, PyTorch ≥ 2.0. A GPU is needed for full-scale training (2,056 sensors).

## Data

`scripts/prepare_data.py` writes the same `.npz` format as the sibling HGP package (`flow`, `inputs`, `timestamps`,
`sensors`, `road`, `dataset`), so one prepared file serves both and the two models see identical windows, splits and
metrics.

```
python scripts/prepare_data.py --dataset gba      --traffic gba_his_raw_2021.h5 --adjacency gba_rn_adj.npy --out data/gba_2021.npz
python scripts/prepare_data.py --dataset pems_bay --traffic pems-bay.h5         --adjacency adj_mx_bay.pkl --out data/pems_bay.npz
```

## Usage

```
python scripts/build_prior.py --data data/gba_2021.npz      # optional: estimate and cache the Granger prior
python scripts/train.py --seed 42                           # the HGP training recipe (default); final runs: seeds 42, 7, 123
python scripts/train.py --seed 42 --epochs 10 --epoch-windows 20000       # same recipe, reduced budget
python scripts/train.py --seed 42 --paper                   # the GC-MGT paper's own training recipe instead
```

Both train on all 12 output steps and score h = 3, 6, 12.

### Training recipe (default): identical to HGP

So that GC-MGT and HGP are compared under the same hyperparameters, the defaults are HGP's:

| Setting | Value |
|---|---|
| Optimiser, learning rate, weight decay | AdamW, 3e-4, 1e-4 |
| Batch size | 16 (as 8 accumulated micro-batches of 2; see below) |
| Epochs, warm-up, schedule | up to 40, 1 epoch linear warm-up, cosine decay |
| Early stopping | patience 6 on the validation mean MAE over h = 3, 6, 12 |
| Gradient clipping | 5.0 |
| Loss | Huber, delta = 1.0 on standardised flow, over the observed targets of the 12 output steps |
| Seeds | 42 (final runs 42, 7, 123, one per call) |

Two notes. The Huber delta is not part of the recipe table and is set to 1.0 (the common default and the value DSTFGCN uses);
`--loss mae` switches to masked MAE, which is what the HGP code in this repository's sibling package currently trains on, so
check that the two losses agree before reporting a comparison. And the model's own regularisers (graph sparsity, graph
smoothness, prediction smoothness, Eq. 7 and 20) stay part of the objective, as they are part of GC-MGT.

Batch 16 at 2,056 sensors is built by gradient accumulation over micro-batches of 2, since the per-step N x N attention does
not fit otherwise. This is exact: losses are normalised by the observed-target counts of the whole batch, so the update equals
that of one real batch (`tests/test_train.py`). It costs time, not accuracy.

### The paper's own recipe (`--paper`)

| Setting | Paper (section 4.2, Table 4) | `--paper` here |
|---|---|---|
| Optimizer, learning rate | Adam, 1e-3 | Adam, 1e-3 |
| Weight decay, dropout | 1e-4, 0.1 | 1e-4, 0.1 |
| Batch size | 32 | 32, accumulated from micro-batches of 2 |
| Loss | MSE (+ Eq. 7, Eq. 20) | the same |
| Schedule, gradient clipping | none stated | none (constant lr, no clipping) |
| Epochs | early stopping; the reported run stopped at 70 | up to 100 with early stopping |
| Architecture (d, layers, heads, d_ff, d_s, h_s, d_f, kernels, K, p, w_G, α, λ's) | Table 4 | identical (`ModelConfig`, `GrangerConfig`) |

Use this only to check the re-implementation against the paper; it is not the comparison with HGP. Differences from the paper that remain, none of them a hyperparameter: the paper's early-stopping patience is not stated (6 here), it
splits 70 / 10 / 20 where this repository uses 80 / 10 / 10 so that GC-MGT and HGP are scored on identical windows, and it
reports the mean of five seeds.

The prior costs about 2 minutes per estimation window at 2,056 sensors on a laptop CPU (measured: 134 s), so the default
eight windows take about 18 minutes once; it is cached in `graphs/`. Training time per sample on a GPU has not been
measured; run a short `--epoch-windows` test first.

Outputs in `results/gcmgt__seed<seed>/`: `config.json`, `history.csv`, `checkpoints/best.pt`, `results.json` (validation
and test MAE / RMSE / MAPE per horizon) and `test_predictions_h<h>.npz` in the same layout as the HGP runs, so paired
tests work across the two packages.

## Notebook

`notebooks/gcmgt_kaggle.ipynb` produces the final result in one Kaggle run ("Save & Run All", nothing to change): environment check,
code and data discovery (the code from `REPO_URL` or from a Kaggle dataset holding this folder; `pems-bay.h5` + `adj_mx_bay.pkl`, or for
`DATASET = "gba"` the prepared `gba_2021.npz` the HGP runs used), tests, the cached Granger prior, a timing step, seed 42 (`SEEDS` takes more),
and a results cell with validation and test metrics (plus mean and standard deviation when several seeds are run) and `summary.json` / `summary.csv`.
`DATASET` selects PEMS-BAY (default; 325 sensors, 70 / 10 / 20 split) or LargeST-GBA. `RECIPE` selects the training recipe: `"paper"`
(the paper's own, the notebook default: `--paper`) or `"hgp"` (the HGP recipe for the fair comparison), and `EPOCHS` (default 40) is the
epoch cap. Results go to `results_<RECIPE>/`, so the two recipes never overwrite each other.

The timing step measures one warm epoch on the GPU. If the seeds of 40 full epochs would not fit in `BUDGET_HOURS` (10.5), it lowers the
number of random training windows per epoch (`EPOCH_WINDOWS`) to fit, prints that prominently, and records it in each run's
`config.json`; a seed is also not started if it could not finish inside the 12 h session. On PEMS-BAY with a normal GPU the full protocol is
expected to fit. The micro-batch (16 on PEMS-BAY, 2 on GBA) only affects speed and memory, because gradient accumulation is exact.

## Protocol

Input window 12 steps (5 min), direct 12-step output trained jointly, scoring at h = 3, 6, 12 (15 / 30 / 60 min).
The training recipe is the HGP one (above). MAPE counts only targets above a dataset-specific floor.

## Tests

```
pytest tests
```

The tests check the batched Granger statistics against a direct per-pair least-squares reference, detection of a planted
causal edge, the Benjamini–Hochberg cut-off, the top-K and row-normalisation properties of the prior, equivalence of
checkpointed and plain attention (outputs and gradients), and the scripts end to end on a synthetic dataset.
