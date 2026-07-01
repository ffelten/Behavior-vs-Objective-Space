# Behavior vs. Objective Space in MORL

Studying the differences between behavior space and objective space in multi-objective
reinforcement learning (MORL).

## Install

```bash
pip install -e .
```

## Workflow

1. Generate trajectories from trained MORL policies.
2. Train a behavior encoder on those trajectories.
3. Analyze the learned embeddings in the notebooks.

### 1. Generate trajectories

`create_trajectory.py` rolls out the MORL/D policies stored under `MORL_policies/` and
writes per-policy trajectories to `trajectories/`. Configure the run via the constants at
the top of the file (`MODE`, `ENV_ID`).

```bash
python morl_behavior_objective/create_trajectory.py
```

### 2. Train a behavior encoder

`new_main_tr.py` trains the encoder for a single environment/seed:

```bash
python new_main_tr.py -MHC --epochs 100 --train --seed 0 --save_data
```

Run all environments and seeds with the batch scripts:

```bash
./training_transformers.sh      # MO-HalfCheetah and MO-Hopper (transformer + MLP baseline)
./training_transformers_dst.sh  # Deep Sea Treasure variants
```

### 3. Analyze

Open the notebooks in `morl_behavior_objective/analysis/`:
`analysis_cheetah.ipynb`, `analysis_dst.ipynb`, `analysis_others.ipynb`.

## Options

Environment selection (one per run):

| Flag | Environment |
|------|-------------|
| `-MHC` | MO-HalfCheetah |
| `-MHo` | MO-Hopper (3 objectives) |
| `-MHo2` | MO-Hopper (2 objectives) |
| `-MHW` | MO-Highway |
| `-RG` | Resource Gathering |
| `-DSTC` | Deep Sea Treasure (concave) |
| `-DSTS` | Deep Sea Treasure (smooth) |
| `-DSTLR` | Deep Sea Treasure (left-right) |
| `-DSTL` | Deep Sea Treasure (Louvre) |

Common flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--train` | off | Train a model (otherwise load from `--model_dir`) |
| `--epochs` | 500 | Number of training epochs |
| `--seed` | 0 | Random seed |
| `--save_data` | off | Save aggregated policy embeddings to JSON |
| `--use_mlp_baseline` | off | Use the MLP baseline instead of the transformer encoder |
| `--emb_dim` | 3 | Embedding dimension |
| `--d_hid` | 128 | Hidden dimension (DST runs use 32) |
| `--model_dir` | `final_models/no_topo/` | Where models are saved/loaded |
| `--model_prefix` | `be` | Filename prefix (`basic`/ `lstm` for the baselines) |
| `--device` | cuda/mps/cpu | Compute device (auto-detected) |
| `--visualize` | off | Plot trajectory embeddings before aggregation |

Run `python new_main_tr.py --help` for the full list, including set-encoder and
contrastive-loss hyperparameters.
