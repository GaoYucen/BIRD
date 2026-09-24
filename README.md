# BIRD — Modern PyTorch Reproduction

Clean, self-contained PyTorch reproduction of **BIRD (ICDE 2023)**.

## Main method

**BIRD-FLL128-Restartable** is the only BIRD version used for article-body results.

### Repository layout

```text
bird/                  # algorithm package
  algorithm.py         # frozen article-body BIRD configuration
  olsc.py              # DChasing + online strategy selection
  strategies.py        # restartable Expert / DTP / DP / DDPG
  selectors.py         # FPL / FLL family
  actor.py             # checkpoint-backed actor wrapper
  baselines/           # DP and rule-based baseline helpers
  rl/                  # PyTorch DDPG model, agent, environment
experiments/           # training and ICDE reproduction entry points
docs/                  # theory audit and reproduction notes
artifacts/
  checkpoints/         # frozen DDPG checkpoint
  results/             # numerical results
  figures/             # publication-ready PNG + SVG figures
data/                  # data required by the current workflow
archive/               # old modern diagnostics / backup variants
```

## Quick start

Install in editable mode:

```bash
pip install -e .
```

Reproduce the ICDE-style main results:

```bash
python -m experiments.reproduce_icde \
  --actor artifacts/checkpoints/ddpg_actor_critic.pt \
  --output-dir artifacts/results/icde_main \
  --search-seeds 80

python -m experiments.plot_icde \
  --input artifacts/results/icde_main/results.json \
  --output artifacts/figures/icde_main
```

Run the frozen main BIRD directly:

```bash
python -m bird.olsc \
  --actor artifacts/checkpoints/ddpg_actor_critic.pt \
  --selector fll \
  --selector-multiplier 128 \
  --seed 0
```

Retrain DDPG if needed:

```bash
python -m experiments.train_ddpg \
  --data-dir data \
  --output artifacts/checkpoints/ddpg_actor_critic.pt
```

## Main outputs

- `artifacts/results/main_5seed_summary.csv` — compact baseline comparison
- `artifacts/results/icde_main/results.json` — raw Fig. 7–11 reproduction values
- `artifacts/figures/icde_main/` — PNG + SVG figures
- `artifacts/results/route_data_summary.json` — real multi-route data summary

## Documentation

- `docs/main_reproduction.md` — main version and ICDE-figure reproduction status
- `docs/reproducibility.md` — reproducibility package and commands
- `docs/theory_audit.md` — theory-to-code audit
- `docs/olsc_results.md` — selector / switching diagnostics

`archive/` is retained only for backup and ablation history; it is not part of the article-body implementation.
