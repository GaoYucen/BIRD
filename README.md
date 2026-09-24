# BIRD — Modern PyTorch Reproduction

This branch is the clean, self-contained PyTorch reproduction of the ICDE 2023 BIRD work.

## Main method

**BIRD-FLL128-Restartable**

Article-body implementation:
- bird_main.py — main BIRD entry
- bird_olsc.py — DChasing + online strategy selection
- restartable_strategies.py — Expert / DTP / DP / DDPG with restartable state
- selectors_olsc.py — FPL / FLL family
- pricing_dp.py and pricing_rule.py — compact baseline pricing helpers

The paper body should refer only to BIRD-FLL128-Restartable as BIRD.
Alternative selectors and modern diagnostics are kept under backup/.

## Reproduction

Frozen DDPG checkpoint: artifacts/ddpg_actor_critic.pt

Run the ICDE-style main experiments:

    python run_icde_main_repro.py --actor artifacts/ddpg_actor_critic.pt --output-dir artifacts/icde_main_repro --search-seeds 80
    python plot_icde_main_repro.py --input artifacts/icde_main_repro/results.json --output artifacts/icde_main_repro/figures

Retrain DDPG if needed:

    python train_ddpg.py --data-dir data --output artifacts/ddpg_actor_critic.pt --device cpu --pretrain-epochs 40 --sim-steps 6000 --warmup 1000 --batch-size 64 --updates-per-step 1 --noise-std 5.0

## Main results

- artifacts/main_5seed_summary.csv — compact baseline comparison
- artifacts/icde_main_repro/results.json — raw ICDE Fig. 7–11 values
- artifacts/icde_main_repro/figures/ — PNG + SVG figures
- artifacts/route_data_summary.json — real multi-route data summary

## Documentation

- MAIN_BIRD_ICDE_REPRO_20260925.md — main version and ICDE figure reproduction status
- REPRO_PACKAGE.md — reproducibility package contents
- THEORY_AUDIT_20260924.md — theory-to-code audit
- OLSC_FLL_RESULTS_20260925.md — selector and switching diagnostics

## Data

data/ contains only files still needed by the current PyTorch workflow and route-level follow-up:
- x_pretrain.npy
- y_pretrain.npy
- YIK_QZH_training_data_source.xlsx
- 2hours_price_setting_env.csv
- num_count.pkl
- all_info_qugang_zhida_TF_2019.csv

The original TensorFlow/virtualenv implementation and four-year-old experimental scripts were intentionally removed from this branch. They remain available in repository history / the original branch.
