# BIRD PyTorch reproducibility package

Current branch: `pytorch-repro`

## Frozen article-body method

**BIRD-FLL128-Restartable**

Main entry:
- `bird_main.py`

Core implementation:
- `bird_olsc.py` — BIRD execution and DChasing loop
- `restartable_strategies.py` — restartable Expert / DTP / DP / DDPG
- `selectors_olsc.py` — FPL / FLL / FPL* / FLL* selector implementations
- `model.py`, `ddpg.py`, `train_ddpg.py` — PyTorch DDPG implementation and training

The article body should use only **BIRD-FLL128-Restartable** as “BIRD”.
Other selector variants and older chasing implementations are retained only for
ablation, diagnostics, and backup.

## Main documentation

- `README.md`
- `MAIN_BIRD_ICDE_REPRO_20260925.md` — main version and ICDE-2023 figure reproduction status
- `OLSC_FLL_RESULTS_20260925.md` — selector / switching experiments
- `THEORY_AUDIT_20260924.md` — theory-to-code audit and resolved implementation issues

## Main numerical results

- `artifacts/main_5seed_summary.csv` — compact baseline table
- `artifacts/icde_main_repro/results.json` — raw values for ICDE Fig. 7–11 reproduction
- `artifacts/olsc_5seed/summary.json` — selector comparison
- `artifacts/olsc_lazy_extension.json` — high-scale FLL / FLL* sensitivity
- `artifacts/route_data_summary.json` — real shipping-lane data summary for rebuilding Fig. 12

## Main article figures

All are committed in both PNG and editable vector SVG:

- `artifacts/icde_main_repro/figures/fig7_representative_sequences.{png,svg}`
- `artifacts/icde_main_repro/figures/fig8_revenue_loss_heatmap.{png,svg}`
- `artifacts/icde_main_repro/figures/fig9_grid_views.{png,svg}`
- `artifacts/icde_main_repro/figures/fig10_sensitivity.{png,svg}`
- `artifacts/icde_main_repro/figures/fig11_sellout_index.{png,svg}`

## Reproduction scripts

- `run_icde_main_repro.py`
- `plot_icde_main_repro.py`
- `run_olsc_5seed.py`
- `run_lazy_extension_5seed.py`

Recommended main-result workflow after a DDPG checkpoint is available:

```bash
cd pytorch_repro

python run_icde_main_repro.py \
  --actor artifacts/ddpg_actor_critic.pt \
  --output-dir artifacts/icde_main_repro \
  --search-seeds 80

python plot_icde_main_repro.py \
  --input artifacts/icde_main_repro/results.json \
  --output artifacts/icde_main_repro/figures
```

To retrain the PyTorch DDPG checkpoint:

```bash
python train_ddpg.py \
  --data-dir data \
  --output artifacts/ddpg_actor_critic.pt \
  --device cpu \
  --pretrain-epochs 40 \
  --sim-steps 6000 \
  --warmup 1000 \
  --batch-size 64 \
  --updates-per-step 1 \
  --noise-std 5.0
```

## Important interpretation notes

1. Fig. 8 / Fig. 9 / Fig. 10(b) reproduce the main ICDE theoretical trends well.
2. Fig. 7 reproduces DP/Expert/BIRD winner heterogeneity, but the current buyer generator did not produce a DDPG-winning sequence in seeds 0–79.
3. Fig. 10(a) should use the corrected theoretical no-purchase semantics rather than force numerical agreement with the released legacy code.
4. Fig. 12 raw multi-route data are present, but the original route-level evaluation pipeline is not in the public repository and must be rebuilt before claiming exact reproduction.
