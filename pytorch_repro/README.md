# PyTorch reproduction

## Current main BIRD (article-body version)

The frozen main implementation is **BIRD-FLL128-Restartable**.

Entry point: `bird_main.py`

Main ICDE-figure reproduction status:
`MAIN_BIRD_ICDE_REPRO_20260925.md`

All FTL/FPL/FPL*/FLL* alternatives, OracleFixed diagnostics, and older
chasing implementations are retained for backup/ablation only and should not
be presented as separate BIRD variants in the article body.

This folder modernizes the executable path of the ICDE 2023 BIRD code without
requiring Python 3.6 or TensorFlow 1.x.

## What is preserved

- Original BIRD strategy-chasing logic.
- Expert, decision-table, and DP pricing baselines.
- The old stacked-LSTM actor topology (2 x 128 hidden units, action range 40).
- Repository data used to generate/train the legacy RL policy.

## What is intentionally modernized

- TensorFlow 1.x actor is replaced by PyTorch.
- The actor is retrained from the committed `x_pretrain.npy/y_pretrain.npy`
  rather than loading the old TF checkpoint.
- The pretraining input is interpreted semantically as
  `[price=0, utilization, remaining_time]`.
- The BIRD exploration epsilon is computed after inventory initialization.
  Use `--legacy-epsilon-zero` to reproduce the legacy code bug where epsilon
  was effectively zero.
- Decision-table cumulative volume is kept across steps rather than reset in
  the inner loop.

This is therefore a modern executable reproduction, not bit-for-bit numerical
reproduction of the 2022 TensorFlow environment.

## Usage

```bash
cd pytorch_repro
python train_actor.py --epochs 80
python chasing_pytorch.py --actor actor_pretrained.pt --seed 10
```

For a quick smoke run:

```bash
python train_actor.py --epochs 5
python chasing_pytorch.py --actor actor_pretrained.pt --seed 10 --T 1000
```
