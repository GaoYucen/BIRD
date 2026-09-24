# BIRD OLSC / DChasing implementation and five-seed validation

Date: 2026-09-25

## 1. What is implemented

This is the current theory-oriented PyTorch implementation on branch
`pytorch-repro`.

### Paired experiment context

Every method within a seed shares exactly the same:

- buyer valuations;
- Expert random-walk noise;
- DChasing epsilon no-purchase mask;
- full-information base-strategy reward sequence.

This removes the Monte-Carlo confounding that affected earlier comparisons.

### Restartable pricing strategies

A common restartable interface is implemented for:

- Expert;
- Decision Table;
- DP;
- DDPG.

When Algorithm 2 changes the target strategy, a fresh target pricing process is
initialized at BIRD's current inventory/time state rather than reading the
target strategy's old global trajectory.

### Theory-consistent DChasing semantics

- no-purchase is implemented semantically as zero sale, not raw price 1;
- the Theorem-3 value is used:
  `epsilon = sqrt(C*N/T)`;
- here C=205, N=2, T=5000, so epsilon=0.2863564213;
- full-information selector feedback is applied after every buyer, including
  epsilon no-purchase rounds;
- target and BIRD states evolve separately inside a chasing episode.

### OLSC selectors

Implemented:

- FTL (diagnostic);
- independent additive uniform FPL;
- exact Kalai--Vempala random-grid FLL;
- independent symmetric-exponential FPL*;
- exact Kalai--Vempala FLL* coupling;
- hindsight OracleFixed (diagnostic only).

For FPL/FLL and FPL*/FLL*, an explicit constant multiplier is exposed around
the theory-derived finite-horizon perturbation scale. This is reported as a
sensitivity parameter rather than hidden tuning, because the ICDE paper states
the asymptotic OLSC bound but does not uniquely specify all finite-horizon
constants for this experimental normalization.

## 2. Base strategies under the paired five-seed protocol

| Method | Mean revenue | Std | Mean loss vs seed-wise best fixed |
|---|---:|---:|---:|
| DP | 3,024,773 | 7,512 | 0.13% |
| Expert | 2,796,443 | 191,136 | 7.67% |
| DDPG | 2,573,605 | 42,088 | 15.03% |
| Decision Table | 2,166,318 | 42,431 | 28.47% |

The current five sequences are therefore relatively favorable to DP; DP is the
best fixed strategy in four of five seeds.

## 3. Core BIRD comparison

| Selector | Mean loss | Std | Mean switches / 5000 | Interpretation |
|---|---:|---:|---:|---|
| OracleFixed + DChasing | 0.13% | 0.29% | 0 | Chasing sanity upper bound |
| FTL + DChasing | 1.52% | 2.63% | 7.2 | Excellent practical stationary-case diagnostic |
| FPL 1x | 1.45% | 1.30% | 3757.8 | Good reward, excessive switching |
| FPL* 32x | -0.12% | 0.43% | 3749.8 | Near-oracle/dynamic reward, excessive switching |
| FLL 32x | 4.61% | 6.12% | 107.4 | Very lazy, some performance cost |
| FLL 64x | 2.67% | 3.70% | 203.2 | Better balance |
| **FLL 128x** | **0.98%** | **1.77%** | **354.2** | **Current lazy-selector Pareto point** |
| FLL 256x | 2.97% | 1.91% | 639.6 | More switching without benefit |
| FLL 512x | 1.51% | 0.84% | 999.0 | Stable but too many switches |
| FLL* 64x | 8.38% | 8.96% | 83.0 | Too sticky |
| FLL* 128x | 6.65% | 7.35% | 175.8 | Too sticky |
| FLL* 256x | 4.72% | 5.21% | 306.8 | Improving |
| FLL* 512x | 1.76% | 2.14% | 492.8 | Good but dominated here by FLL-128x |

Negative loss means the online selector beats every single fixed strategy on
that buyer sequence; this is possible because BIRD is allowed to change its
target online whereas the comparator is the best single fixed strategy.

## 4. FLL-128x per-seed results

| Seed | Loss vs best fixed | Target switches |
|---|---:|---:|
| 0 | 2.21% | 357 |
| 1 | -0.70% | 375 |
| 2 | 0.60% | 276 |
| 3 | 3.36% | 403 |
| 4 | -0.55% | 360 |
| **Mean** | **0.98%** | **354.2** |

Mean revenue is 2,999,036 with a standard deviation of 54,860.

## 5. What the results mean

### DChasing is no longer the bottleneck

With a correct fixed target, OracleFixed+DChasing loses only 0.13% on average.
There are no chasing missing steps in these experiments. This is much stronger
than the earlier 3% result because the current experiment now uses identical
buyer/randomness streams and truly restartable target strategies.

### Independent FPL is not appropriate for BIRD in practice

FPL and FPL* achieve very low regret, but switch target on roughly 75% of
buyers. Since each target switch starts a new chasing episode, this ignores the
operational reason for introducing an OLSC/lazy selector.

### Exact FLL exposes the intended tradeoff

The random-grid FLL creates a clean finite-horizon loss/switch tradeoff:

- 32x: 4.61% loss, 107 switches;
- 64x: 2.67% loss, 203 switches;
- 128x: 0.98% loss, 354 switches.

At larger multipliers the improvement is no longer monotone. Thus 128x is the
current empirical Pareto choice, not a newly claimed theoretical constant.

### FTL is a strong diagnostic but not a worst-case replacement for OLSC

FTL uses only 7.2 switches and reaches 1.52% mean loss in these mostly
stationary sequences. This is an important sanity result, but it does not
replace the paper's no-regret OLSC machinery under adversarial sequences.

## 6. Current default recommendation

For continued BIRD reproduction experiments:

- preserve the theoretical DChasing epsilon;
- use paired randomness for every ablation/comparison;
- use restartable strategy state on each target switch;
- report FTL as a practical diagnostic;
- report exact FLL with a finite-horizon multiplier sensitivity;
- use FLL-128x as the current empirical operating point for this dataset,
  explicitly labeling 128x as finite-horizon tuning rather than a theorem
  constant;
- retain FPL/FPL* only to show the reward/switching extreme.

Files:

- `restartable_strategies.py`
- `selectors_olsc.py`
- `bird_olsc.py`
- `run_olsc_5seed.py`
- `run_lazy_extension_5seed.py`
- `artifacts/olsc_5seed/`
- `artifacts/olsc_lazy_extension.json`
