# Main BIRD baseline and ICDE-2023 figure reproduction status

Date: 2026-09-25

## Main version used in the paper body

The only BIRD variant designated as the current main method is:

**BIRD-FLL128-Restartable**

Implementation:
- restartable Expert / Decision Table / DP / DDPG base strategies;
- theoretical DChasing epsilon `sqrt(C*N/|B|)`;
- semantically correct no-purchase steps;
- full-information reward feedback after every buyer;
- target-strategy restart from BIRD's current state after a selector switch;
- exact Kalai--Vempala random-grid FLL;
- finite-horizon perturbation multiplier = 128x.

Other selector variants (FTL, FPL, FPL*, FLL*, alternative FLL multipliers,
OracleFixed) are retained as diagnostics/ablation backups only. They should not
be shown as BIRD variants in the article body.

## Five-seed main comparison

| Method | Mean revenue | Std | Mean loss vs seed-wise best fixed |
|---|---:|---:|---:|
| DP | 3,024,773 | 7,512 | 0.13% |
| **BIRD-FLL128-Restartable** | **2,999,036** | **54,860** | **0.98%** |
| Expert | 2,796,443 | 191,136 | 7.67% |
| DDPG | 2,573,605 | 42,088 | 15.03% |
| Decision Table (DTP) | 2,166,318 | 42,431 | 28.47% |

BIRD is 0.85% below DP on mean revenue in these mostly DP-friendly sequences,
but exceeds Expert by 7.24%, DDPG by 16.53%, and DTP by 38.44%.

FLL128 uses 354.2 target switches per 5000 buyers on average (7.1% of buyer
steps). This is the current empirical finite-horizon operating point; 128 is
not claimed as a theorem constant.

## ICDE Fig. 7 — representative buyer sequences

The ICDE paper shows three C=205, |B|=5000 examples:
DDPG wins / Expert wins / BIRD wins.

With the corrected main implementation, searching seeds 0--79 gives:

- BIRD wins: 25 sequences;
- DP wins: 36 sequences;
- Expert wins: 19 sequences;
- DDPG wins: 0 sequences;
- Decision Table wins: 0 sequences.

Representative cases found:
- BIRD wins at seed 1:
  - BIRD = 3,059,485
  - Expert = 3,038,134
  - DP = 3,018,110
  - DDPG = 2,599,403
  - DTP = 2,106,200
- Expert wins at seed 16:
  - Expert = 3,057,561
  - BIRD = 3,036,532
  - DP = 3,009,190
  - DDPG = 2,509,448
  - DTP = 2,194,040

Therefore the central Fig. 7 phenomenon ("the best fixed strategy depends on
the buyer sequence, while BIRD can itself win") is reproduced, but the exact
"DDPG wins" panel is not reproduced by the current random buyer generator.
The public repository does not contain the original construction script for
the three specific ICDE buyer sequences. The main-text replacement should use
DP-wins / Expert-wins / BIRD-wins unless the original DDPG-favoring sequence
construction is recovered.

## ICDE Fig. 8 — C x |B| revenue-loss heatmap

The exact visible ICDE grid was reproduced:

C = {51,102,153,204,307,409,512,615,717,819}

|B| = {2.5k,5k,7.5k,10k,12.5k,15k,17.5k,20k,22.5k,25k}

Results are strongly consistent with the ICDE figure.

Most extreme degradation:
- current: |B|=2500, C=819, signed gap = **-76.54%**
- ICDE text: same |B|=2500, C=819, approximately **-81.84%**

Small-inventory setting:
- current: |B|=2500, C=51, signed gap = **-0.86%**
- ICDE text: same setting, approximately **+1.15%**

Across the 100 current grid cells:
- signed gap range = -76.54% to +2.09%;
- mean signed gap = -4.56%;
- BIRD exceeds every fixed strategy in 10 cells.

This is a strong reproduction of the main theoretical phenomenon:
performance degrades when C*N is no longer small relative to |B|.

## ICDE Fig. 9 — 100-grid strategy landscape

The 100-cell grid reproduces the heterogeneity of the best fixed strategy:

- DP best: 55 cells;
- DDPG best: 23 cells;
- Expert best: 19 cells;
- DTP best: 3 cells.

BIRD exceeds all fixed strategies in 10 cells.

Thus the main Fig. 9 conclusion is reproduced: there is no single fixed
pricing strategy that dominates over the full inventory / buyer-length
parameter space, and BIRD remains near the envelope over much of the grid.

## ICDE Fig. 10(a) — epsilon sensitivity

This figure **cannot be numerically reproduced with the corrected theoretical
semantics**, and the discrepancy is informative.

Current seed-0 C=205, |B|=5000 results:

| requested epsilon | effective epsilon | BIRD revenue | signed gap vs best fixed |
|---:|---:|---:|---:|
| 0.072 | 0.072 | 2,981,501 | -1.45% |
| 0.143 | 0.143 | 3,026,741 | +0.05% |
| 0.286 | 0.286 | 2,958,428 | -2.21% |
| 0.573 | 0.573 | 3,010,926 | -0.47% |
| 1.145 | 1.000 | 0 | -100.00% |

The ICDE caption itself states that epsilon >= 1 should always choose the
no-purchase action, yet the published bar for epsilon=1.145 is non-zero. The
legacy code also computed C before inventory initialization, making epsilon
effectively zero in the released implementation. Therefore forcing the modern
code to match the old Fig. 10(a) would reintroduce an implementation bug.

The corrected result should replace the old figure rather than be forced to
numerically match it.

## ICDE Fig. 10(b) — buyer-sequence-length sensitivity

This result reproduces the published trend very well:

| |B| | theoretical epsilon | BIRD revenue | signed gap |
|---:|---:|---:|---:|
| 500 | 0.906 | 305,590 | -89.20% |
| 1250 | 0.573 | 2,668,988 | -11.99% |
| 2500 | 0.405 | 2,917,353 | -6.93% |
| 5000 | 0.286 | 2,958,428 | -2.21% |

The ICDE figure similarly shows very poor performance at |B|=500, followed by
rapid recovery at 1250/2500 and near-best performance at 5000. This is a strong
qualitative reproduction of the vanishing-regret trend.

## ICDE Fig. 11 — sell-out buyer index

Using category-local buyer indices (buyer index counted from the beginning of
each category's selling window), current seed-0 values are approximately:

| Strategy | Cat.1 | Cat.2 | Cat.3 | Cat.4 | Cat.5 |
|---|---:|---:|---:|---:|---:|
| Expert | 101 | 43 | 111 | 313 | 79 |
| DP | 103 | 93 | 106 | 103 | 98 |
| DDPG | 156 | 151 | 152 | 157 | 155 |
| BIRD | 156 | 139 | 157 | 132 | 155 |
| DTP | not sold out | not sold out | not sold out | not sold out | not sold out |

The qualitative point that BIRD trades selling speed for revenue is visible
against DP/Expert in most categories, but the exact ICDE curves are not
reproduced. The corrected Decision Table also does not sell out in this
sequence, unlike the published plot.

## ICDE Fig. 12 — multiple shipping lanes

The public raw CSV is sufficient to confirm that multiple routes exist:

- 34,449 records;
- 14 POL-POD lane pairs;
- 459 voyage identifiers;
- data period roughly Apr 2019 -- Jan 2020.

Five lane pairs have substantial sample sizes:
- YIK -> QZH: 15,743 rows;
- TSN -> NSH: 9,699 rows;
- NSH -> YIK: 3,999 rows;
- YIK -> NSH: 3,312 rows;
- NSH -> TSN: 1,683 rows.

However, the repository does not contain the original Fig. 12 lane-selection,
valuation-construction, and monthly evaluation script. Therefore Fig. 12 is
**feasible to reconstruct from the raw data, but is not yet an exact
reproduction**. It should not be claimed reproduced until that route-level
pipeline is rebuilt explicitly.

## Main figure artifacts

Generated from the single main BIRD version:

- `artifacts/icde_main_repro/figures/fig7_representative_sequences.{png,svg}`
- `artifacts/icde_main_repro/figures/fig8_revenue_loss_heatmap.{png,svg}`
- `artifacts/icde_main_repro/figures/fig9_grid_views.{png,svg}`
- `artifacts/icde_main_repro/figures/fig10_sensitivity.{png,svg}`
- `artifacts/icde_main_repro/figures/fig11_sellout_index.{png,svg}`

Raw values:
- `artifacts/icde_main_repro/results.json`

Main implementation entry:
- `bird_main.py`

Backup / ablation implementations remain in the repository but are not part of
the article-body method definition.
