# BIRD theory-aligned validation — 5 seeds

Date: 2026-09-25

## Scope

This validation uses seeds 0--4, T=5000, inventory argument 200
(effective initial inventory 205 after the legacy +m buffer), maximum
simultaneously active categories N=2, and the Theorem-3 value

- epsilon = sqrt(C*N/T) = 0.2863564213
- sigma = epsilon*T + C*N/epsilon = 2863.5642127

The restart-aware implementation follows Algorithm 2's requirement that a
target-strategy switch starts a new DChasing episode from BIRD's current
inventory state. No-purchase actions are implemented semantically (zero sale)
rather than as raw price 1, and full-information reward feedback is applied
after every buyer.

The KV-FPL variant is a conservative finite-expert instantiation based on the
Kalai--Vempala additive FPL bound with the BIRD switching-cost upper bound
Delta=sigma. It is useful as a theory-oriented diagnostic; the ICDE paper does
not specify all finite-horizon constants needed to uniquely reproduce one FTPL
parameterization.

## Per-seed revenue

| Method | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | Mean ± Std |
|---|---:|---:|---:|---:|---:|---:|
| Expert | 3,138,783 | 3,168,887 | 3,425,705 | 2,806,764 | 3,146,827 | 3,137,393 ± 219,956 |
| Decision Table | 2,298,480 | 2,212,650 | 2,030,450 | 2,194,810 | 2,168,320 | 2,180,942 ± 97,237 |
| DP | 3,018,870 | 3,033,570 | 3,038,430 | 3,027,435 | 3,022,925 | 3,028,246 ± 7,889 |
| DDPG | 2,550,139 | 2,478,733 | 2,445,665 | 2,473,733 | 2,479,751 | 2,485,604 ± 38,673 |
| BIRD-CurrentClean | 2,502,529 | 2,500,348 | 2,586,312 | 2,406,817 | 2,354,410 | 2,470,083 ± 90,638 |
| BIRD-DChasing-OracleFixed | 3,023,090 | 3,139,135 | 3,166,029 | 3,027,080 | 3,056,967 | 3,082,460 ± 66,026 |
| BIRD-DChasing-FTL | 2,750,705 | 2,813,612 | 3,050,848 | 2,667,763 | 2,787,289 | 2,814,043 ± 143,355 |
| BIRD-DChasing-KVFPL | 3,026,610 | 3,034,505 | 1,384,450 | 3,027,080 | 3,023,470 | 2,699,223 ± 734,992 |

## Revenue loss versus each seed's best fixed strategy

| Method | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | Mean ± Std |
|---|---:|---:|---:|---:|---:|---:|
| BIRD-CurrentClean | 20.27% | 21.10% | 24.50% | 20.50% | 25.18% | 22.31% ± 2.34% |
| BIRD-DChasing-OracleFixed | 3.69% | 0.94% | 7.58% | 0.01% | 2.86% | 3.01% ± 2.94% |
| BIRD-DChasing-FTL | 12.36% | 11.21% | 10.94% | 11.88% | 11.43% | 11.56% ± 0.56% |
| BIRD-DChasing-KVFPL | 3.57% | 4.24% | 59.59% | 0.01% | 3.92% | 14.27% ± 25.39% |

## Selector diagnostics

- FTL switches: 6, 10, 12, 10, 6 (mean 8.8).
- KV-FPL switches: 0 for all five seeds.
- KV-FPL selected DP for seeds 0,1,3,4; it selected Decision Table for all
  5000 buyers in seed 2. That one initialization explains the 59.59% outlier.
- DChasing missing-step count was zero in all restart-aware runs; the observed
  loss is therefore dominated by epsilon no-purchase steps and selector choice,
  not by inability to catch the target inventory trajectory.

## Interpretation

1. Correct DChasing is highly effective: with the hindsight best fixed target,
   mean loss falls from 22.31% to 3.01%.
2. Restart-aware FTL reduces the mean loss to 11.56% and is very stable across
   seeds, showing that the remaining practical bottleneck is target selection.
3. The conservative KV-FPL parameterization is unstable at this finite horizon.
   Its theoretical switching penalty is large enough that it never switches;
   four seeds are strong, but one unlucky initial perturbation locks onto the
   Decision Table strategy and causes a 59.59% loss.
4. Therefore the chasing component is experimentally validated. The exact
   finite-horizon OLSC selector remains the part that requires care if one wants
   a strict theorem-to-code reproduction rather than a practical selector.
