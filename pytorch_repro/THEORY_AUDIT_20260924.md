# BIRD theory audit — 2026-09-24

This note separates theory-preserving implementation corrections from changes
that would define a different algorithm.

## Comparator and information model

BIRD compares against the best *fixed* pricing strategy in hindsight. Algorithm
2 assumes full information after each buyer: every base strategy's reward for
the buyer is simulated and fed back to the online selector.

## DChasing requirements

Algorithm 1 has two kinds of deliberate no-purchase steps:

1. the epsilon random no-purchase step;
2. a no-purchase step when BIRD's inventory is below the target strategy's
   inventory (or below the per-buyer purchase limit).

The paper uses normalized prices/valuations, so an all-1 price vector is a
no-purchase action. In the repository's raw price scale (~2000--4000), the
literal value 1 must therefore be implemented semantically as "sell zero",
not as a raw price of 1.

For the independent-goods + K_j case, Theorem 3 uses

    epsilon = sqrt(C N / |B|)

where N is the maximum number of container categories sold simultaneously,
not the total number of categories. In the repository synthetic schedule,
at most two of the five categories overlap, hence N=2 and with C=205,
|B|=5000 the theoretical epsilon is 0.286356..., matching the paper.

## Theory/implementation mismatches found

### 1. Time-indexed target inventory was missing

The legacy code stores only one final inventory vector per base strategy and
later compares BIRD's current inventory against that final vector. Algorithm 1
requires the target strategy's inventory at the current buyer.

### 2. Full-information update was skipped on epsilon steps

The modern reproduction previously used `continue` after an epsilon
no-purchase step. Algorithm 2 requires feeding the reward vector of *all*
strategies to the selector after every buyer, including no-purchase steps.

### 3. Raw price 1 is not a no-purchase action

Under raw experimental valuations, posting price 1 causes maximum purchase.
The theory's all-1 normalized action must be represented as zero sale.

### 4. Legacy selector is not the theoretical FTPL selector

The legacy score

    cumulative_profit + price * m / sqrt(D / (R^2 T))

contains no random perturbation and is not the Following-The-Perturbed-Leader
algorithm used in the proof.

### 5. Target switches must restart DChasing

Algorithm 2 states that when gamma_j != gamma_{j-1}, DChasing is invoked from
scratch with the new target strategy initialized at BIRD's current state.
The first theory-validation implementation still referenced each target
strategy's global trajectory from the original initial inventory. Therefore
results with target switching (notably the FTL diagnostic) are not yet a
formal validation of Algorithm 2.

## What the completed diagnostics establish

A no-switch seed is useful because item 5 is irrelevant there. For seed 0,
the theory-preserving DChasing corrections with theoretical epsilon changed:

- Current-Clean BIRD: 2,502,528.50, loss 20.27%
- corrected no-switch chasing: 3,059,829.02, loss 2.52%
- best fixed strategy (Expert): 3,138,782.56

This is strong evidence that the chasing/state/no-purchase bugs materially
caused the earlier performance gap.

Across 20 seeds:

- Current-Clean mean loss: 18.51%
- FTL diagnostic with corrected no-purchase/full-information mechanics:
  mean loss 8.01%
- the first generic FTPL parameterization performed poorly and often made
  zero switches (mean loss roughly 23--25%).

The FTL number is diagnostic only because the current multi-switch simulator
does not yet implement the required restart semantics, and FTL itself is not
the FTPL selector used in the proof.

## Selector parameterization

Kalai--Vempala FPL/FLL perturbs cumulative objective vectors. FLL correlates
the perturbations across periods so that it has the same per-period expected
behavior as FPL while changing decisions rarely. BIRD's OLSC reduction sets
the switching-cost upper bound Delta to the chasing-regret upper bound sigma.
The first generic FTPL experiment used a learning-rate rule that ignored
Delta/sigma and therefore should not be treated as the paper's theoretical
selector.

## Status

- DChasing no-purchase semantics: aligned.
- Theorem-3 epsilon and N definition: aligned.
- full-information selector feedback: aligned.
- current-buyer target inventory tracking: aligned for a fixed target.
- restart on target switch: **not yet fully aligned**.
- exact OLSC FTPL/FLL parameterization with Delta=sigma: **not yet fully
  aligned**.

Therefore the current branch contains useful theory diagnostics, but it should
not yet be labeled a complete theoretical reproduction of BIRD.
