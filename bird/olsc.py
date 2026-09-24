import argparse
import json

import numpy as np

from .actor import TorchActor
from .strategies import (
    BASE_NAMES,
    generate_base_feedback,
    make_context,
    make_strategy,
    active,
    sell_units,
)
from .selectors import (
    FLLSelector,
    FLLStarSelector,
    FPLStarSelector,
    FTLSelector,
    OracleFixedSelector,
    UniformFPLSelector,
    theoretical_selector_epsilon,
    theoretical_star_epsilon,
)


def make_selector(name, rewards, ctx, multiplier, seed):
    k = rewards.shape[1]
    best_local = int(np.argmax(rewards.sum(axis=0)))
    eps_sel = theoretical_selector_epsilon(
        k, ctx.T, ctx.sigma, multiplier=multiplier
    )

    if name == "ftl":
        return FTLSelector(k), None
    if name == "fpl":
        return UniformFPLSelector(k, eps_sel, seed + 404_003), eps_sel
    if name == "fll":
        return FLLSelector(k, eps_sel, seed + 505_003), eps_sel
    if name == "fpl-star":
        eps_star = theoretical_star_epsilon(
            k, ctx.T, ctx.sigma, multiplier=multiplier
        )
        return FPLStarSelector(k, eps_star, seed + 606_003), eps_star
    if name == "fll-star":
        eps_star = theoretical_star_epsilon(
            k, ctx.T, ctx.sigma, multiplier=multiplier
        )
        return FLLStarSelector(k, eps_star, seed + 707_003), eps_star
    if name == "oracle-fixed":
        return OracleFixedSelector(best_local, k), None
    raise ValueError(name)


def run_bird(
    actor,
    seed=0,
    selector_name="fll",
    selector_multiplier=1.0,
    T=5000,
    inventory=200,
    ctx=None,
    trajectories=None,
    rewards=None,
):
    if ctx is None:
        ctx = make_context(seed=seed, T=T, inventory=inventory)
    if trajectories is None or rewards is None:
        trajectories, rewards = generate_base_feedback(ctx, actor)

    selector, eps_sel = make_selector(
        selector_name,
        rewards,
        ctx,
        selector_multiplier,
        seed,
    )

    base_revenue = {
        name: trajectories[name]["revenue"] for name in BASE_NAMES
    }
    best_name = max(base_revenue, key=base_revenue.get)
    best_revenue = float(base_revenue[best_name])

    bird_inventory = ctx.initial_inventory.copy()
    bird_last_price = ctx.base_price.copy()
    bird_profit = np.zeros((ctx.T, ctx.N_total), dtype=float)
    bird_sold_out_index = np.full(ctx.N_total, np.nan, dtype=float)

    target_name = None
    target_strategy = None
    restarts = 0
    missing_steps = 0
    target_counts = {name: 0 for name in BASE_NAMES}
    max_chasing_distance = 0.0

    for t in range(ctx.T):
        local = selector.select()
        name = BASE_NAMES[local]
        target_counts[name] += 1

        if name != target_name:
            target_name = name
            restarts += 1
            target_strategy = make_strategy(name, ctx, actor)
            # Algorithm 2: restart DChasing at BIRD's current state.
            # Experimental pricing policies have internal state in addition to
            # inventory, so initialize their price component from BIRD's latest
            # effective non-no-sale prices and all other episode statistics from
            # the current inventory/time.
            target_strategy.reset(
                t,
                bird_inventory.copy(),
                bird_last_price.copy(),
            )

        target_before = target_strategy.inventory.copy()
        max_chasing_distance = max(
            max_chasing_distance,
            float(np.maximum(target_before - bird_inventory, 0.0).sum()),
        )
        target_price = target_strategy.prices(t)

        q_target = np.zeros(ctx.N_total, dtype=float)
        q_bird = np.zeros(ctx.N_total, dtype=float)

        epsilon_nosell = bool(ctx.epsilon_mask[t])

        for j in range(ctx.N_total):
            if not active(ctx, t, j):
                continue

            q_target[j] = sell_units(
                ctx,
                t,
                j,
                target_price[j],
                target_strategy.inventory[j],
            )

            if bird_inventory[j] < ctx.m or epsilon_nosell:
                continue

            if bird_inventory[j] >= target_before[j]:
                q_bird[j] = sell_units(
                    ctx,
                    t,
                    j,
                    target_price[j],
                    bird_inventory[j],
                )
                bird_inventory[j] -= q_bird[j]
                bird_profit[t, j] = q_bird[j] * target_price[j]
                # This is the effective target price BIRD actually follows.
                bird_last_price[j] = target_price[j]
            else:
                missing_steps += 1

        target_strategy.update(t, q_target, target_price)
        for j in range(ctx.N_total):
            if np.isnan(bird_sold_out_index[j]) and bird_inventory[j] < ctx.m:
                bird_sold_out_index[j] = float(t)

        # Full-information feedback is revealed after every buyer, even on an
        # epsilon no-purchase step.
        selector.update(rewards[t])

    revenue = float(bird_profit.sum())
    loss_pct = (best_revenue - revenue) / best_revenue * 100.0

    out = {
        "seed": int(seed),
        "selector": selector_name,
        "selector_multiplier": float(selector_multiplier),
        "selector_epsilon": eps_sel,
        "T": int(ctx.T),
        "C": float(ctx.initial_inventory.max()),
        "N_active": int(ctx.N_active),
        "dchasing_epsilon": float(ctx.epsilon),
        "dchasing_epsilon_theory": float(ctx.epsilon_theory),
        "sigma_delta": float(ctx.sigma),
        "reward_bound": float(ctx.reward_bound),
        "base_revenue": base_revenue,
        "best_fixed": best_name,
        "best_fixed_revenue": best_revenue,
        "BIRD": revenue,
        "BIRD_category_revenue": [float(x) for x in bird_profit.sum(axis=0)],
        "BIRD_sold_out_index": [
            None if np.isnan(x) else float(x) for x in bird_sold_out_index
        ],
        "revenue_loss_pct": float(loss_pct),
        "switches": int(selector.switches),
        "restarts": int(restarts),
        "missing_steps": int(missing_steps),
        "epsilon_nosell_steps": int(ctx.epsilon_mask.sum()),
        "max_chasing_distance": float(max_chasing_distance),
        "target_counts": target_counts,
        "remaining_inventory": [float(x) for x in bird_inventory],
    }
    if hasattr(selector, "grid_updates"):
        out["grid_updates"] = int(selector.grid_updates)
    if hasattr(selector, "lazy_keeps"):
        out["lazy_keeps"] = int(selector.lazy_keeps)
        out["lazy_resets"] = int(selector.lazy_resets)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor", default="artifacts/checkpoints/ddpg_actor_critic.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--selector",
        choices=["ftl", "fpl", "fll", "fpl-star", "fll-star", "oracle-fixed"],
        default="fll",
    )
    ap.add_argument("--selector-multiplier", type=float, default=1.0)
    ap.add_argument("--T", type=int, default=5000)
    ap.add_argument("--inventory", type=int, default=200)
    args = ap.parse_args()

    actor = TorchActor(args.actor)
    print(
        json.dumps(
            run_bird(
                actor,
                seed=args.seed,
                selector_name=args.selector,
                selector_multiplier=args.selector_multiplier,
                T=args.T,
                inventory=args.inventory,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
