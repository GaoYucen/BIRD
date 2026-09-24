import argparse
import json
from pathlib import Path

import numpy as np

from bird_main import MAIN_VERSION, run_main_bird
from chasing_pytorch import TorchActor
from restartable_strategies import BASE_NAMES, generate_base_feedback, make_context


PAPER_LABEL = {
    "Expert": "Experts",
    "Decision Table": "DTP",
    "DP": "DP",
    "DDPG": "DDPG",
    "BIRD": "BIRD",
}


def evaluate(actor, seed, T, effective_C, epsilon_override=None):
    inventory_arg = max(0, int(effective_C) - 5)
    ctx = make_context(
        seed=seed,
        T=int(T),
        inventory=inventory_arg,
        epsilon_override=epsilon_override,
    )
    trajectories, rewards = generate_base_feedback(ctx, actor)
    bird = run_main_bird(
        actor,
        seed=seed,
        T=int(T),
        inventory=inventory_arg,
        ctx=ctx,
        trajectories=trajectories,
        rewards=rewards,
    )

    revenue = {name: float(trajectories[name]["revenue"]) for name in BASE_NAMES}
    revenue["BIRD"] = float(bird["BIRD"])
    category_revenue = {
        name: [float(x) for x in trajectories[name]["category_revenue"]]
        for name in BASE_NAMES
    }
    category_revenue["BIRD"] = list(bird["BIRD_category_revenue"])

    sold_out = {}
    for name in BASE_NAMES:
        sold_out[name] = [
            None if np.isnan(x) else float(x)
            for x in trajectories[name]["sold_out_index"]
        ]
    sold_out["BIRD"] = list(bird["BIRD_sold_out_index"])

    best_fixed = max(BASE_NAMES, key=lambda n: revenue[n])
    best_fixed_revenue = revenue[best_fixed]
    signed_paper_loss_pct = (
        (revenue["BIRD"] - best_fixed_revenue)
        / best_fixed_revenue
        * 100.0
    )
    return {
        "seed": int(seed),
        "T": int(T),
        "C": int(effective_C),
        "epsilon": float(ctx.epsilon),
        "epsilon_theory": float(ctx.epsilon_theory),
        "revenue": revenue,
        "category_revenue": category_revenue,
        "sold_out_index": sold_out,
        "best_fixed": best_fixed,
        "best_fixed_revenue": best_fixed_revenue,
        "paper_signed_loss_pct": float(signed_paper_loss_pct),
        "bird": bird,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor", default="artifacts/ddpg_actor_critic.pt")
    ap.add_argument("--output-dir", default="artifacts/icde_main_repro")
    ap.add_argument("--search-seeds", type=int, default=80)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    actor = TorchActor(args.actor)

    payload = {
        "main_version": MAIN_VERSION,
        "fig7": {},
        "fig8_9_grid": [],
        "fig10": {"epsilon": [], "horizon": []},
        "fig11": None,
    }

    # Fig. 7: search for representative buyer sequences with the same C,T
    # used in the paper. We do not cherry-pick numerical margins; the first
    # seed encountered for each requested winner is stored.
    desired = ["DDPG", "Expert", "BIRD"]
    found = {}
    winner_counts = {name: 0 for name in [*BASE_NAMES, "BIRD"]}
    for seed in range(args.search_seeds):
        r = evaluate(actor, seed=seed, T=5000, effective_C=205)
        winner = max(r["revenue"], key=r["revenue"].get)
        winner_counts[winner] += 1
        if winner in desired and winner not in found:
            found[winner] = r
        if len(found) == len(desired):
            break
    payload["fig7"] = {
        "representatives": found,
        "searched_seeds": int(args.search_seeds),
        "winner_counts_until_stop_or_limit": winner_counts,
        "missing_requested_winners": [x for x in desired if x not in found],
    }

    # Fig. 8/9: exact C and |B| grid visible in the ICDE figure.
    C_values = [51, 102, 153, 204, 307, 409, 512, 615, 717, 819]
    B_values = [2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000, 22500, 25000]
    for T in B_values:
        for C in C_values:
            r = evaluate(actor, seed=0, T=T, effective_C=C)
            payload["fig8_9_grid"].append({
                "T": T,
                "C": C,
                "epsilon": r["epsilon"],
                "paper_signed_loss_pct": r["paper_signed_loss_pct"],
                "best_fixed": r["best_fixed"],
                "revenue": r["revenue"],
            })

    # Fig. 10(a): epsilon sensitivity using the exact labels from ICDE.
    for eps in [0.072, 0.143, 0.286, 0.573, 1.145]:
        r = evaluate(
            actor,
            seed=0,
            T=5000,
            effective_C=205,
            epsilon_override=eps,
        )
        payload["fig10"]["epsilon"].append({
            "requested_epsilon": eps,
            "effective_epsilon": r["epsilon"],
            "revenue": r["revenue"],
            "paper_signed_loss_pct": r["paper_signed_loss_pct"],
        })

    # Fig. 10(b): exact sequence lengths from ICDE.
    for T in [500, 1250, 2500, 5000]:
        r = evaluate(actor, seed=0, T=T, effective_C=205)
        payload["fig10"]["horizon"].append({
            "T": T,
            "epsilon": r["epsilon"],
            "revenue": r["revenue"],
            "paper_signed_loss_pct": r["paper_signed_loss_pct"],
        })

    # Fig. 11: use the same standard C,T sequence as the other diagnostics.
    payload["fig11"] = evaluate(actor, seed=0, T=5000, effective_C=205)

    (out / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({
        "main_version": MAIN_VERSION,
        "fig7_missing": payload["fig7"]["missing_requested_winners"],
        "fig7_winner_counts": payload["fig7"]["winner_counts_until_stop_or_limit"],
        "grid_cells": len(payload["fig8_9_grid"]),
        "epsilon_points": len(payload["fig10"]["epsilon"]),
        "horizon_points": len(payload["fig10"]["horizon"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
