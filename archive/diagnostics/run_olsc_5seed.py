import argparse
import csv
import json
from pathlib import Path

import numpy as np

from bird_olsc import run_bird
from chasing_pytorch import TorchActor
from restartable_strategies import (
    BASE_NAMES,
    generate_base_feedback,
    make_context,
)


MULTIPLIERS = [1, 2, 4, 8, 16, 32, 64]


def stats(xs):
    a = np.asarray(xs, dtype=float)
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "min": float(a.min()),
        "max": float(a.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor", default="artifacts/ddpg_actor_critic.pt")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--output-dir", default="artifacts/olsc_5seed")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    actor = TorchActor(args.actor)

    rows = []
    details = {}
    base_rows = []

    for seed in range(args.seeds):
        ctx = make_context(seed=seed, T=5000, inventory=200)
        trajectories, rewards = generate_base_feedback(ctx, actor)

        details[str(seed)] = {}
        best_fixed = max(
            BASE_NAMES, key=lambda n: trajectories[n]["revenue"]
        )
        best_revenue = trajectories[best_fixed]["revenue"]

        for name in BASE_NAMES:
            base_rows.append({
                "seed": seed,
                "method": name,
                "revenue": trajectories[name]["revenue"],
                "loss_pct": (
                    (best_revenue - trajectories[name]["revenue"])
                    / best_revenue
                    * 100.0
                ),
                "switches": 0,
                "restarts": 0,
                "missing_steps": 0,
                "selector_multiplier": "",
                "selector_epsilon": "",
                "best_fixed": best_fixed,
            })

        configs = [
            ("OracleFixed", "oracle-fixed", 1.0),
            ("FTL", "ftl", 1.0),
        ]
        for m in MULTIPLIERS:
            configs.append((f"FPL-{m}x", "fpl", float(m)))
            configs.append((f"FLL-{m}x", "fll", float(m)))
            configs.append((f"FPLstar-{m}x", "fpl-star", float(m)))
            configs.append((f"FLLstar-{m}x", "fll-star", float(m)))

        for label, selector, multiplier in configs:
            r = run_bird(
                actor,
                seed=seed,
                selector_name=selector,
                selector_multiplier=multiplier,
                T=5000,
                inventory=200,
                ctx=ctx,
                trajectories=trajectories,
                rewards=rewards,
            )
            details[str(seed)][label] = r
            rows.append({
                "seed": seed,
                "method": label,
                "revenue": r["BIRD"],
                "loss_pct": r["revenue_loss_pct"],
                "switches": r["switches"],
                "restarts": r["restarts"],
                "missing_steps": r["missing_steps"],
                "selector_multiplier": multiplier if selector in ("fpl","fll","fpl-star","fll-star") else "",
                "selector_epsilon": r["selector_epsilon"] if r["selector_epsilon"] is not None else "",
                "best_fixed": r["best_fixed"],
            })

    all_rows = base_rows + rows
    fields = [
        "seed","method","revenue","loss_pct","switches","restarts",
        "missing_steps","selector_multiplier","selector_epsilon","best_fixed"
    ]
    with (out / "per_seed.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    methods = []
    for r in all_rows:
        if r["method"] not in methods:
            methods.append(r["method"])

    summary = {}
    for method in methods:
        rr = [r for r in all_rows if r["method"] == method]
        summary[method] = {
            "revenue": stats([r["revenue"] for r in rr]),
            "loss_pct": stats([r["loss_pct"] for r in rr]),
            "switches": stats([r["switches"] for r in rr]),
            "restarts": stats([r["restarts"] for r in rr]),
            "missing_steps": stats([r["missing_steps"] for r in rr]),
            "per_seed_revenue": {
                str(r["seed"]): float(r["revenue"]) for r in rr
            },
            "per_seed_loss_pct": {
                str(r["seed"]): float(r["loss_pct"]) for r in rr
            },
        }

    # Paired improvement relative to FTL and Oracle for each seed.
    paired = {}
    for method in methods:
        if method in BASE_NAMES:
            continue
        rr = {r["seed"]: r for r in all_rows if r["method"] == method}
        ftl = {r["seed"]: r for r in all_rows if r["method"] == "FTL"}
        oracle = {r["seed"]: r for r in all_rows if r["method"] == "OracleFixed"}
        paired[method] = {
            "loss_minus_ftl_pct_points": stats([
                rr[s]["loss_pct"] - ftl[s]["loss_pct"] for s in rr
            ]) if method != "FTL" else stats([0.0] * args.seeds),
            "loss_minus_oracle_pct_points": stats([
                rr[s]["loss_pct"] - oracle[s]["loss_pct"] for s in rr
            ]) if method != "OracleFixed" else stats([0.0] * args.seeds),
        }

    meta = {
        "seeds": list(range(args.seeds)),
        "T": 5000,
        "inventory_argument": 200,
        "multipliers": MULTIPLIERS,
        "paired_randomness": [
            "buyer valuations",
            "expert noise",
            "DChasing epsilon mask",
            "base-strategy full-information rewards",
        ],
    }

    payload = {"meta": meta, "summary": summary, "paired": paired}
    (out / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out / "detail.json").write_text(
        json.dumps(details, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
