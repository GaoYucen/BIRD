import argparse
import csv
import json
from pathlib import Path

import numpy as np

from chasing_pytorch import STRATEGIES, TorchActor, run_simulation


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor", default="ddpg_actor_critic.pt")
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--main-seeds", type=int, default=20)
    ap.add_argument("--sensitivity-seeds", type=int, default=10)
    ap.add_argument("--representative-search", type=int, default=60)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    actor = TorchActor(args.actor)
    cache = {}

    def run(seed, T=5000, inventory=200, epsilon_factor=1.0):
        key = (int(seed), int(T), int(inventory), float(epsilon_factor))
        if key not in cache:
            cache[key] = run_simulation(
                actor,
                seed=seed,
                T=T,
                inventory=inventory,
                epsilon_factor=epsilon_factor,
            )
        return cache[key]

    main_rows = []
    loss_rows = []
    for seed in range(args.main_seeds):
        r = run(seed)
        for s in STRATEGIES:
            main_rows.append({"seed": seed, "strategy": s, "revenue": r["revenue"][s]})
        loss_rows.append({
            "seed": seed,
            "revenue_loss_pct": r["revenue_loss_pct"],
            "fixed_best": r["fixed_best"],
            "bird_revenue": r["revenue"]["BIRD"],
        })
    write_csv(out / "main_seeds.csv", main_rows, ["seed", "strategy", "revenue"])
    write_csv(
        out / "main_loss.csv",
        loss_rows,
        ["seed", "revenue_loss_pct", "fixed_best", "bird_revenue"],
    )

    eps_rows = []
    for factor in [0.25, 0.5, 1.0, 2.0]:
        for seed in range(args.sensitivity_seeds):
            r = run(seed, epsilon_factor=factor)
            for s in STRATEGIES:
                eps_rows.append({
                    "factor": factor,
                    "epsilon": r["epsilon"],
                    "seed": seed,
                    "strategy": s,
                    "revenue": r["revenue"][s],
                })
    write_csv(
        out / "epsilon_sensitivity.csv",
        eps_rows,
        ["factor", "epsilon", "seed", "strategy", "revenue"],
    )

    horizon_rows = []
    for T in [500, 1250, 2500, 5000]:
        for seed in range(args.sensitivity_seeds):
            r = run(seed, T=T)
            for s in STRATEGIES:
                horizon_rows.append({
                    "T": T,
                    "seed": seed,
                    "strategy": s,
                    "revenue": r["revenue"][s],
                    "epsilon": r["epsilon"],
                })
    write_csv(
        out / "horizon_sensitivity.csv",
        horizon_rows,
        ["T", "seed", "strategy", "revenue", "epsilon"],
    )

    reps = {}
    desired = ["DDPG", "Expert", "BIRD"]
    for seed in range(args.representative_search):
        r = run(seed)
        winner = max(STRATEGIES, key=lambda s: r["revenue"][s])
        if winner in desired and winner not in reps:
            reps[winner] = r
        if len(reps) == len(desired):
            break

    rep_rows = []
    for winner, r in reps.items():
        for strategy in STRATEGIES:
            cats = r["category_revenue"][strategy]
            rep_rows.append({
                "winner": winner,
                "seed": r["seed"],
                "strategy": strategy,
                **{f"category_{i+1}": cats[i] for i in range(5)},
                "total": r["revenue"][strategy],
            })
    write_csv(
        out / "representative_sequences.csv",
        rep_rows,
        [
            "winner",
            "seed",
            "strategy",
            "category_1",
            "category_2",
            "category_3",
            "category_4",
            "category_5",
            "total",
        ],
    )
    (out / "representative_sequences.json").write_text(
        json.dumps(reps, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    summary = {}
    for s in STRATEGIES:
        values = [r["revenue"] for r in main_rows if r["strategy"] == s]
        summary[s] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    losses = [r["revenue_loss_pct"] for r in loss_rows]
    summary["BIRD_loss_pct"] = {
        "mean": float(np.mean(losses)),
        "std": float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0,
        "min": float(np.min(losses)),
        "max": float(np.max(losses)),
    }
    summary["representative_seeds"] = {k: int(v["seed"]) for k, v in reps.items()}
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
