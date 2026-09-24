import argparse
import csv
import json
from pathlib import Path

import numpy as np

from chasing_pytorch import TorchActor, run_simulation
from chasing_theory import generate_base_trajectories, run_theory_bird


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


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
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--output-dir", default="artifacts/theory_validation")
    args = ap.parse_args()

    actor = TorchActor(args.actor)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    configs = [
        ("Theory-FTPL-0.5x", "ftpl", 0.5, None),
        ("Theory-FTPL-1x", "ftpl", 1.0, None),
        ("Theory-FTPL-2x", "ftpl", 2.0, None),
        ("Theory-FTL", "ftl", 1.0, None),
        ("Theory-FTPL-eps0", "ftpl", 1.0, 0.0),
    ]

    for seed in range(args.seeds):
        base = generate_base_trajectories(actor, seed=seed, T=5000, inventory=200)
        legacy = run_simulation(actor, seed=seed, T=5000, inventory=200)
        rows.append({
            "seed": seed,
            "variant": "Current-Clean",
            "bird_revenue": legacy["revenue"]["BIRD"],
            "best_fixed_revenue": max(
                legacy["revenue"]["Expert"],
                legacy["revenue"]["Decision Table"],
                legacy["revenue"]["DP"],
                legacy["revenue"]["DDPG"],
            ),
            "loss_pct": legacy["revenue_loss_pct"],
            "fixed_best": legacy["fixed_best"],
            "epsilon": legacy["epsilon"],
            "switches": "",
            "random_nosell_steps": "",
            "mismatch_nosell_category_steps": "",
            "max_chasing_distance": "",
        })

        for name, selector, eta_mult, eps in configs:
            r = run_theory_bird(
                actor,
                seed=seed,
                T=5000,
                inventory=200,
                selector=selector,
                eta_multiplier=eta_mult,
                epsilon_override=eps,
                base=base,
            )
            rows.append({
                "seed": seed,
                "variant": name,
                "bird_revenue": r["BIRD"],
                "best_fixed_revenue": r["best_fixed_revenue"],
                "loss_pct": r["revenue_loss_pct"],
                "fixed_best": r["fixed_best"],
                "epsilon": r["epsilon"],
                "switches": r["switches"],
                "random_nosell_steps": r["random_nosell_steps"],
                "mismatch_nosell_category_steps": r["mismatch_nosell_category_steps"],
                "max_chasing_distance": r["max_chasing_distance"],
            })

    fields = [
        "seed",
        "variant",
        "bird_revenue",
        "best_fixed_revenue",
        "loss_pct",
        "fixed_best",
        "epsilon",
        "switches",
        "random_nosell_steps",
        "mismatch_nosell_category_steps",
        "max_chasing_distance",
    ]
    write_csv(out / "per_seed.csv", rows, fields)

    summary = {}
    for variant in sorted(set(r["variant"] for r in rows)):
        vr = [r for r in rows if r["variant"] == variant]
        summary[variant] = {
            "bird_revenue": stats([r["bird_revenue"] for r in vr]),
            "loss_pct": stats([r["loss_pct"] for r in vr]),
        }
        sw = [float(r["switches"]) for r in vr if r["switches"] != ""]
        if sw:
            summary[variant]["switches"] = stats(sw)
            summary[variant]["random_nosell_steps"] = stats(
                [float(r["random_nosell_steps"]) for r in vr]
            )
            summary[variant]["mismatch_nosell_category_steps"] = stats(
                [float(r["mismatch_nosell_category_steps"]) for r in vr]
            )
            summary[variant]["max_chasing_distance"] = stats(
                [float(r["max_chasing_distance"]) for r in vr]
            )

    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
