import argparse
import csv
import json
from pathlib import Path

import numpy as np

from chasing_pytorch import TorchActor, run_simulation
from chasing_theory_v2 import run_bird_restart


BASE = ["Expert", "Decision Table", "DP", "DDPG"]


def mean_std(xs):
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
    ap.add_argument("--output-dir", default="artifacts/theory_v2_5seed")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    actor = TorchActor(args.actor)

    rows = []
    detail = {}
    variants = [
        ("BIRD-CurrentClean", "current"),
        ("BIRD-DChasing-OracleFixed", "oracle-fixed"),
        ("BIRD-DChasing-FTL", "ftl"),
        ("BIRD-DChasing-KVFPL", "kv-fpl"),
    ]

    for seed in range(args.seeds):
        current = run_simulation(
            actor, seed=seed, T=5000, inventory=200
        )
        oracle = run_bird_restart(
            actor, seed=seed, T=5000, inventory=200, selector="oracle-fixed"
        )
        ftl = run_bird_restart(
            actor, seed=seed, T=5000, inventory=200, selector="ftl"
        )
        kv = run_bird_restart(
            actor, seed=seed, T=5000, inventory=200, selector="kv-fpl"
        )
        detail[str(seed)] = {
            "current": current,
            "oracle_fixed": oracle,
            "ftl": ftl,
            "kv_fpl": kv,
        }

        best_fixed = oracle["best_fixed_revenue"]
        best_name = oracle["best_fixed"]

        # Base strategies.
        for strategy in BASE:
            revenue = oracle["base_revenue"][strategy]
            rows.append({
                "seed": seed,
                "method": strategy,
                "type": "base",
                "revenue": revenue,
                "best_fixed": best_name,
                "best_fixed_revenue": best_fixed,
                "loss_pct_vs_best_fixed": (best_fixed - revenue) / best_fixed * 100.0,
                "switches": 0,
                "epsilon_nosell_steps": 0,
                "missing_steps": 0,
            })

        bird_runs = {
            "BIRD-CurrentClean": {
                "revenue": current["revenue"]["BIRD"],
                "loss": current["revenue_loss_pct"],
                "switches": None,
                "epsilon_nosell_steps": None,
                "missing_steps": None,
            },
            "BIRD-DChasing-OracleFixed": {
                "revenue": oracle["BIRD"],
                "loss": oracle["revenue_loss_pct"],
                "switches": oracle["switches"],
                "epsilon_nosell_steps": oracle["epsilon_nosell_steps"],
                "missing_steps": oracle["missing_steps"],
            },
            "BIRD-DChasing-FTL": {
                "revenue": ftl["BIRD"],
                "loss": ftl["revenue_loss_pct"],
                "switches": ftl["switches"],
                "epsilon_nosell_steps": ftl["epsilon_nosell_steps"],
                "missing_steps": ftl["missing_steps"],
            },
            "BIRD-DChasing-KVFPL": {
                "revenue": kv["BIRD"],
                "loss": kv["revenue_loss_pct"],
                "switches": kv["switches"],
                "epsilon_nosell_steps": kv["epsilon_nosell_steps"],
                "missing_steps": kv["missing_steps"],
            },
        }
        for method, d in bird_runs.items():
            rows.append({
                "seed": seed,
                "method": method,
                "type": "bird",
                "revenue": d["revenue"],
                "best_fixed": best_name,
                "best_fixed_revenue": best_fixed,
                "loss_pct_vs_best_fixed": d["loss"],
                "switches": "" if d["switches"] is None else d["switches"],
                "epsilon_nosell_steps": "" if d["epsilon_nosell_steps"] is None else d["epsilon_nosell_steps"],
                "missing_steps": "" if d["missing_steps"] is None else d["missing_steps"],
            })

    fields = [
        "seed","method","type","revenue","best_fixed","best_fixed_revenue",
        "loss_pct_vs_best_fixed","switches","epsilon_nosell_steps","missing_steps"
    ]
    with (out / "per_seed.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    methods = [*BASE, *[x[0] for x in variants]]
    summary = {}
    for method in methods:
        rr = [r for r in rows if r["method"] == method]
        revenue = [float(r["revenue"]) for r in rr]
        loss = [float(r["loss_pct_vs_best_fixed"]) for r in rr]
        summary[method] = {
            "revenue": mean_std(revenue),
            "loss_pct_vs_best_fixed": mean_std(loss),
            "per_seed_revenue": {str(r["seed"]): float(r["revenue"]) for r in rr},
            "per_seed_loss_pct": {str(r["seed"]): float(r["loss_pct_vs_best_fixed"]) for r in rr},
        }
        sw = [float(r["switches"]) for r in rr if r["switches"] != ""]
        if sw:
            summary[method]["switches"] = mean_std(sw)
        ns = [float(r["epsilon_nosell_steps"]) for r in rr if r["epsilon_nosell_steps"] != ""]
        if ns:
            summary[method]["epsilon_nosell_steps"] = mean_std(ns)
        ms = [float(r["missing_steps"]) for r in rr if r["missing_steps"] != ""]
        if ms:
            summary[method]["missing_steps"] = mean_std(ms)

    theory_meta = {
        "seeds": list(range(args.seeds)),
        "T": 5000,
        "inventory_argument": 200,
        "effective_initial_inventory": oracle["C"],
        "N_active": oracle["N_active"],
        "epsilon_theory": oracle["epsilon_theory"],
        "sigma": oracle["sigma"],
        "delta": oracle["delta"],
        "kv_fpl_eta": kv["selector_eta"],
        "reward_bound": oracle["reward_bound"],
    }

    payload = {
        "meta": theory_meta,
        "summary": summary,
    }
    (out / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8"
    )
    (out / "detail.json").write_text(
        json.dumps(detail, indent=2, sort_keys=True),
        encoding="utf-8"
    )

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
