import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ORDER = [
    "Current-Clean",
    "Theory-FTL",
    "Theory-FTPL-0.5x",
    "Theory-FTPL-1x",
    "Theory-FTPL-2x",
    "Theory-FTPL-eps0",
]


def save(fig, out, stem):
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="artifacts/theory_validation/per_seed.csv")
    ap.add_argument("--output", default="artifacts/theory_validation/figures")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    stats = df.groupby("variant")["loss_pct"].agg(["mean", "std"]).reindex(ORDER)
    fig, ax = plt.subplots(figsize=(9.2, 4.7))
    ax.bar(stats.index, stats["mean"], yerr=stats["std"], capsize=4)
    ax.axhline(0, linewidth=1)
    ax.set_ylabel("Revenue loss vs best fixed strategy (%)")
    ax.set_title("Theory-aligned BIRD validation")
    ax.tick_params(axis="x", rotation=25)
    save(fig, out, "theory_loss_comparison")

    stats_r = df.groupby("variant")["bird_revenue"].agg(["mean", "std"]).reindex(ORDER)
    fig, ax = plt.subplots(figsize=(9.2, 4.7))
    ax.bar(stats_r.index, stats_r["mean"], yerr=stats_r["std"], capsize=4)
    ax.set_ylabel("BIRD revenue")
    ax.set_title("Revenue after theory-consistent implementation fixes")
    ax.tick_params(axis="x", rotation=25)
    save(fig, out, "theory_revenue_comparison")

    ftpl = df[df["variant"] == "Theory-FTPL-1x"].sort_values("seed")
    current = df[df["variant"] == "Current-Clean"].sort_values("seed")
    merged = current[["seed", "loss_pct"]].merge(
        ftpl[["seed", "loss_pct"]], on="seed", suffixes=("_current", "_theory")
    )
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.plot(merged["seed"], merged["loss_pct_current"], marker="o", label="Current-Clean")
    ax.plot(merged["seed"], merged["loss_pct_theory"], marker="o", label="Theory-FTPL-1x")
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Revenue loss (%)")
    ax.set_title("Per-seed effect of theory-consistent fixes")
    ax.legend()
    save(fig, out, "theory_per_seed_loss")

    print("generated", len(list(out.glob("*.png"))), "png figures")


if __name__ == "__main__":
    main()
