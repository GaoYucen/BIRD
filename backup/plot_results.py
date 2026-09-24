import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ORDER = ["Expert", "Decision Table", "DP", "DDPG", "BIRD"]


def save(fig, out, stem):
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--output", default="figures")
    args = ap.parse_args()

    res = Path(args.results)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    main_df = pd.read_csv(res / "main_seeds.csv")
    stats = (
        main_df.groupby("strategy")["revenue"]
        .agg(["mean", "std"])
        .reindex(ORDER)
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.bar(stats.index, stats["mean"], yerr=stats["std"], capsize=4)
    ax.set_ylabel("Revenue")
    ax.set_title("BIRD PyTorch reproduction: mean revenue across seeds")
    ax.tick_params(axis="x", rotation=18)
    save(fig, out, "fig_main_revenue")

    loss_df = pd.read_csv(res / "main_loss.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(loss_df["seed"], loss_df["revenue_loss_pct"], marker="o")
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Seed")
    ax.set_ylabel("BIRD revenue loss vs best fixed strategy (%)")
    ax.set_title("Per-seed BIRD gap to the best fixed strategy")
    save(fig, out, "fig_revenue_loss_by_seed")

    eps = pd.read_csv(res / "epsilon_sensitivity.csv")
    eps_stats = (
        eps.groupby(["epsilon", "strategy"])["revenue"]
        .mean()
        .unstack("strategy")
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for s in ORDER:
        if s in eps_stats:
            ax.plot(eps_stats.index, eps_stats[s], marker="o", label=s)
    ax.set_xlabel("epsilon")
    ax.set_ylabel("Mean revenue")
    ax.set_title("Sensitivity to BIRD exploration epsilon")
    ax.legend(ncol=2)
    save(fig, out, "fig_epsilon_sensitivity")

    horizon = pd.read_csv(res / "horizon_sensitivity.csv")
    h_stats = (
        horizon.groupby(["T", "strategy"])["revenue"]
        .mean()
        .unstack("strategy")
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for s in ORDER:
        if s in h_stats:
            ax.plot(h_stats.index, h_stats[s], marker="o", label=s)
    ax.set_xlabel("|B| / horizon T")
    ax.set_ylabel("Mean revenue")
    ax.set_title("Sensitivity to buyer-sequence length")
    ax.legend(ncol=2)
    save(fig, out, "fig_horizon_sensitivity")

    rep = pd.read_csv(res / "representative_sequences.csv")
    for winner in rep["winner"].unique():
        sub = rep[rep["winner"] == winner].set_index("strategy").reindex(ORDER)
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        bottom = np.zeros(len(sub))
        for i in range(1, 6):
            vals = sub[f"category_{i}"].to_numpy()
            ax.bar(sub.index, vals, bottom=bottom, label=f"Category {i}")
            bottom += vals
        seed = int(sub["seed"].iloc[0])
        ax.set_ylabel("Revenue")
        ax.set_title(f"Representative sequence: {winner} wins (seed={seed})")
        ax.tick_params(axis="x", rotation=18)
        ax.legend(ncol=3, fontsize=8)
        save(fig, out, f"fig_representative_{winner.lower().replace(' ', '_')}")

    print("generated figures:")
    for p in sorted(out.glob("*.png")):
        print(p)


if __name__ == "__main__":
    main()
