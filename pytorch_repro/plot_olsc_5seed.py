import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MULTIPLIERS = [1, 2, 4, 8, 16, 32, 64]


def save(fig, out, stem):
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="artifacts/olsc_5seed/per_seed.csv")
    ap.add_argument("--output", default="artifacts/olsc_5seed/figures")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for family in ["FPL", "FLL"]:
        means = []
        stds = []
        for m in MULTIPLIERS:
            x = df[df["method"] == f"{family}-{m}x"]["loss_pct"]
            means.append(x.mean())
            stds.append(x.std(ddof=1))
        ax.errorbar(MULTIPLIERS, means, yerr=stds, marker="o", capsize=3, label=family)
    ftl = df[df["method"] == "FTL"]["loss_pct"].mean()
    oracle = df[df["method"] == "OracleFixed"]["loss_pct"].mean()
    ax.axhline(ftl, linewidth=1, linestyle="--", label="FTL mean")
    ax.axhline(oracle, linewidth=1, linestyle=":", label="OracleFixed mean")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Perturbation scale multiplier")
    ax.set_ylabel("Revenue loss vs best fixed strategy (%)")
    ax.set_title("FPL/FLL finite-horizon sensitivity")
    ax.legend()
    save(fig, out, "loss_vs_multiplier")

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for family in ["FPL", "FLL"]:
        means = []
        stds = []
        for m in MULTIPLIERS:
            x = df[df["method"] == f"{family}-{m}x"]["switches"]
            means.append(x.mean())
            stds.append(x.std(ddof=1))
        ax.errorbar(MULTIPLIERS, means, yerr=stds, marker="o", capsize=3, label=family)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Perturbation scale multiplier")
    ax.set_ylabel("Number of target switches")
    ax.set_title("Selector stability under perturbation scaling")
    ax.legend()
    save(fig, out, "switches_vs_multiplier")

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for family in ["FPLstar", "FLLstar"]:
        means = []
        stds = []
        for m in MULTIPLIERS:
            x = df[df["method"] == f"{family}-{m}x"]["loss_pct"]
            means.append(x.mean())
            stds.append(x.std(ddof=1))
        ax.errorbar(MULTIPLIERS, means, yerr=stds, marker="o", capsize=3, label=family)
    ftl = df[df["method"] == "FTL"]["loss_pct"].mean()
    oracle = df[df["method"] == "OracleFixed"]["loss_pct"].mean()
    ax.axhline(ftl, linewidth=1, linestyle="--", label="FTL mean")
    ax.axhline(oracle, linewidth=1, linestyle=":", label="OracleFixed mean")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Perturbation scale multiplier")
    ax.set_ylabel("Revenue loss vs best fixed strategy (%)")
    ax.set_title("Expert FPL*/FLL* finite-horizon sensitivity")
    ax.legend()
    save(fig, out, "loss_vs_multiplier_star")

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for family in ["FPLstar", "FLLstar"]:
        means = []
        stds = []
        for m in MULTIPLIERS:
            x = df[df["method"] == f"{family}-{m}x"]["switches"]
            means.append(x.mean())
            stds.append(x.std(ddof=1))
        ax.errorbar(MULTIPLIERS, means, yerr=stds, marker="o", capsize=3, label=family)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Perturbation scale multiplier")
    ax.set_ylabel("Number of target switches")
    ax.set_title("Expert selector stability under perturbation scaling")
    ax.legend()
    save(fig, out, "switches_vs_multiplier_star")

    shortlist = ["Expert","DP","DDPG","FTL","OracleFixed"]
    # Add the empirically lowest-mean FPL and FLL only for the compact summary plot.
    for family in ["FPL","FLL","FPLstar","FLLstar"]:
        candidates = []
        for m in MULTIPLIERS:
            method = f"{family}-{m}x"
            candidates.append((df[df["method"] == method]["loss_pct"].mean(), method))
        shortlist.append(min(candidates)[1])

    stats = (
        df[df["method"].isin(shortlist)]
        .groupby("method")["loss_pct"]
        .agg(["mean","std"])
        .reindex(shortlist)
    )
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(stats.index, stats["mean"], yerr=stats["std"], capsize=4)
    ax.axhline(0, linewidth=1)
    ax.set_ylabel("Revenue loss vs best fixed strategy (%)")
    ax.set_title("Restartable BIRD: five-seed paired comparison")
    ax.tick_params(axis="x", rotation=25)
    save(fig, out, "paired_summary")

    print("generated", len(list(out.glob("*.png"))), "png figures")


if __name__ == "__main__":
    main()
