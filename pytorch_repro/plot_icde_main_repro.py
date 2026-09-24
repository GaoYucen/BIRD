import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PAPER_ORDER = ["Expert", "Decision Table", "DP", "DDPG", "BIRD"]
DISPLAY = {
    "Expert": "Experts",
    "Decision Table": "DTP",
    "DP": "DP",
    "DDPG": "DDPG",
    "BIRD": "BIRD",
}


def save(fig, out, stem):
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(out / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def fig7(data, out):
    reps = data["fig7"]["representatives"]
    requested = [x for x in ["DDPG", "Expert", "BIRD"] if x in reps]
    if not requested:
        return
    fig, axes = plt.subplots(1, len(requested), figsize=(5.1 * len(requested), 4.2))
    if len(requested) == 1:
        axes = [axes]
    for ax, winner in zip(axes, requested):
        r = reps[winner]
        x = np.arange(len(PAPER_ORDER))
        bottom = np.zeros(len(PAPER_ORDER))
        for cat in range(5):
            vals = [r["category_revenue"][m][cat] for m in PAPER_ORDER]
            ax.bar(x, vals, bottom=bottom, label=f"Category {cat+1}")
            bottom += np.asarray(vals)
        ax.set_xticks(x, [DISPLAY[m] for m in PAPER_ORDER], rotation=15)
        ax.set_ylabel("Revenue")
        ax.set_title(f"{DISPLAY[winner]} wins (seed={r['seed']})")
        ax.legend(fontsize=7)
    save(fig, out, "fig7_representative_sequences")


def fig8(data, out):
    rows = data["fig8_9_grid"]
    Cs = sorted({r["C"] for r in rows})
    Ts = sorted({r["T"] for r in rows})
    z = np.full((len(Ts), len(Cs)), np.nan)
    for r in rows:
        z[Ts.index(r["T"]), Cs.index(r["C"])] = r["paper_signed_loss_pct"]
    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    im = ax.imshow(z, aspect="auto", origin="lower")
    ax.set_xticks(np.arange(len(Cs)), Cs, rotation=45)
    ax.set_yticks(np.arange(len(Ts)), [f"{t/1000:g}" for t in Ts])
    ax.set_xlabel("Inventory C")
    ax.set_ylabel("|B| (×10³)")
    ax.set_title("Fig. 8 reproduction: signed revenue gap of BIRD (%)")
    fig.colorbar(im, ax=ax, label="(BIRD - best fixed) / best fixed (%)")
    save(fig, out, "fig8_revenue_loss_heatmap")


def fig9(data, out):
    rows = data["fig8_9_grid"]
    Cs = np.asarray([r["C"] for r in rows], dtype=float)
    Ts = np.asarray([r["T"] for r in rows], dtype=float)
    methods = PAPER_ORDER

    fig = plt.figure(figsize=(13.5, 10.0))
    ax1 = fig.add_subplot(221)
    winners = []
    for r in rows:
        winners.append(max(r["revenue"], key=r["revenue"].get))
    for method in methods:
        mask = np.asarray([w == method for w in winners])
        ax1.scatter(Cs[mask], Ts[mask] / 1000.0, label=DISPLAY[method], s=22)
    ax1.set_xlabel("Inventory C")
    ax1.set_ylabel("|B| (×10³)")
    ax1.set_title("(a) Best strategy map")
    ax1.legend(fontsize=7)

    ax2 = fig.add_subplot(222)
    for method in methods:
        vals = np.asarray([r["revenue"][method] for r in rows])
        ax2.scatter(Cs, vals / 1e6, s=15, label=DISPLAY[method])
    ax2.set_xlabel("Inventory C")
    ax2.set_ylabel("Revenue (×10⁶)")
    ax2.set_title("(b) Inventory projection")
    ax2.legend(fontsize=7)

    ax3 = fig.add_subplot(223)
    for method in methods:
        vals = np.asarray([r["revenue"][method] for r in rows])
        ax3.scatter(Ts / 1000.0, vals / 1e6, s=15, label=DISPLAY[method])
    ax3.set_xlabel("|B| (×10³)")
    ax3.set_ylabel("Revenue (×10⁶)")
    ax3.set_title("(c) Buyer-sequence projection")
    ax3.legend(fontsize=7)

    ax4 = fig.add_subplot(224, projection="3d")
    for method in methods:
        vals = np.asarray([r["revenue"][method] for r in rows])
        ax4.scatter(Cs, Ts / 1000.0, vals / 1e6, s=12, label=DISPLAY[method])
    ax4.set_xlabel("Inventory C")
    ax4.set_ylabel("|B| (×10³)")
    ax4.set_zlabel("Revenue (×10⁶)")
    ax4.set_title("(d) 3-D view")
    ax4.legend(fontsize=6)
    save(fig, out, "fig9_grid_views")


def fig10(data, out):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
    eps_rows = data["fig10"]["epsilon"]
    x = np.arange(len(eps_rows))
    width = 0.16
    for idx, method in enumerate(PAPER_ORDER):
        vals = [r["revenue"][method] / 1e6 for r in eps_rows]
        axes[0].bar(x + (idx - 2) * width, vals, width=width, label=DISPLAY[method])
    axes[0].set_xticks(x, [str(r["requested_epsilon"]) for r in eps_rows])
    axes[0].set_xlabel("ε")
    axes[0].set_ylabel("Revenue (×10⁶)")
    axes[0].set_title("(a) Sensitivity to ε")
    axes[0].legend(fontsize=7)

    b_rows = data["fig10"]["horizon"]
    x = np.arange(len(b_rows))
    for idx, method in enumerate(PAPER_ORDER):
        vals = [r["revenue"][method] / 1e6 for r in b_rows]
        axes[1].bar(x + (idx - 2) * width, vals, width=width, label=DISPLAY[method])
    axes[1].set_xticks(x, [str(r["T"]) for r in b_rows])
    axes[1].set_xlabel("|B|")
    axes[1].set_ylabel("Revenue (×10⁶)")
    axes[1].set_title("(b) Sensitivity to |B|")
    axes[1].legend(fontsize=7)
    save(fig, out, "fig10_sensitivity")


def fig11(data, out):
    r = data["fig11"]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    cats = np.arange(1, 6)
    starts = np.arange(0, r["T"] / 2, r["T"] / 10)
    for method in PAPER_ORDER:
        raw = r["sold_out_index"][method]
        vals = []
        for j, x in enumerate(raw):
            vals.append(np.nan if x is None else x - starts[j])
        ax.plot(cats, vals, marker="o", label=DISPLAY[method])
    ax.set_xticks(cats)
    ax.set_xlabel("Category")
    ax.set_ylabel("Buyer index since category selling begins")
    ax.set_title("Fig. 11 reproduction: selling speed")
    ax.legend()
    save(fig, out, "fig11_sellout_index")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="artifacts/icde_main_repro/results.json")
    ap.add_argument("--output", default="artifacts/icde_main_repro/figures")
    args = ap.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    fig7(data, out)
    fig8(data, out)
    fig9(data, out)
    fig10(data, out)
    fig11(data, out)
    print("generated", len(list(out.glob("*.png"))), "png figures")


if __name__ == "__main__":
    main()
