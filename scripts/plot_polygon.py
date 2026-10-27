import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
NAMES = ["$u_x$", "$u_y$", "$p$"]
COLS = ["ux", "uy", "p"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="per-shape error vs polygon side count.")
    parser.add_argument("csv", nargs="?", default=str(ROOT / "data/ood_polygon_eval.csv"),
                        help="output of scripts/evaluate.py on a polygon set")
    parser.add_argument("--ref", help="evaluate.py CSV for a reference set, e.g. data/test_convex_eval.csv")
    parser.add_argument("--wall", action="store_true", help="wall-band errors instead of all fluid cells")
    parser.add_argument("--out", help="default: docs/figures/ood_polygon[_wall].png")
    args = parser.parse_args()
    cols = [c + "_wall" for c in COLS] if args.wall else COLS
    out = args.out or str(ROOT / f"docs/figures/ood_polygon{'_wall' if args.wall else ''}.png")

    d = np.genfromtxt(args.csv, delimiter=",", names=True)
    stem = Path(args.csv).stem.replace("_eval", "")
    sides = np.load(ROOT / "data" / f"{stem}.npz")["sides"][d["index"].astype(int)]
    ns = np.unique(sides)
    ref = np.genfromtxt(args.ref, delimiter=",", names=True) if args.ref else None

    # --- table: mean and median per side count ---
    print(f"{'sides':>5}  {'n':>4}" + "".join(f"  {c + ' mean':>13}  {c + ' med':>12}" for c in cols))
    for n in ns:
        sel = sides == n
        print(f"{n:>5}  {sel.sum():>4}" + "".join(
            f"  {d[c][sel].mean():>13.3f}  {np.median(d[c][sel]):>12.3f}" for c in cols))
    print(f"{'all':>5}  {len(d):>4}" + "".join(
        f"  {d[c].mean():>13.3f}  {np.median(d[c]):>12.3f}" for c in cols))
    for c in cols:
        rho, _ = spearmanr(sides, d[c])
        print(f"Spearman rho(sides, {c}) = {rho:.2f}")

    # --- strip plot: every shape, median and IQR per side count ---
    rng = np.random.default_rng(0)
    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    for ax, c, name in zip(axes, cols, NAMES):
        x = sides + rng.uniform(-0.15, 0.15, len(sides))
        ax.scatter(x, d[c], s=10, color=color, alpha=0.35, linewidths=0)
        q1, med, q3 = np.array([np.percentile(d[c][sides == n], [25, 50, 75]) for n in ns]).T
        ax.errorbar(ns, med, yerr=[med - q1, q3 - med], fmt="o-", color=color,
                    markersize=6, linewidth=2, capsize=3, label="median, IQR")
        if ref is not None:
            ax.axhline(np.median(ref[c]), color="#444444", linestyle="--", linewidth=0.8,
                       label=f"{Path(args.ref).stem.replace('_eval', '')} median")
        ax.set_title(name)
        ax.set_xlabel("polygon sides")
        ax.set_xticks(ns)
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7)
    region = "wall band" if args.wall else "all fluid cells"
    axes[0].set_ylabel(f"relative $L^2$ ({region})")

    fig.savefig(out, dpi=100)
    print(f"saved {out}")
