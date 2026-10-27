import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from data import crop, downsample_sdf

NAMES = ["$u_x$", "$u_y$", "$p$"]
COLS = ["ux", "uy", "p"]


def branch_input(dataset, res, idx=None):
    """(N, res[0] * res[1]) downsampled SDF, exactly what the model's branch net sees."""
    path = ROOT / "data" / dataset
    with open(str(path) + ".json") as f:
        meta = json.load(f)
    with np.load(str(path) + ".npz") as d:
        sdf = d["sdf"] if idx is None else d["sdf"][idx]
    return downsample_sdf(crop(sdf, meta) / meta["r0"], res)


def nn_distance(X, train):
    """RMS SDF difference (units of r0) to the nearest training shape."""
    return (torch.cdist(X, train).min(dim=1).values / X.shape[1] ** 0.5).numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="per-shape error vs distance to the training set.")
    parser.add_argument("ckpt", help="e.g. models/best.pt")
    parser.add_argument("csv", nargs="+", help="output of scripts/evaluate.py, e.g. data/test_convex_eval.csv")
    parser.add_argument("--fit", nargs="+", default=["test_convex", "ood_lobed"],
                        help="datasets the log-error vs distance fit is made on")
    parser.add_argument("--out", default=str(ROOT / "docs/figures/error_vs_distance.png"))
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["data_cfg"]
    res = tuple(cfg["sdf_res"])
    train = branch_input(Path(cfg["data"]).name, res, np.array(cfg["tr_idx"]))
    print(f"{len(train)} training shapes, SDF at {res}")

    sets = []
    for csv in args.csv:
        d = np.genfromtxt(csv, delimiter=",", names=True)
        name = Path(csv).stem.replace("_eval", "")
        X = branch_input(name, res, d["index"].astype(int))
        sets.append((name, nn_distance(X, train), d))

    # fit log(error) = a + b * distance on the --fit sets, then check the others against it
    fit = [s for s in sets if s[0] in args.fit]
    coef = {}
    for c in COLS:
        if fit:
            x = np.concatenate([s[1] for s in fit])
            y = np.log(np.concatenate([s[2][c] for s in fit]))
            coef[c] = np.polyfit(x, y, 1)

    print(f"\n{'dataset':<16} {'n':>4} {'dist med':>9}" + "".join(
        f"  {c + ' rho':>7}  {c + ' resid':>9}" for c in COLS))
    for name, dist, d in sets:
        row = f"{name:<16} {len(d):>4} {np.median(dist):>9.3f}"
        for c in COLS:
            rho, _ = spearmanr(dist, d[c])
            # median ratio of actual to fitted error; > 1 means worse than its distance predicts
            resid = np.exp(np.median(np.log(d[c]) - np.polyval(coef[c], dist))) if coef else np.nan
            row += f"  {rho:>7.2f}  {resid:>9.2f}"
        print(row)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    xs = np.linspace(0, max(s[1].max() for s in sets), 100)
    for ax, c, name in zip(axes, COLS, NAMES):
        for (label, dist, d), color in zip(sets, colors):
            ax.scatter(dist, d[c], s=12, color=color, alpha=0.6, linewidths=0, label=label)
        if coef:
            ax.plot(xs, np.exp(np.polyval(coef[c], xs)), color="#444444", linestyle="--",
                    linewidth=1, label=f"fit on {', '.join(args.fit)}")
        ax.set_title(name)
        ax.set_xlabel("distance to nearest training shape (RMS SDF, $r_0$)")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("relative $L^2$ (all fluid cells)")

    fig.savefig(args.out, dpi=100)
    print(f"saved {args.out}")
