import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
NAMES = ["$u_x$", "$u_y$", "$p$"]
COLS = ["ux", "uy", "p"]


def lobe_params(csv_path):
    """Return (k, a) per shape from the stored radii via FFT.

    k = dominant wavenumber, a = relative amplitude (2|c_k|/c_0).
    Returns (None, None) if the sidecar is absent or sampler != 'lobed'.
    """
    stem = Path(csv_path).stem.replace("_eval", "")
    json_path = ROOT / "data" / f"{stem}.json"
    if not json_path.exists():
        return None, None
    with open(json_path) as f:
        meta = json.load(f)
    if meta.get("sampler") != "lobed":
        return None, None
    radii = np.load(ROOT / "data" / f"{stem}.npz")["radii"]
    spec = np.abs(np.fft.rfft(radii, axis=-1))
    k = np.argmax(spec[:, 1:], axis=-1) + 1
    a = 2 * spec[np.arange(len(k)), k] / spec[:, 0]
    return k, a


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="per-shape error vs convexity margin.")
    parser.add_argument("csv", nargs="+", help="output of scripts/evaluate.py")
    parser.add_argument("--out", default=str(ROOT / "docs/figures/ood_concavity.png"))
    parser.add_argument("--x", choices=["margin", "amp"], default="margin",
                        help="x-axis variable (amp: lobed only)")
    args = parser.parse_args()
    prop_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # load all CSVs upfront as (path, label, data) tuples
    datasets = [
        (csv, Path(csv).stem, np.genfromtxt(csv, delimiter=",", names=True))
        for csv in args.csv
    ]

    # reference baseline: first CSV, median in margin [0, 0.8]
    _, ref_label, ref_d = datasets[0]
    ref_mask = (ref_d["margin"] >= 0.0) & (ref_d["margin"] <= 0.8)
    ref_baseline = {col: np.median(ref_d[col][ref_mask]) for col in COLS}
    ref_idx = np.where(ref_mask)[0]

    # --- gap table with margin-match check and bootstrap CIs ---
    boot_rng = np.random.default_rng(0)
    header = f"{'dataset':<30}  {'n':>5}  {'med_margin':>10}" + "".join(
        f"  {c + ' ratio [95% CI]':<22}" for c in COLS
    )
    print(header)
    for _, label, d in datasets:
        m = (d["margin"] >= 0.0) & (d["margin"] <= 0.8)
        n_in = int(m.sum())
        med_margin = float(np.median(d["margin"][m]))
        idx = np.where(m)[0]
        ratios = np.array([np.median(d[col][idx]) / ref_baseline[col] for col in COLS])
        boot = np.array(
            [
                [
                    np.median(d[col][boot_rng.choice(idx, size=len(idx), replace=True)])
                    / np.median(ref_d[col][boot_rng.choice(ref_idx, size=len(ref_idx), replace=True)])
                    for col in COLS
                ]
                for _ in range(1000)
            ]
        )
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        ratio_strs = "".join(
            f"  {r:.3f} [{l:.3f},{h:.3f}]" for r, l, h in zip(ratios, lo, hi)
        )
        print(f"  {label:<28}  {n_in:>5}  {med_margin:>10.3f}{ratio_strs}")

    # bin-by-bin check if median margins differ by more than 0.1
    if len(datasets) > 1:
        _, _, d1 = datasets[1]
        m1 = (d1["margin"] >= 0.0) & (d1["margin"] <= 0.8)
        if abs(np.median(ref_d["margin"][ref_mask]) - np.median(d1["margin"][m1])) > 0.1:
            print("\nMargins differ >0.1 — bin-by-bin (width 0.2):")
            print(f"  {'bin':<10}" + "".join(f"  {c:>8}" for c in COLS) + "  (set2/set1)")
            for lo_b, hi_b in zip(np.arange(0.0, 0.8, 0.2), np.arange(0.2, 1.0, 0.2)):
                b0 = (ref_d["margin"] >= lo_b) & (ref_d["margin"] < hi_b)
                b1 = (d1["margin"] >= lo_b) & (d1["margin"] < hi_b)
                if b0.sum() < 3 or b1.sum() < 3:
                    continue
                r = [np.median(d1[col][b1]) / np.median(ref_d[col][b0]) for col in COLS]
                print(
                    f"  [{lo_b:.1f},{hi_b:.1f})  n={b0.sum():3d}/{b1.sum():3d}"
                    + "".join(f"  {v:>8.3f}" for v in r)
                )

    # --- scatter ---
    k_colors = {3: "#E07B39", 4: "#7BB356", 5: "#A45CA4"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)

    for (csv_path, label, d), base_color in zip(datasets, prop_colors):
        k_arr, a_arr = lobe_params(csv_path)
        indices = d["index"].astype(int)

        if k_arr is not None:
            k_rows = k_arr[indices]
            a_rows = a_arr[indices]
            for k_val, kc in k_colors.items():
                sel = k_rows == k_val
                if not sel.any():
                    continue
                x_vals = a_rows[sel] if args.x == "amp" else d["margin"][sel]
                for ax, col in zip(axes, COLS):
                    rho, _ = spearmanr(x_vals, d[col][sel])
                    ax.scatter(
                        x_vals,
                        d[col][sel] / ref_baseline[col],
                        s=14, color=kc, alpha=0.75, linewidths=0,
                        label=f"k={k_val}  ρ={rho:.2f}",
                    )
        elif args.x == "margin":
            for ax, col in zip(axes, COLS):
                rho, _ = spearmanr(d["margin"], d[col])
                ax.scatter(
                    d["margin"],
                    d[col] / ref_baseline[col],
                    s=14, color=base_color, alpha=0.8, linewidths=0,
                    label=f"{label}  ρ={rho:.2f}",
                )
        # if args.x == "amp" and k_arr is None: skip (no amp defined for convex shapes)

    # decoration pass
    x_label = "amplitude $a$" if args.x == "amp" else "convexity margin (lower = less convex)"
    for ax, col, name in zip(axes, COLS, NAMES):
        ax.set_title(name)
        ax.set_xlabel(x_label)
        ax.set_yscale("log")
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        if args.x == "margin":
            ax.axvline(-0.2, color="#444444", linestyle="--", linewidth=0.8, alpha=0.5,
                       label="train boundary")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7)
    axes[0].set_ylabel(f"relative $L^2$ / {ref_label} median (margin 0–0.8)")

    fig.savefig(args.out, dpi=100)
    print(f"saved {args.out}")
