import argparse
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from operator_ood_flow.preprocessing import EVAL_BAND, load_data

ROOT = Path(__file__).resolve().parent.parent
COLS = ["ux", "uy", "p"]


def stem(family, s):
    """Dataset stem in data/ for one family at scale s, from scripts/convergence_study.py."""
    (path,) = (ROOT / "data").glob(f"conv_{family}_s{s}_*.json")
    return path.with_suffix("")


def axes(coords, shape):
    """1D y and x axes (r0 units) of a crop's (H * W, 2) query points."""
    xy = coords.reshape(*shape, 2)
    return xy[:, 0, 1], xy[0, :, 0]


def rel_err(a, b, m):
    """Per-shape relative L2 of a against b over cells m, (N, 3)."""
    m = m[..., None]
    num = np.sqrt((((a - b) * m) ** 2).sum(axis=(1, 2)))
    den = np.sqrt(((b * m) ** 2).sum(axis=(1, 2)))
    return num / den


def compare(family, s, ref):
    """Relative L2 of the scale-s solution against the ref solution, all-fluid and wall band."""
    sdf, masks, fields, coords = load_data(stem(family, s))
    _, f_masks, f_fields, f_coords = load_data(stem(family, ref))
    assert len(fields) == len(f_fields), "shape counts differ; non-converged shapes were dropped"

    H, W = masks.shape[1:]
    pts = coords.reshape(H, W, 2)[..., ::-1]  # (y, x) order for the interpolator
    f_axes = axes(f_coords, f_masks.shape[1:])
    interp = lambda a: RegularGridInterpolator(f_axes, a, bounds_error=False, fill_value=np.nan)(pts)

    fine = np.stack([interp(f) for f in f_fields])
    # drop coarse cells whose stencil touches a fine solid cell (solid p is zeroed) or leaves the crop
    f_solid = np.stack([interp(m.astype(np.float32)) for m in f_masks])
    fluid = ~masks & (f_solid == 0) & ~np.isnan(fine).any(axis=-1)
    fine = np.nan_to_num(fine)
    wall = fluid & (np.abs(sdf) < EVAL_BAND)
    return rel_err(fields, fine, fluid), rel_err(fields, fine, wall)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LBM grid convergence: each scale against the finest.")
    parser.add_argument("--families", nargs="+", default=["convex", "lobed", "polygon"])
    parser.add_argument("--scales", nargs="+", default=["1.5", "2"])
    parser.add_argument("--ref", default="3", help="reference scale s")
    args = parser.parse_args()

    print(f"relative L2 vs s = {args.ref}, mean (max) over shapes")
    print(f"{'family':<8} {'s':>4} {'region':<6} " + " ".join(f"{c:>15}" for c in COLS))
    for family in args.families:
        for s in args.scales:
            for region, err in zip(["fluid", "wall"], compare(family, s, args.ref)):
                cells = " ".join(f"{m:>7.2%} ({x:>5.2%})" for m, x in zip(err.mean(0), err.max(0)))
                print(f"{family:<8} {s:>4} {region:<6} {cells}")
