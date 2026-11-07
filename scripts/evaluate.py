import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from operator_ood_flow.geometry import polygon_corners
from operator_ood_flow.metrics import relative_velocity_error
from operator_ood_flow.model import DeepONet
from operator_ood_flow.preprocessing import EVAL_BAND, downsample_sdf, load_data, rel_l2

ROOT = Path(__file__).resolve().parent.parent

BATCH = 32  # shapes per forward pass
CORNER_R = 0.3  # corner region: query points within this distance (r0 units) of a vertex


def corner_mask(path, coords):
    """(N, H * W) True within CORNER_R of a rounding centre, for polygon grid datasets
    (they save sides, rho and phi); None for any other dataset."""
    with np.load(str(path) + ".npz") as d:
        if "phi" not in d.files:
            return None
        params = zip(d["sides"], d["rho"], d["phi"])
        return np.array([
            np.linalg.norm(
                coords[:, None] - polygon_corners(int(n), rho, phi, 1.0)[None], axis=-1
            ).min(1) < CORNER_R
            for n, rho, phi in params
        ])


def geometry(path):
    """Per-shape geometry labels for a dataset: family, class, convexity, and the
    sampler parameters that define it (polygon sides and rounding rho; lobe count k and
    amplitude a). Grid samplers save these; for the random lobed sampler k and a are
    recovered from radii."""
    with open(str(path) + ".json") as f:
        sampler = json.load(f)["sampler"]
    family = sampler.removesuffix(
        "_grid"
    )  # lobed_grid -> lobed, polygon_grid -> polygon
    with np.load(str(path) + ".npz") as d:
        margins, radii = d["margins"], d["radii"]
        n = len(margins)
        saved = {
            c: d[c] if c in d.files else np.full(n, np.nan)
            for c in ("sides", "rho", "k", "amp")
        }
    sides, rho, k, amp = saved["sides"], saved["rho"], saved["k"], saved["amp"]

    if family == "lobed" and np.isnan(k).all():
        # r = r0 (1 + a cos k(theta - phi)): dominant Fourier mode gives k, its size gives a
        c = np.abs(np.fft.rfft(radii, axis=1))
        k = c[:, 1:].argmax(1) + 1.0
        amp = 2 * c[np.arange(n), k.astype(int)] / c[:, 0]

    if family == "polygon":
        label = [f"polygon_n{int(s)}" for s in sides]
        if not np.isnan(rho).all():
            label = [f"{lab}_rho{r:.3f}" for lab, r in zip(label, rho)]
    elif family == "lobed":
        label = [f"lobed_k{int(kk)}" for kk in k]
    else:
        label = [family] * n
    return {
        "family": [family] * n,
        "geometry": label,
        "convex": margins >= 0,
        "margin": margins,
        "sides": sides,
        "rho": rho,
        "k": k,
        "amp": amp,
    }


def val_all_seeds_(dataset, val=False, seeds=(0, 123, 456, 789, 2026), out=None):
    """Evaluate each seed's checkpoints/seed<s>/best.pt on dataset. Writes one long
    CSV (a row per seed and shape, with geometry labels) and prints the mean and std
    across seeds. Returns {seed: (9,) array of ux, uy, p, then wall, then velocity
    vector (ux, uy) over fluid and wall, then the velocity RMSE / u_in near the
    corners (NaN unless the dataset is a polygon grid)}."""
    checkpoints = ROOT / "checkpoints"
    path = ROOT / "data" / dataset
    sdf, masks, fields, coords = load_data(path)  # same data for every seed
    geo = geometry(path)
    corners = corner_mask(path, coords)
    with open(str(path) + ".json") as f:
        u_in = json.load(f)["u_in"]
    Q = torch.tensor(coords)

    results, rows = {}, []
    for seed in seeds:
        ckpt = torch.load(
            checkpoints / f"seed{seed}" / "best.pt",
            map_location="cpu",
            weights_only=False,
        )
        model = DeepONet(**ckpt["model_cfg"])
        model.load_state_dict(ckpt["model"])
        model.eval()

        mean, std = ckpt["mean"], ckpt["std"]

        idx = np.array(ckpt["data_cfg"]["val_idx"]) if val else np.arange(len(masks))

        X = downsample_sdf(sdf[idx], tuple(ckpt["data_cfg"]["sdf_res"]))
        Y = torch.tensor(fields[idx].reshape(len(idx), -1, 3), dtype=torch.float32)
        M = torch.tensor(~masks[idx].reshape(len(idx), -1), dtype=torch.float32)
        Wm = M * torch.tensor(np.abs(sdf[idx].reshape(len(idx), -1)) < EVAL_BAND)
        C = None if corners is None else (M * torch.tensor(corners[idx])).bool()

        rel, rel_wall, vel = [], [], []
        with torch.no_grad():
            for s in range(0, len(idx), BATCH):
                pred = model(X[s : s + BATCH], Q) * std + mean
                rel.append(rel_l2(pred, Y[s : s + BATCH], M[s : s + BATCH]))
                rel_wall.append(rel_l2(pred, Y[s : s + BATCH], Wm[s : s + BATCH]))
                for b in range(len(pred)):  # velocity vector (ux, uy) over each region
                    vel.append(
                        [
                            relative_velocity_error(
                                pred[b, m[s + b].bool(), :2].numpy(),
                                Y[s + b, m[s + b].bool(), :2].numpy(),
                            )
                            for m in (M, Wm)
                        ]
                    )
                    # RMSE of the velocity vector over the corner region, in units of u_in
                    if C is None:
                        vel[-1].append(np.nan)
                    else:
                        c = C[s + b]
                        d2 = ((pred[b, c, :2] - Y[s + b, c, :2]) ** 2).sum(-1)
                        vel[-1].append(d2.mean().sqrt().item() / u_in)
        err = np.concatenate(
            [torch.cat(rel).numpy(), torch.cat(rel_wall).numpy(), np.array(vel)], axis=1
        )  # (shapes, 9)
        results[seed] = err.mean(0)  # mean over shapes: one number per field and region
        for i, e in zip(idx, err):
            rows.append([seed, i, *(geo[g][i] for g in geo), *e])

    table = np.stack(list(results.values()))  # (n_seeds, 9)
    cols = [
        "ux", "uy", "p", "ux_wall", "uy_wall", "p_wall", "vel", "vel_wall", "vel_corner_rmse"
    ]

    out = Path(out) if out else ROOT / "data" / "eval" / f"{dataset}_seeds.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "index", *geo, *cols])
        w.writerows(rows)
    print(f"saved {out}")

    print(f"{dataset}: mean rel L2 per seed")
    print("seed  " + "  ".join(f"{c:>7}" for c in cols))
    for seed, row in results.items():
        print(f"{seed:<5} " + "  ".join(f"{v:7.4f}" for v in row))
    print("mean  " + "  ".join(f"{v:7.4f}" for v in table.mean(0)))
    print(
        "std   " + "  ".join(f"{v:7.4f}" for v in table.std(0, ddof=1))
    )  # across seeds
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="per-shape errors of every seed's checkpoint."
    )
    parser.add_argument("dataset", help="name in data/, e.g. test_convex_s2_n200_seed1")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 123, 456, 789, 2026],
        help="evaluates checkpoints/seed<s>/best.pt for each seed",
    )
    parser.add_argument(
        "--val", action="store_true", help="only the checkpoints' val_idx"
    )
    parser.add_argument(
        "--out", help="CSV path (default: data/eval/<dataset>_seeds.csv)"
    )
    args = parser.parse_args()

    val_all_seeds_(args.dataset, val=args.val, seeds=args.seeds, out=args.out)
