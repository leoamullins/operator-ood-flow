"""How far each rounded polygon is from its sharp (rho = 0) counterpart, per shape:

- sdf_native, sdf_branch: ||z_rho - z_0||_2 of the SDF (r0 units) at the native crop
  resolution and at the branch resolution the model sees (downsample_sdf). Each is
  weighted by pixel area (r0^2), so both approximate the same continuous L2 norm
  and can be compared.
- flow_change: RMSE / u_in of the true LBM velocity between the two shapes.
- pred_change: RMSE / u_in of the model's predicted velocity between the two shapes.
- alignment: cosine similarity <dq, dq_hat> / (|dq| |dq_hat|) between the true velocity
  change dq = q_rho - q_0 and the predicted change dq_hat; 1 = the model responds in
  the right direction, 0 = unrelated, -1 = opposite. NaN at rho = 0 (no change).
- response_err: relative response error ||dq_hat - dq|| / ||dq||, i.e. how much of the
  rounding-induced flow change the model misses (0 = exact, 1 = no response at all).
  Equals sqrt(1 + R^2 - 2 R C) with R = ||dq_hat|| / ||dq|| = pred_change / flow_change
  and C = alignment. NaN at rho = 0.

The flow and prediction changes use the same fixed corner region as vel_corner_rmse
(scripts.evaluate.corner_mask). Writes data/eval/<dataset>_rounding.csv, one row per
seed and shape (the SDF and flow columns do not depend on the seed).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from operator_ood_flow.model import DeepONet
from operator_ood_flow.preprocessing import downsample_sdf, load_data
from scripts.evaluate import corner_mask

ROOT = Path(__file__).resolve().parent.parent


def rounding_sensitivity(dataset, seeds=(0, 123, 456, 789, 2026), out=None):
    path = ROOT / "data" / dataset
    sdf, masks, fields, coords = load_data(path)
    with open(str(path) + ".json") as f:
        u_in = json.load(f)["u_in"]
    with np.load(str(path) + ".npz") as d:
        sides, rho, phi = d["sides"].astype(int), d["rho"], d["phi"]
    C = corner_mask(path, coords, masks)
    N, H, W = masks.shape
    vel_true = fields.reshape(N, -1, 3)[..., :2]

    # index of the sharp counterpart (same sides and orientation, rho = 0) of each shape
    sharp = {(n, ph): i for i, (n, ph, r) in enumerate(zip(sides, phi, rho)) if r == 0}
    i0 = np.array([sharp[(n, ph)] for n, ph in zip(sides, phi)])

    def rmse(a, b, region):  # velocity RMSE / u_in over each shape's corner region
        return np.array([
            np.sqrt(((a[i, region[i]] - b[i, region[i]]) ** 2).sum(-1).mean()) / u_in
            for i in range(N)
        ])

    def cosine(a, b, region):  # alignment of two velocity changes over the corner region
        out = np.full(N, np.nan)
        for i in range(N):
            x, y = a[i, region[i]].ravel(), b[i, region[i]].ravel()
            norm = np.linalg.norm(x) * np.linalg.norm(y)
            if norm > 0:
                out[i] = x @ y / norm
        return out

    def response_error(dq, dq_hat, region):  # ||dq_hat - dq|| / ||dq|| over the region
        out = np.full(N, np.nan)
        for i in range(N):
            x, y = dq[i, region[i]], dq_hat[i, region[i]]
            if np.linalg.norm(x) > 0:
                out[i] = np.linalg.norm(y - x) / np.linalg.norm(x)
        return out

    rows = []
    for seed in seeds:
        ckpt = torch.load(
            ROOT / "checkpoints" / f"seed{seed}" / "best.pt",
            map_location="cpu",
            weights_only=False,
        )
        model = DeepONet(**ckpt["model_cfg"])
        model.load_state_dict(ckpt["model"])
        model.eval()
        res = tuple(ckpt["data_cfg"]["sdf_res"])
        X = downsample_sdf(sdf, res)
        with torch.no_grad():
            pred = (model(X, torch.tensor(coords)) * ckpt["std"] + ckpt["mean"]).numpy()

        z = sdf.reshape(N, -1)
        sdf_native = np.linalg.norm(z - z[i0], axis=1) * np.sqrt(5 / H * 10 / W)
        sdf_branch = np.linalg.norm(
            (X - X[i0]).numpy(), axis=1
        ) * np.sqrt(5 / res[0] * 10 / res[1])  # crop is 5 r0 x 10 r0
        rows.append(pd.DataFrame({
            "seed": seed,
            "index": np.arange(N),
            "sides": sides,
            "rho": rho,
            "phi": phi,
            "sdf_native": sdf_native,
            "sdf_branch": sdf_branch,
            "flow_change": rmse(vel_true, vel_true[i0], C),
            "pred_change": rmse(pred[..., :2], pred[i0][..., :2], C),
            "alignment": cosine(
                vel_true - vel_true[i0], pred[..., :2] - pred[i0][..., :2], C
            ),
            "response_err": response_error(
                vel_true - vel_true[i0], pred[..., :2] - pred[i0][..., :2], C
            ),
        }))

    df = pd.concat(rows, ignore_index=True)
    out = Path(out) if out else ROOT / "data" / "eval" / f"{dataset}_rounding.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"saved {out}")
    cols = [
        "sdf_native", "sdf_branch", "flow_change", "pred_change", "alignment", "response_err"
    ]
    print(df.groupby(["sides", "rho"])[cols].mean().round(4).to_string())
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SDF, flow and prediction change of rounded polygons vs sharp ones."
    )
    parser.add_argument("dataset", help="polygon grid in data/, e.g. ood_polygrid_s2_n180")
    parser.add_argument("--out", help="CSV path (default: data/eval/<dataset>_rounding.csv)")
    args = parser.parse_args()
    rounding_sensitivity(args.dataset, out=args.out)
