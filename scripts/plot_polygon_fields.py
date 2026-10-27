import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "scripts"))

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from data import downsample_sdf, load_data
from model import DeepONet
from plot_trefoil import plot_fields

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepONet vs LBM on the median polygon of each side count.")
    parser.add_argument("ckpt", help="e.g. models/best.pt")
    parser.add_argument("--dataset", default="ood_polygon", help="name in data/")
    parser.add_argument("--sides", type=int, nargs="+", default=[3, 4, 6])
    parser.add_argument("--out-dir", default=str(ROOT / "docs/figures"))
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = DeepONet(**ckpt["model_cfg"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    path = ROOT / "data" / args.dataset
    sdf, masks, fields, coords = load_data(path)
    with np.load(str(path) + ".npz") as d:
        sides = d["sides"]
    ev = np.genfromtxt(ROOT / "data" / f"{args.dataset}_eval.csv", delimiter=",", names=True)
    err = np.empty(len(masks))
    err[ev["index"].astype(int)] = ev["ux"]

    extent = [coords[:, 0].min(), coords[:, 0].max(), coords[:, 1].min(), coords[:, 1].max()]
    for n in args.sides:
        # median ux error among shapes with n sides
        idx = np.where(sides == n)[0]
        i = idx[np.argsort(err[idx])[len(idx) // 2]]
        x = downsample_sdf(sdf[i : i + 1], tuple(ckpt["data_cfg"]["sdf_res"]))
        with torch.no_grad():
            pred = model(x, torch.tensor(coords)) * ckpt["std"] + ckpt["mean"]
        pred = pred.numpy().reshape(*masks[i].shape, 3)
        true, solid = fields[i], masks[i]
        fluid = ~solid
        rel = np.linalg.norm((pred - true)[fluid], axis=0) / np.linalg.norm(true[fluid], axis=0)
        print(f"{n} sides, shape {i}: rel L2 (ux, uy, p) {rel.round(3)}")

        fig = plot_fields(true, pred, solid, extent, f"{n}-sided polygon (median of {len(idx)}), shape {i}", rel)
        out = Path(args.out_dir) / f"ood_polygon_fields_n{n}.png"
        fig.savefig(out, dpi=100)
        plt.close(fig)
        print(f"saved {out}")
