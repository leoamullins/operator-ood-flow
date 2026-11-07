from pathlib import Path

import numpy as np
import torch

from operator_ood_flow.model import DeepONet
from operator_ood_flow.preprocessing import downsample_sdf, load_data
from scripts.evaluate import corner_mask

ROOT = Path(__file__).resolve().parent.parent
DATASET = "ood_polygrid_s2_n180"
SEEDS = (0, 123, 456, 789, 2026)

path = ROOT / "data" / DATASET
sdf, masks, fields, coords = load_data(path)
with np.load(str(path) + ".npz") as d:
    sides, rho, phi = d["sides"].astype(int), d["rho"], d["phi"]
N = len(masks)
q = fields.reshape(N, -1, 3)

# index of the sharp counterpart (same sides and orientation, rho = 0) of each shape
sharp = {(n, ph): i for i, (n, ph, r) in enumerate(zip(sides, phi, rho)) if r == 0}
i0 = np.array([sharp[(n, ph)] for n, ph in zip(sides, phi)])

# cells fluid for every rho of the same family
cell_fluid = ~masks.reshape(N, -1)
fluid = np.zeros_like(cell_fluid)
for n, ph in set(zip(sides, phi)):
    group = (sides == n) & (phi == ph)
    fluid[group] = cell_fluid[group].all(0)
corner = corner_mask(path, coords, masks)

for seed in SEEDS:
    ckpt = torch.load(
        ROOT / f"checkpoints/seed{seed}/best.pt", map_location="cpu", weights_only=False
    )
    model = DeepONet(**ckpt["model_cfg"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    # float64: dq_hat is ~1% of q_hat, so float32 cancellation in q_hat_rho - q_hat_0
    # leaves a ~1e-4 relative error outside the trunk span
    model.double()
    X = downsample_sdf(sdf, tuple(ckpt["data_cfg"]["sdf_res"])).double()
    with torch.no_grad():
        q_hat = model(X, torch.tensor(coords).double()).numpy()  # normalised units
    mean, std = ckpt["mean"].double().numpy(), ckpt["std"].double().numpy()
    q_norm = (q - mean) / std

    out = ROOT / "experiments" / f"responses_{DATASET}_seed{seed}.npz"
    np.savez(
        out,
        dq=(q_norm - q_norm[i0]).astype(np.float32),  # (N, H*W, 3): ux, uy, p
        dq_hat=q_hat - q_hat[i0],
        fluid=fluid,  # (N, H*W)
        corner=corner,  # (N, H*W)
        i0=i0,
        sides=sides,
        rho=rho,
        phi=phi,
        mean=mean,
        std=std,
    )
    print(f"saved {out}")
