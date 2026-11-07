from pathlib import Path

import numpy as np
import torch

from operator_ood_flow.model import DeepONet
from operator_ood_flow.preprocessing import load_data

ROOT = Path(__file__).resolve().parent.parent
SEEDS = (0, 123, 456, 789, 2026)

masks = coords = Q = None
for seed in SEEDS:
    ckpt = torch.load(
        ROOT / f"checkpoints/seed{seed}/best.pt", map_location="cpu", weights_only=False
    )
    if Q is None:  # same training data for every seed
        _, masks, _, coords = load_data(ROOT / ckpt["data_cfg"]["data"])
        Q = torch.tensor(coords)

    model = DeepONet(**ckpt["model_cfg"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.double()  # float64 so the predicted responses lie in this basis's span to ~1e-13
    with torch.no_grad():
        T = model.trunk(Q.double()).view(
            -1, model.n_out, model.p
        )  # (n_points, 3, p): p basis functions each for ux, uy, p
        if model.zero_at_origin:
            T = T - model.trunk(torch.zeros(1, model.coord_dim).double()).view(
                1, model.n_out, model.p
            )

    # T: (n_points, 3, p); reshape a basis function with T[:, f, k].reshape(masks.shape[1:])
    out = ROOT / "experiments" / f"trunk_basis_seed{seed}.npz"
    np.savez(out, T=T.numpy(), coords=coords, masks=masks)
    print(f"saved {out}")
