from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import block_diag


def response_decomposition(T, true_delta, predicted_delta):
    """
    T:               [n_points, n_basis]
    true_delta:      [n_points]
    predicted_delta: [n_points]

    Assumes a common uniform grid and Euclidean norm.
    """

    # Best approximation within the trunk column space
    coefficients, *_ = np.linalg.lstsq(T, true_delta, rcond=1e-10)

    projected_delta = T @ coefficients

    # Orthogonal error components
    trunk_residual = true_delta - projected_delta
    coefficient_residual = projected_delta - predicted_delta

    denominator = np.linalg.norm(true_delta)

    if denominator < 1e-12:
        raise ValueError("True flow response is nearly zero")

    E_trunk = np.linalg.norm(trunk_residual) / denominator
    E_branch = np.linalg.norm(coefficient_residual) / denominator

    E_total = np.linalg.norm(true_delta - predicted_delta) / denominator

    return {
        "trunk_error": E_trunk,
        "branch_error": E_branch,
        "response_error": E_total,
        "trunk_fraction": E_trunk**2 / E_total**2,
        "branch_fraction": E_branch**2 / E_total**2,
        "identity_residual": abs(E_total**2 - E_trunk**2 - E_branch**2),
    }


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    EXP = ROOT / "experiments"
    DATASET = "ood_polygrid_s2_n180"
    SEEDS = (0, 123, 456, 789, 2026)
    # all: ux, uy and p together, in the training normalisation (q - mean) / std
    FIELDS = {"ux": [0], "uy": [1], "p": [2], "vel": [0, 1], "all": [0, 1, 2]}

    rows = []
    for seed in SEEDS:
        T = np.load(EXP / f"trunk_basis_seed{seed}.npz")["T"]  # (n_points, 3, p)
        d = np.load(EXP / f"responses_{DATASET}_seed{seed}.npz")
        # stored in normalised units; quadrature weights are omitted because the grid is
        # uniform, so the constant cell area cancels in every ratio and in the lstsq fit
        T, dq, dq_hat, std = (np.float64(a) for a in (T, d["dq"], d["dq_hat"], d["std"]))

        for i in np.flatnonzero(d["rho"] > 0):  # rho = 0 has no response
            for region in ("corner", "fluid"):
                r = d[region][i]
                for name, f in FIELDS.items():
                    # each field has its own trunk basis, so a vector field's basis is
                    # block diagonal over its components (component-major ordering)
                    # vel is in physical units, matching the response-error figures;
                    # scaling a field's rows by its std leaves that field's span unchanged
                    w = std[f] if name == "vel" else np.ones(len(f))
                    basis = block_diag(*(T[r, c] * wc for c, wc in zip(f, w)))
                    try:
                        res = response_decomposition(
                            basis,
                            (dq[i, r][:, f] * w).T.ravel(),
                            (dq_hat[i, r][:, f] * w).T.ravel(),
                        )
                    except ValueError:  # rounded shape rasterises like the sharp one
                        res = {}
                    rows.append(
                        {
                            "seed": seed,
                            "index": i,
                            "sides": d["sides"][i],
                            "rho": d["rho"][i],
                            "phi": d["phi"][i],
                            "region": region,
                            "field": name,
                            **res,
                        }
                    )

    df = pd.DataFrame(rows)
    out = ROOT / "data" / "eval" / f"{DATASET}_projection.csv"
    df.to_csv(out, index=False)
    print(f"saved {out}")
    cols = ["response_error", "trunk_error", "branch_error", "trunk_fraction"]
    print(
        df[df.field == "vel"]
        .groupby(["region", "sides", "rho"])[cols]
        .mean()
        .round(4)
        .to_string()
    )
