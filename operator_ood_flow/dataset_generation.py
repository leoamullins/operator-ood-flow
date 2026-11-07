import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

from .geometry import (
    sample_convex_shapes,
    sample_lobed_grid,
    sample_lobed_shapes,
    sample_polygon_grid,
    sample_polygon_shapes,
    signed_distance,
)
from .lbm import solve_batch

DEFAULT_SHAPE_KW = {
    "convex": {"eps": 0.08, "ell": 1.0, "a1_max": 0.3, "a2_max": 0.18, "tol": -0.2},
    "lobed": {"amp": [0.1, 0.4]},
    "polygon": {"sides": [3, 4, 5, 6, 7, 8]},
    "lobed_grid": {"ks": [3, 4, 5], "amps": [0.04, 0.08, 0.12, 0.16, 0.2], "n_orient": 12},
    "polygon_grid": {"sides": [3, 4, 6], "rhos": [0.0, 0.025, 0.05, 0.075], "n_orient": 10},
}
SAMPLERS = {
    "convex": sample_convex_shapes,
    "lobed": sample_lobed_shapes,
    "polygon": sample_polygon_shapes,
    "lobed_grid": sample_lobed_grid,
    "polygon_grid": sample_polygon_grid,
}
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Generator:
    """Generate steady-flow datasets from seeded obstacle samples."""

    sampler: str = "convex"
    n_shapes: int = 1200
    seed: int = 0
    s: float = 1.5
    Re: float = 20
    u_in: float = 0.1
    max_steps: int = 60_000
    batch: int = 100
    shape_kw: dict = field(default_factory=dict)

    def __post_init__(self):
        """Fill sampler defaults and derive lattice dimensions from the scale."""
        self.shape_kw = self.shape_kw or dict(DEFAULT_SHAPE_KW[self.sampler])

        self.r0 = 10 * self.s
        self.Ny, self.Nx = int(100 * self.s), int(400 * self.s)
        self.cx, self.cy = int(100 * self.s), int(50 * self.s)

    def to_dict(self):
        """Return generation settings and derived lattice dimensions for JSON."""
        derived = {
            "r0": self.r0,
            "Ny": self.Ny,
            "Nx": self.Nx,
            "cx": self.cx,
            "cy": self.cy,
        }
        return {**asdict(self), **derived}

    @classmethod
    def from_json(cls, path):
        """Load saved generation settings and recompute the lattice dimensions."""
        with open(path) as f:
            d = json.load(f)
        return cls(**{k.name: d[k.name] for k in fields(cls) if k.name in d})

    def sample(self):
        """Return masks, polar radii, convexity margins, and per-shape sampler parameters."""
        rng = np.random.default_rng(self.seed)
        out = SAMPLERS[self.sampler](
            self.n_shapes,
            self.Ny,
            self.Nx,
            self.cx,
            self.cy,
            r0=self.r0,
            rng=rng,
            **self.shape_kw,
        )
        return out if len(out) == 4 else (*out, {})

    def solve(self, masks):
        """Solve masks in batches and return ux, uy, pressure, and convergence flags."""
        ux, uy, p, conv = [], [], [], []
        for start in range(0, len(masks), self.batch):
            b = masks[start : start + self.batch]
            rho, u, c, steps = solve_batch(
                b, D=2 * self.r0, Re=self.Re, u_in=self.u_in, max_steps=self.max_steps
            )
            p_b = (rho / 3).astype(np.float32)
            p_b[b] = 0
            ux.append(u[:, 0].astype(np.float32))
            uy.append(u[:, 1].astype(np.float32))
            p.append(p_b)
            conv.append(c)
            print(f"shapes {start}-{start + len(b) - 1}: {steps} steps", flush=True)
        return [np.concatenate(a) for a in (ux, uy, p, conv)]

    def save(self, out, data_dir, **arrays):
        """Write compressed arrays and a JSON sidecar with settings and the saved count."""
        data_dir.mkdir(exist_ok=True)
        np.savez_compressed(data_dir / f"{out}.npz", **arrays)
        with open(data_dir / f"{out}.json", "w") as f:
            json.dump({**self.to_dict(), "n": len(arrays["ux"])}, f, indent=2)

    def run(self, out, data_dir=DEFAULT_PATH):
        """Sample obstacles, solve their flows, and save converged cases with SDFs."""
        masks, radii, margins, params = self.sample()
        ux, uy, p, keep = self.solve(masks)
        print(f"{(~keep).sum()} of {len(keep)} shapes did not converge, dropping them")
        sdf = np.array([signed_distance(m) for m in masks[keep]]).astype(np.float32)
        self.save(
            out,
            data_dir,
            masks=masks[keep],
            sdf=sdf,
            ux=ux[keep],
            uy=uy[keep],
            p=p[keep],
            radii=radii[keep],
            margins=margins[keep],
            **{name: v[keep] for name, v in params.items()},
        )
