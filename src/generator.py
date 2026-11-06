import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

from geometry import sample_convex_shapes, sample_lobed_shapes, signed_distance
from solvers.lbm import solve_batch

DEFAULT_SHAPE_KW = {
    "convex": {"eps": 0.08, "ell": 1.0, "a1_max": 0.3, "a2_max": 0.18, "tol": -0.2},
    "lobed": {"amp": [0.1, 0.4]},
}
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Generator:
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
        self.shape_kw = self.shape_kw or dict(DEFAULT_SHAPE_KW[self.sampler])

        self.r0 = 10 * self.s
        self.Ny, self.Nx = int(100 * self.s), int(400 * self.s)
        self.cx, self.cy = int(100 * self.s), int(50 * self.s)

    def to_dict(self):
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
        with open(path) as f:
            d = json.load(f)
        return cls(**{k.name: d[k.name] for k in fields(cls) if k.name in d})

    def sample(self):
        rng = np.random.default_rng(self.seed)
        fn = sample_convex_shapes if self.sampler == "convex" else sample_lobed_shapes
        return fn(
            self.n_shapes,
            self.Ny,
            self.Nx,
            self.cx,
            self.cy,
            r0=self.r0,
            rng=rng,
            **self.shape_kw,
        )

    def solve(self, masks):
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
        data_dir.mkdir(exist_ok=True)
        np.savez_compressed(data_dir / f"{out}.npz", **arrays)
        with open(data_dir / f"{out}.json", "w") as f:
            json.dump({**self.to_dict(), "n": len(arrays["ux"])}, f, indent=2)

    def run(self, out, data_dir=DEFAULT_PATH):
        masks, radii, margins = self.sample()
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
        )
