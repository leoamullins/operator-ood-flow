# Data generation

[← Back to README](../README.md)

## Gaussian random functions

The boundary perturbation $g(\theta)$ is drawn from a Gaussian process on $[0, 2\pi)$ with a periodic squared-exponential kernel

$$
k(\theta, \theta') = \exp\left(-\frac{2\sin^2\left((\theta - \theta')/2\right)}{\ell^2}\right)
$$

This uses a Cholesky factorisation to transform a random sample $z$ into $g \sim N(0, K)$.

## Convex shapes

Body boundaries are defined in polar form around the centre $(c_x, c_y)$:

$$
r(\theta) = r_0 \left(1 + a_1 \cos(\theta - \phi_1) + a_2 \cos 2(\theta - \phi_2) + \varepsilon\, g(\theta)\right)
$$

Here $g$ is a periodic Gaussian random function and $a_1, a_2, \phi_1, \phi_2$ are drawn uniformly. Shapes are kept only if they are (close to) convex. The check uses the curvature of the boundary, which is computed spectrally from $r$, $r'$ and $r''$.

## Flow solver

Each shape is solved with a D2Q9 lattice Boltzmann solver ([lbm.py](../operator_ood_flow/lbm.py)), written in JAX and batched across shapes. The body uses bounce-back walls. The solver runs until the field stops changing (tolerance $10^{-5}$), and shapes that don't converge are dropped.

## Generating the dataset

```bash
uv run python -m scripts.generate_dataset
```

Settings are passed as flags (`--sampler`, `--n-shapes`, `--seed`, `--scale`, ...), or as a JSON file of `Generator` fields with `--config`, in which case the output is named after that file.

This writes `data/train_convex.npz` (masks, SDFs, $u_x$, $u_y$, $p$, radii, convexity margins, and per-shape sampler parameters such as polygon `sides`) and `data/train_convex.json`. The JSON holds the full generation settings, the grid size, $r_0$ and centre, and a `provenance` block: git commit (and whether the tree had uncommitted changes), Python and package versions, JAX backend and creation time. Passing a saved dataset `.json` back to `--config` regenerates the same shapes.

The lattice scale $s$ (`--scale`, default 1.5) sets $r_0 = 10s$ cells and a $100s \times 400s$ grid, so the default is $150 \times 600$ with $r_0 = 15$. A given seed draws the same shapes at every $s$, so datasets at different resolutions are directly comparable.

Next: [Model and training](training.md)
