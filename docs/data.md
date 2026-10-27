# Data generation

[← Back to README](../README.md)

## Gaussian random functions

Input functions are drawn from a Gaussian process on $[0, 1]$ with an RBF kernel

$$
k(x, x') = \exp\left(-\frac{(x - x')^2}{2\ell^2}\right)
$$

This uses a Cholesky factorisation to transform a random sample $z$ into $u \sim N(0, K)$.

For the antiderivative, each $u$ is sampled on a fine grid and integrated using the cumulative trapezium rule.

## Convex shapes

Body boundaries are defined in polar form around the centre $(c_x, c_y)$:

$$
r(\theta) = r_0 \left(1 + a_1 \cos(\theta - \phi_1) + a_2 \cos 2(\theta - \phi_2) + \varepsilon\, g(\theta)\right)
$$

Here $g$ is a periodic Gaussian random function and $a_1, a_2, \phi_1, \phi_2$ are drawn uniformly. Shapes are kept only if they are (close to) convex. The check uses the curvature of the boundary, which is computed spectrally from $r$, $r'$ and $r''$.

## Flow solver

Each shape is solved with a D2Q9 lattice Boltzmann solver ([src/solvers/lbm.py](../src/solvers/lbm.py)), written in JAX and batched across shapes. The body uses bounce-back walls. The solver runs until the field stops changing (tolerance $10^{-5}$), and shapes that don't converge are dropped.

## Generating the dataset

```bash
uv run python scripts/generate_dataset.py
```

This writes `data/train_convex.npz` (masks, SDFs, $u_x$, $u_y$, $p$, radii, convexity margins) and `data/train_convex.json` (grid size, $r_0$, centre, $Re$ and the shape-sampling settings). The default grid is $150 \times 600$ with $r_0 = 15$.

Next: [Model and training](training.md)
