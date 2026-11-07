# Operator OOD Flow

**When a DeepONet trained on flow around smooth convex obstacles meets unseen shapes, where and why does it fail?**

Neural operator surrogates predict a simulation's output in milliseconds rather than minutes, which makes them attractive in engineering design. In design, though, they are used on new shapes unlike anything in their training data, and there a surrogate can be wrong while its predictions still look physically plausible. This project measures which kinds of new geometry a DeepONet handles, and investigates why it fails on the rest.

**Approach.** A lattice Boltzmann solver, written from scratch in JAX, generates steady 2D flow at Re = 20 around 1200 random smooth, near-convex obstacles. A DeepONet learns the map from each obstacle's signed distance field (SDF) to the velocity and pressure fields, and is trained with 5 model seeds. It is then tested on systematic grids that move away from the training family one property at a time: lobe count and amplitude, and polygon side count and corner rounding. All shapes have the same area. A rounding-sensitivity analysis compares how the input, the true flow and the prediction each change between a sharp polygon and its rounded versions.

**Status:** in progress, results preliminary (2026-11-07).

---

## Key preliminary findings

| Velocity relative $L^2$, mean of 5 seeds | All fluid cells | Wall band |
|---|---|---|
| In distribution (convex test set) | 0.012 | 0.093 |
| Lobes, k = 3 → 5 at amplitude 0.2 | 0.046 – 0.079 | 0.27 – 0.44 |
| Sharp polygons, hexagon → triangle | 0.024 – 0.220 | 0.16 – 0.77 |

- **In distribution the error is small** (about 1% over all fluid cells) and concentrated at the body surface. Seeds agree to within 0.01.
- **Lobed shapes:** error grows steadily with lobe amplitude, but not with non-convexity. k = 4 and k = 5 have nearly the same error even though k = 5 is far less convex.
- **Polygons:** side count sets the error. Triangles reach 18× the in-distribution error, hexagons 2×. Rounding the corners lowers near-corner error by up to 58%.
- **The model barely responds to the geometry change.** Rounding a triangle's corners to $0.3\,r_0$ changes the true near-corner flow by 0.23 $u_\mathrm{in}$, but the prediction by only 0.019 $u_\mathrm{in}$. The 32 × 64 input keeps the full SDF change, and the response points roughly the right way (alignment ≈ 0.5). Even so, the relative response error is 0.92–0.97, barely better than predicting no change (1.0). The falling error comes from the true flow becoming smoother, not from the model tracking the geometry.

Details, figures and limitations: [findings](docs/findings.md).

---

## Documentation

| Page | Contents |
|---|---|
| [Methodology](docs/methodology.md) | Experimental protocol, solver, datasets, model and training, metrics |
| [Findings](docs/findings.md) | Measured results, figures, interpretation and limitations |
| [Theory](docs/theory.md) | LBM, shape families, DeepONet, metric definitions and the response-error identity |
| [Issues log](docs/issues-log.md) | Problems found, their impact, and how each was fixed |

---

## Reproduction

Set up the Python 3.13 environment with `uv sync --locked`, then run from the project root. Dataset generation and training need a GPU. The Colab notebooks in `notebooks/` wrap the same commands.

```bash
# 1. Data (s = 2). A dataset's saved .json can be passed back to --config to regenerate it.
uv run python -m scripts.generate_dataset --scale 2 --max-steps 80000 --out train_convex_s2_n1200_seed0
uv run python -m scripts.generate_dataset --scale 2 --max-steps 80000 --seed 1 --n-shapes 200 --out test_convex_s2_n200_seed1
uv run python -m scripts.generate_dataset --sampler lobed_grid --scale 2 --max-steps 80000 --out ood_lobedgrid_s2_n180
uv run python -m scripts.generate_dataset --config <config with rhos 0..0.3>.json --out ood_polygrid_s2_n180
#    (config: copy data/ood_polygrid_s2_n180.json, or see notebooks/generate_polygrid_rho.ipynb)

# 2. Train: Ray Tune search, then retrain the best config -> checkpoints/best.pt
uv run python -m scripts.train
#    Seeds 0, 123, 456, 789, 2026 retrain the tuned config into checkpoints/seed<s>/best.pt
#    (see notebooks/train_seeds.ipynb; the train/val split is fixed at seed 0)

# 3. Evaluate every seed -> data/eval/<dataset>_seeds.csv
uv run python -m scripts.evaluate test_convex_s2_n200_seed1
uv run python -m scripts.evaluate ood_lobedgrid_s2_n180
uv run python -m scripts.evaluate ood_polygrid_s2_n180

# 4. Rounding sensitivity -> data/eval/ood_polygrid_s2_n180_rounding.csv
uv run python -m scripts.rounding_sensitivity ood_polygrid_s2_n180
```

Figures come from `notebooks/plots.ipynb`, using the functions in `operator_ood_flow/plotting.py`. Current results were generated from commit `5e83718` onward.

---

## Layout

| Path | Contents |
|---|---|
| `operator_ood_flow/` | Geometry and shape samplers, Gaussian random fields, LBM solver, DeepONet, preprocessing, metrics, plotting |
| `scripts/` | Dataset generation, training, evaluation, rounding sensitivity, LBM convergence study |
| `data/` | Datasets (`.npz` + `.json` settings); evaluation CSVs in `data/eval/` |
| `checkpoints/` | Model weights per seed (`seed<s>/best.pt`) |
| `figures/` | Figures used in the findings |
| `docs/` | Documentation |
