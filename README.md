# Operator OOD Flow

**Status:** In progress. The project now asks *why* the DeepONet fails out of distribution, following the four-phase programme in the [experiment design](docs/experiment_design.md). The LBM grid-convergence study is done and set the lattice scale to s = 2. Five model seeds have been trained at s = 2 and reach 1–4% relative error in distribution. Systematic OOD grids of lobed shapes and rounded polygons have been generated and evaluated across all seeds, but not yet analysed.

---

## Problem statement

Neural operator surrogates predict a simulation's output in milliseconds rather than minutes, which makes them useful in engineering design, where many shapes need to be evaluated quickly. But in design they are used on new shapes, unlike anything in their training data. There a surrogate can be wrong while its predictions still look physically plausible. To know when to trust it, we need to know which kinds of new geometry it handles, and why it fails on the rest.

This project asks:

> **When a DeepONet trained on flow around smooth convex obstacles meets unseen shapes, where and why does it fail?**

**Approach.** A lattice Boltzmann solver, written from scratch and validated against flow past a cylinder, generates steady 2D flow at Re = 20 around random smooth convex obstacles. A DeepONet learns to map each obstacle's signed distance field to the velocity and pressure fields. It is then tested on shapes that move step by step away from the training data, along systematic grids of lobe amplitude and polygon side count and corner rounding. Controlled experiments then separate the candidate failure mechanisms: resolution, spectral bias, the operator's steepness, and collapse onto the nearest training shape.

---

## Key results so far

| Relative $L^2$ (test set, s = 2, mean over shapes and 5 model seeds) | $u_x$ | $u_y$ | $p$ |
|---|---|---|---|
| All fluid cells | 0.010 | 0.036 | 0.012 |
| Wall band ($\|\text{sdf}\| < 0.3\,r_0$) | 0.084 | 0.112 | 0.036 |

- **Seed spread is small:** the five seeds' mean errors are within 0.01 of each other (largest gap: wall-band $u_y$, 0.108–0.118).
- **Solver error:** going from s = 1.5 to s = 2 cuts the LBM error by a third to a half for convex and lobed shapes. Polygons converge slowest, which fits bounce-back being least accurate at sharp vertices.

---

## Documentation

| Page | Contents |
|---|---|
| [Data generation](docs/data.md) | Gaussian random functions, convex shape sampling, the LBM solver |
| [Experiment design](docs/experiment_design.md) | Four-phase programme for finding why the OOD failure happens |
| [Corner rounding](docs/corner-rounding.md) | Near-corner error vs rounding radius (preliminary) |
| [Issues log](docs/issues-log.md) | Problems found, their impact, and how each was fixed |

---

## Quickstart

Set up the Python 3.13 environment with `uv sync --locked`, then run from the project root:

```bash
uv run python -m scripts.generate_dataset --scale 2 --out train_convex_s2_n1200_seed0
uv run python -m scripts.generate_dataset --scale 2 --seed 1 --n-shapes 200 --out test_convex_s2_n200_seed1
uv run python -m scripts.train                                         # Ray Tune search, then retrains the best -> checkpoints/best.pt
uv run python -m scripts.evaluate test_convex_s2_n200_seed1            # checkpoints/seed<s>/best.pt -> data/eval/<dataset>_seeds.csv
```

---

## Layout

| Path | Contents |
|---|---|
| `operator_ood_flow/` | Geometry, Gaussian random fields, LBM solver, DeepONet, and data preparation |
| `scripts/` | Dataset generation, training, evaluation, and the LBM convergence study |
| `data/` | Generated datasets; per-seed evaluation CSVs in `data/eval/` |
| `checkpoints/` | Model weights per seed (`seed<s>/best.pt`) and training history |
| `docs/` | Project documentation |
