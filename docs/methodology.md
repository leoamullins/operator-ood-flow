# Methodology

[← Back to README](../README.md) · [Findings](findings.md) · [Theory](theory.md) · [Issues log](issues-log.md)

The protocol behind every result in [findings](findings.md): how the data is generated, how the model is trained, and how errors are measured. Values are those used for the current results (s = 2). Derivations are in [theory](theory.md).

---

## 1. Protocol overview

1. **Generate training data:** 1200 smooth, near-convex obstacles, each solved to steady flow with the LBM.
2. **Train** a DeepONet from obstacle SDF to $(u_x, u_y, p)$. Hyperparameters are tuned once with Ray Tune, and the tuned configuration is then retrained with 5 model seeds.
3. **Generate test sets:** an in-distribution test set and two OOD grids. The grids vary one geometric property at a time (lobe count and amplitude; side count and corner rounding), with 10–12 orientations at each grid point and a fixed area of $\pi r_0^2$.
4. **Evaluate** every seed on every set (`scripts/evaluate.py`): one row per seed and shape, with geometry labels.
5. **Rounding sensitivity** (`scripts/rounding_sensitivity.py`): compare each rounded polygon with its sharp counterpart, in the input, the true flow and the prediction.

Every number is reported as a mean over the 5 seeds, with the std across seeds as the uncertainty. Each seed's value is first averaged over shapes (or orientations).

---

## 2. Flow solver

| Item | Value |
|---|---|
| Method | D2Q9 lattice Boltzmann, BGK collision, in JAX, batched over shapes (`operator_ood_flow/lbm.py`) |
| Domain | $N_y \times N_x = 100s \times 400s$ = 200 × 800 cells at s = 2 |
| Body | centred at $(c_x, c_y) = (100s, 50s)$, length scale $r_0 = 10s$ = 20 cells |
| Flow | Re = 20 on $D = 2r_0$; $u_\mathrm{in}$ = 0.1 (Ma ≈ 0.17); $\tau = 3u_\mathrm{in}D/\mathrm{Re} + 0.5$ = 1.1 |
| Boundaries | equilibrium inlet at $u_\mathrm{in}$ (left); zero-gradient outlet (right); bounce-back on the body; periodic top and bottom (from streaming) |
| Convergence | in chunks of 2000 steps, until the maximum velocity change per chunk is below $10^{-5}$; at most 80,000 steps at s = 2 |
| Non-converged shapes | dropped; none were dropped in any current dataset ([issues log #6](issues-log.md)) |

**Resolution choice.** s = 2 was chosen by a grid-convergence study at s = 1.5, 2 and 3 on 20 convex, 20 lobed and 10 polygon shapes. At s = 1.5 the solver's own error was about as large as the model's in-distribution error; going to s = 2 cut it by a third to a half for convex and lobed shapes. Polygons converge slowest ([issues log #5](issues-log.md)).

**Pressure** is $p = c_s^2\rho = \rho/3$, measured relative to the mean at the upstream edge of the crop, and set to zero inside the body.

---

## 3. Geometry and datasets

All shapes are star-shaped around the body centre and described by polar radii $r(\theta)$ on 128 angles. Each is rasterised to a solid mask, from which the signed distance field (SDF) is computed with an exact Euclidean distance transform (`operator_ood_flow/geometry.py`).

### Samplers

- **Convex (training and test):** $r(\theta) = r_0\,(1 + a_1\cos(\theta - \phi_1) + a_2\cos 2(\theta - \phi_2) + \varepsilon g(\theta))$. Here $a_1 \sim U(0, 0.3)$, $a_2 \sim U(0, 0.18)$, the phases are uniform, $\varepsilon$ = 0.08, and $g$ is a periodic Gaussian random function with length scale $\ell$ = 1. A shape is kept if its convexity margin is at least −0.2, so the set is *near*-convex: training margins run from −0.197 to 0.84, with median 0.35, and 12.5% of training shapes are slightly non-convex (margin < 0).
- **Lobed grid:** $r(\theta) = r_0\,(1 + a\cos k(\theta - \phi)) / \sqrt{1 + a^2/2}$, which has area exactly $\pi r_0^2$. It covers k ∈ {3, 4, 5} × a ∈ {0.04, 0.08, …, 0.20} × 12 orientations spanning one lobe period.
- **Polygon grid:** regular n-gons with corners rounded to radius $\rho r_0$. Each is built as an inner n-gon grown by $\rho r_0$, sized so that the area stays $\pi r_0^2$ (`geometry.polygon_corners`). It covers n ∈ {3, 4, 6} × ρ × 10 orientations spanning one symmetry period.

### Datasets (all at s = 2, in `data/`)

| Name | Role | Sampler | Shapes |
|---|---|---|---|
| `train_convex_s2_n1200_seed0` | train (1000) + validation (200) | convex, seed 0 | 1200 |
| `test_convex_s2_n200_seed1` | in-distribution test | convex, seed 1 | 200 |
| `ood_lobedgrid_s2_n180` | OOD: lobes | lobed grid | 3 × 5 × 12 = 180 |
| `ood_polygrid_s2_n120` | OOD: corners (narrow ρ) | polygon grid, ρ ∈ {0, 0.025, 0.05, 0.075} | 3 × 4 × 10 = 120 |
| `ood_polygrid_s2_n180` | OOD: corners (wide ρ) | polygon grid, ρ ∈ {0, 0.05, 0.1, 0.15, 0.2, 0.3} | 3 × 6 × 10 = 180 |

Each dataset's `.json` sidecar records its full generation settings, so passing it back to `--config` regenerates the dataset. Grid samplers are deterministic. A dataset's `.json` doesn't record the code version, so the commit is noted where it matters ([issues log #12](issues-log.md)).

---

## 4. Preprocessing

- **Crop:** $[-2r_0, 8r_0] \times [-2.5r_0, 2.5r_0]$ around the body centre, which is 100 × 200 cells at s = 2.
- **Branch input:** SDF in units of $r_0$, bilinearly downsampled to 32 × 64 and flattened, giving m = 2048 inputs. One pixel is $0.156\,r_0$.
- **Query points:** every cell of the crop, as $(x, y)$ in units of $r_0$ relative to the body centre.
- **Targets:** $(u_x, u_y, p)$, normalised per field with the mean and std over the training set's fluid cells.

---

## 5. Model and training

**DeepONet** (`operator_ood_flow/model.py`): the branch MLP (SDF → features) and the trunk MLP (query point → features) each have two tanh hidden layers of width $w$, and the trunk output also passes through a tanh. There are $p$ features per output field. Each field is the dot product of its branch and trunk features plus a learned bias.

**Training** (`scripts/train.py`):
- **Loss:** MSE over fluid cells on 2048 random query points per batch, with 32 shapes per batch.
- **Optimiser:** AdamW, with linear warmup over the first 2% of steps, then cosine decay to $10^{-3}$ of the peak LR.
- **Stability:** gradient norm clipped to 1.0. The forward pass runs under bf16 autocast.
- **Hyperparameter search:** Ray Tune with ASHA (20 samples, up to 1000 epochs, grace period 50) over LR ∈ log-U(1e-4, 1e-3), width ∈ {256, 512}, p ∈ {128, 256} and weight decay ∈ {0, 1e-4}.
- **Selected configuration:** LR $2.65\times10^{-4}$, width 512, p = 128, weight decay 0, 1000 epochs.

**Seeds:** the selected configuration was retrained with model seeds 0, 123, 456, 789 and 2026. The seed sets the initialisation, batch order and query-point sampling. The train/validation split is the same for all seeds (seed 0). Each checkpoint is the epoch with the lowest mean validation relative $L^2$, which was between epochs 982 and 999.

---

## 6. Metrics

All metrics are computed per shape on the crop, in physical (un-normalised) units. Definitions are in [theory §5](theory.md#5-error-metrics).

| Column | Definition | Region |
|---|---|---|
| `ux`, `uy`, `p` | relative $L^2$ per field | all fluid cells |
| `ux_wall`, `uy_wall`, `p_wall` | relative $L^2$ per field | wall band: fluid cells with $\|\mathrm{sdf}\| < 0.3\,r_0$ |
| `vel`, `vel_wall` | relative $L^2$ of the velocity vector $(u_x, u_y)$ | all fluid / wall band |
| `vel_corner_rmse` | velocity RMSE / $u_\mathrm{in}$ | corner region (polygon grids only) |

**Corner region:** fluid cells within $0.5\,r_0$ of a vertex of the *sharp* polygon with the same side count and orientation, keeping only cells that are fluid for every ρ. Every ρ is therefore scored on identical cells ([issues log #11](issues-log.md)). Relative errors are avoided there because the velocity is near zero at a no-slip wall, which would inflate them ([issues log #10](issues-log.md)).

### Rounding sensitivity

For each rounded polygon $(n, \phi, \rho)$ and its sharp counterpart $(n, \phi, 0)$, `scripts/rounding_sensitivity.py` computes:

| Column | Quantity |
|---|---|
| `sdf_native`, `sdf_branch` | $\|z_\rho - z_0\|_2$ of the SDF, weighted by pixel area, at native and at branch resolution |
| `flow_change` | $\|\Delta q\|$ as RMSE / $u_\mathrm{in}$, where $\Delta q = q_\rho - q_0$ is the change in true velocity |
| `pred_change` | $\|\Delta\hat q\|$ as RMSE / $u_\mathrm{in}$, the change in predicted velocity |
| `alignment` | $C = \langle\Delta q, \Delta\hat q\rangle / (\|\Delta q\|\|\Delta\hat q\|)$ |
| `response_err` | $E = \|\Delta\hat q - \Delta q\| / \|\Delta q\|$ |

The flow quantities use the corner region. Output: `data/eval/<dataset>_rounding.csv`, one row per seed and shape.
