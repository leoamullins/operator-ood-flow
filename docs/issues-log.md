# Issues log

[← Back to README](../README.md)

Problems found during the project, what each one affected, and how it was fixed. This is meant to support transparent reporting: each entry cites the commit or check that shows it. Entries marked **Open** are not fixed yet, and any result they affect should be reported with that caveat.

| # | Issue | Area | Status |
|---|---|---|---|
| 1 | Training diverged at peak LR 1e-3 | Training | Fixed |
| 2 | Validation error plateaued without LR decay | Training | Fixed |
| 3 | Whole-domain error hides wall error | Metrics | Fixed |
| 4 | A finer branch input did not help | Model | Explained, not pursued |
| 5 | Solver error comparable to model error at s = 1.5 | Data | Fixed (moved to s = 2) |
| 6 | Shapes that don't converge are dropped | Data | Checked: none dropped |
| 7 | Bootstrap CI resampled only one set | Analysis (retired) | Fixed |
| 8 | Quadrature step mismatch in the S¹ velocity error | Metrics | Fixed |
| 9 | Per-component errors can't be combined into a velocity error | Metrics | Fixed |
| 10 | Relative L² is inflated near no-slip walls | Metrics | Worked around |
| 11 | Corner measurement region shrinks as ρ grows | Metrics | **Open** |
| 12 | Corner rounding is below the branch input's resolution | Experiment design | **Open** |
| 13 | Corrupted header in an evaluation CSV | Data handling | Fixed |

---

## Training and model

### 1. Training diverged at peak LR 1e-3 — Fixed
- **What happened:** a 1000-epoch run at peak learning rate 1e-3 spiked around epoch 80 and never recovered.
- **Fix:** gradient norm clipped to 1.0, and peak LR lowered to 5e-4.
- **Evidence:** commit `3c555f3` (2026-10-22).

### 2. Validation error plateaued without LR decay — Fixed
- **What happened:** when training was simplified for the Ray Tune search, the warmup + cosine schedule was dropped. With a constant LR, validation relative error plateaued at about 0.15.
- **Fix:** the warmup + cosine schedule and gradient clipping were restored. On the same data, the scheduled run reached about 0.06 by epoch 160.
- **Evidence:** commit `ac0d398` (2026-11-07).

### 3. Whole-domain error hides wall error — Fixed
- **What happened:** errors are concentrated in a thin layer at the body surface, so they barely move the error averaged over all fluid cells.
- **Fix:** validation and evaluation also report relative L² over a wall band, the fluid cells with $|\text{sdf}| < 0.3\,r_0$ (`EVAL_BAND`). Near-wall oversampling of query points (`WALL_FRAC`) was also added. It was removed when training was simplified (`868fb9c`), so the current s = 2 models were trained with uniform query sampling.
- **Evidence:** commit `93a5ef3` (2026-10-22). The `*_wall` columns are in `data/eval/*.csv`.

### 4. A finer branch input did not help — Explained, not pursued
- **What happened:** a 128 × 256 downsampled SDF as branch input gave no improvement over 32 × 64.
- **Diagnosis:** Adam's effective learning rate for the first layer grows with the number of inputs to that layer (its fan-in), so simply adding pixels changes the optimisation as well as the resolution.
- **Status:** the 32 × 64 input was kept. This limits what the model can see; see #12.

## Data generation

### 5. Solver error comparable to model error at s = 1.5 — Fixed
- **What happened:** a grid-convergence study solved the same 20 convex, 20 k = 5 lobed and 10 polygon shapes at s = 1.5, 2 and 3, compared against s = 3. On convex shapes at s = 1.5, the solver's all-fluid error ($u_x, u_y, p$ = 0.006, 0.016, 0.015) was about as large as the DeepONet's test error at that time (0.011, 0.034, 0.013). The model was therefore close to the accuracy of its own training data.
- **Fix:** all current datasets use s = 2 ($r_0$ = 20 cells), with `max_steps` scaled with $r_0$ to 80,000. Going from s = 1.5 to 2 cut the solver error by about a third to a half for convex and lobed shapes.
- **Remaining caveat:** polygons converge slowest. At s = 2, the worst polygon still has a wall-band $u_x$ error of about 0.19 against s = 3, which is consistent with bounce-back boundaries being least accurate at sharp vertices.
- **How the comparison was done:** coarse cells whose interpolation stencil touches a fine-grid solid cell were excluded, because pressure is set to zero inside the body.
- **Evidence:** commits `f5e129a`, `8cb5c41`, `8b7353f`. The write-up (`docs/convergence.md`) has been deleted; the numbers above come from it.

### 6. Shapes that don't converge are dropped — Checked: none dropped
- **Risk:** `Generator.run` drops shapes that don't reach the convergence tolerance. If hard shapes fail more often, this could bias a dataset toward easier shapes.
- **Check:** every current dataset has its full requested count (`n` in its `.json`): train 1200, test 200, lobed grid 180, polygon grid 120.

## Analysis and metrics

### 7. Bootstrap CI resampled only one set — Fixed (analysis since retired)
- **What happened:** in the lobed-vs-convex family-shift comparison, the bootstrap resampled only one of the two sets. That made the reference confidence interval artificially narrow.
- **Fix:** both sets are now resampled.
- **Evidence:** commit `3cc9f18` (2026-10-24). This analysis belongs to the earlier research angle and has since been removed from the repo.

### 8. Quadrature step mismatch in the S¹ velocity error — Fixed
- **What happened:** the first version of `relative_velocity_error` passed `dx=0.01` to `simpson` for the numerator but left the default `dx=1` for the denominator. The result was therefore too small by a factor of $\sqrt{dx}$, 10× at `dx=0.01`. Simpson's rule also misses the wrap-around interval on a periodic domain.
- **Fix:** `operator_ood_flow/metrics.py` now uses `np.linalg.norm(pred - true) / np.linalg.norm(true)`. On a uniform periodic grid, the grid spacing cancels in the ratio, and the plain sum is the periodic trapezoid rule.
- **Assumption:** samples are evenly spaced, and the endpoint is not repeated.

### 9. Per-component errors can't be combined into a velocity error — Fixed
- **What happened:** the evaluation CSVs stored separate relative errors for $u_x$ and $u_y$. A relative error for the velocity vector can't be recovered from those two ratios.
- **Fix:** `scripts/evaluate.py` computes `vel` and `vel_wall` directly, as $\|\mathbf{u}_\text{pred} - \mathbf{u}_\text{true}\| / \|\mathbf{u}_\text{true}\|$ over the fluid region and the wall band.
- **Verification:** with one component set to zero, the result matches `rel_l2` exactly. On every row of every CSV, `vel` lies between `ux` and `uy`, as it must.

### 10. Relative L² is inflated near no-slip walls — Worked around
- **Issue:** velocity goes to zero at the wall, so the denominator of a relative error over a small near-wall region is small. Wall-band and corner relative errors are therefore inflated, and hard to compare between regions and shapes.
- **Workaround:** the corner metric is an absolute RMSE divided by $u_\mathrm{in}$ (`vel_corner_rmse`), not a relative error. The existing `*_wall` relative errors still have this issue and should be read with it in mind.

### 11. Corner measurement region shrinks as ρ grows — Open
- **What happened:** `vel_corner_rmse` is measured over fluid cells within 0.3 $r_0$ of each rounding centre. As the rounding radius ρ grows, the body fills more of that region; for triangles it goes from 280 to 221 cells. Comparisons across ρ are therefore made over different regions.
- **Impact:** with the shrinking region, triangle error looks flat in ρ (0.388 → 0.391). On a region fixed across ρ, it falls by 11% (0.435 → 0.386, seed 0).
- **Planned fix:** define the region as cells that are fluid for every ρ of the same polygon and orientation. See [corner rounding](corner-rounding.md).

### 12. Corner rounding is below the branch input's resolution — Open
- **What happened:** one pixel of the 32 × 64 branch SDF is 0.156 $r_0$ across, about twice the largest rounding radius tested (0.075 $r_0$). Rounding changes the SDF input about 20× less than rotating the shape does. The model's near-corner prediction moves by at most 0.005 $u_\mathrm{in}$ between ρ = 0 and 0.075.
- **Impact:** the polygon grid can't test how sensitive the model is to corner sharpness. The small error drop in #11 comes from the true flow getting smoother, not from the model responding to the rounding.
- **Planned fix:** a new polygon grid with ρ up to about 0.3 $r_0$, which is at least two SDF pixels and 6 LBM cells at s = 2. See [corner rounding](corner-rounding.md).

### 13. Corrupted header in an evaluation CSV — Fixed
- **What happened:** `data/eval/ood_polygrid_s2_n120_seeds.csv` had a stray backtick at the start of its header, so its first column was named `` `seed `` and grouping by `seed` failed. `evaluate.py` doesn't write it, so it was an accidental manual edit.
- **Fix:** the file was regenerated with `scripts.evaluate`. The evaluation CSVs are generated outputs and should only be changed by re-running the evaluation, not by editing them by hand.
