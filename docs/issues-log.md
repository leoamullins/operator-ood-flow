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
| 11 | Corner measurement region shrinks as ρ grows | Metrics | Fixed |
| 12 | Corner rounding is below the branch input's resolution | Experiment design | Fixed (data); model insensitivity found |
| 13 | Corrupted header in an evaluation CSV | Data handling | Fixed |
| 14 | Corner metric crashed on the lobed grid | Metrics | Fixed |
| 15 | Data-generation doc deleted without being recorded | Documentation | Fixed |
| 16 | Some rounded polygons rasterise identically to their sharp shape | Data | Handled (NaN) |
| 17 | float32 predictions put Δq̂ outside the trunk span | Metrics | Fixed |

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

### 11. Corner measurement region shrinks as ρ grows — Fixed
- **What happened:** `vel_corner_rmse` is measured over fluid cells within 0.3 $r_0$ of each rounding centre. As the rounding radius ρ grows, the body fills more of that region; for triangles it goes from 280 to 221 cells. Comparisons across ρ are therefore made over different regions.
- **Impact:** with the shrinking region, triangle error looks flat in ρ (0.388 → 0.391). On a region fixed across ρ, it falls by 11% (0.435 → 0.386, seed 0).
- **Worse on the wider grid:** on `ood_polygrid_s2_n180`, ρ goes up to 0.3. The region (radius 0.3 $r_0$ around the rounding centre) falls from 280 to 101 cells for triangles at ρ = 0.2. At ρ = 0.3 it is empty, because the curved edge sits exactly 0.3 $r_0$ from the centre, so `vel_corner_rmse` is NaN there. The other columns were unaffected.
- **Fix:** `corner_mask` in `scripts/evaluate.py` now anchors the region to the vertices of the sharp (ρ = 0) polygon, which don't move with ρ, with `CORNER_R` raised to 0.5 $r_0$. It keeps only cells that are fluid for every ρ of the same side count and orientation.
- **Verification:** within each polygon and orientation, the region is identical across all ρ values: about 780 cells for triangles, 910–970 for squares and 1210–1250 for hexagons. Vertex regions don't overlap, since the closest vertices (hexagon) are 1.1 $r_0$ apart. There are no NaNs. The ρ = 0 values agree between the two polygon grids, which contain the same sharp shapes. Both polygon evaluations were re-run, and their other columns were unchanged.
- **Effect on results:** these numbers come from a different region, so they replace all earlier `vel_corner_rmse` values. Triangle error now falls clearly with rounding, from 0.324 at ρ = 0 to 0.135 at ρ = 0.3 (mean of 5 seeds, std across seeds ≤ 0.006). See [findings](findings.md).

### 12. Corner rounding is below the branch input's resolution — Fixed (data); model insensitivity found
- **What happened:** one pixel of the 32 × 64 branch SDF is 0.156 $r_0$ across, about twice the largest rounding radius tested (0.075 $r_0$). Rounding changes the SDF input about 20× less than rotating the shape does. The model's near-corner prediction moves by at most 0.005 $u_\mathrm{in}$ between ρ = 0 and 0.075.
- **Impact:** the polygon grid can't test how sensitive the model is to corner sharpness. The small error drop in #11 comes from the true flow getting smoother, not from the model responding to the rounding.
- **Partial fix:** a new polygon grid, `ood_polygrid_s2_n180`, extends ρ to 0.3 $r_0$, which is about 2 SDF pixels and 6 LBM cells at s = 2. It has ρ ∈ {0, 0.05, 0.1, 0.15, 0.2, 0.3} for 3, 4 and 6 sides at 10 orientations, generated from commit `5e83718` with `notebooks/generate_polygrid_rho.ipynb`.
- **Check on the wider grid** (`scripts/rounding_sensitivity.py`, all 5 seeds, written to `data/eval/ood_polygrid_s2_n180_rounding.csv`): the change from each sharp shape to its rounded version, for triangles at ρ = 0.3:
  - **SDF input:** 0.787 $r_0^2$ at native resolution and 0.786 at the 32 × 64 branch resolution (both are $\|z_\rho - z_0\|_2$ weighted by pixel area). Every side count and ρ agrees to within 1%.
  - **True near-corner flow:** changes by 0.232 $u_\mathrm{in}$.
  - **Model's near-corner prediction:** changes by only 0.019 $u_\mathrm{in}$, about 12× less.
- **Conclusion:** at this range, the input resolution is not the bottleneck, at least by this global measure. The 32 × 64 input preserves the rounding change, but the model barely responds to it. The falling near-corner error (#11) comes from the true flow moving toward the model's nearly fixed prediction, not from the model tracking the geometry.
- **Caveat:** the global SDF L² is dominated by smooth, far-reaching changes, because keeping the area fixed moves the whole boundary. It may therefore miss whether fine detail at the corner itself survives downsampling. Comparing the SDFs only within the corner region would be a stricter test. See [findings](findings.md).

### 13. Corrupted header in an evaluation CSV — Fixed
- **What happened:** `data/eval/ood_polygrid_s2_n120_seeds.csv` had a stray backtick at the start of its header, so its first column was named `` `seed `` and grouping by `seed` failed. `evaluate.py` doesn't write it, so it was an accidental manual edit.
- **Fix:** the file was regenerated with `scripts.evaluate`. The evaluation CSVs are generated outputs and should only be changed by re-running the evaluation, not by editing them by hand.

### 14. Corner metric crashed on the lobed grid — Fixed
- **What happened:** `corner_mask` in `scripts/evaluate.py` treated any dataset that saves `phi` as a polygon grid. The lobed grid also saves `phi` (its orientation), so evaluating `ood_lobedgrid_s2_n180` failed with `KeyError: 'sides is not a file in the archive'`.
- **Impact:** evaluation stopped with an error, so no results were wrong.
- **Fix:** the check now requires `sides`, `rho` and `phi` together. The lobed grid was re-evaluated with `vel_corner_rmse` = NaN, and its existing columns were unchanged. The polygon grid still gets a corner mask, and the in-distribution test set still doesn't.

### 15. Data-generation doc deleted without being recorded — Fixed
- **What happened:** commit `60c27ff` deleted `docs/data.md` (Gaussian random functions, the convex sampler, the solver, dataset generation), although its commit message describes only removing the plot scripts, figures and results pages. The README kept linking to it.
- **Fix:** its content was recovered from `60c27ff^`, checked against the current code and folded into [methodology](methodology.md) (§2–3) and [theory](theory.md) (§3). The 2026-11-07 docs restructure replaced all the old pages with methodology, findings and theory.

### 16. Some rounded polygons rasterise identically to their sharp shape — Handled (NaN)
- **What happened:** in `ood_polygrid_s2_n180`, 16 of the 150 rounded shapes (all at ρ = 0.05, e.g. squares at several orientations) give exactly the same LBM fields as their ρ = 0 shape, so Δq = 0. `response_decomposition` in `scripts/projection_experiment.py` raised `ValueError: True flow response is nearly zero` on the first one.
- **Impact:** there is no response to decompose for these shapes, and their ρ = 0.05 averages come from fewer shapes. `rounding_sensitivity.py` already gives NaN for them.
- **Fix:** the projection run catches the error and writes NaN for these rows. Unverified: the likely cause is that a 0.05 $r_0$ rounding (0.15 LBM cells at s = 2) doesn't change which cells are solid.

### 17. float32 predictions put Δq̂ outside the trunk span — Fixed
- **What happened:** the trunk-span projection requires $E_\mathrm{total}^2 = E_\mathrm{trunk}^2 + E_\mathrm{branch}^2$, which holds only if Δq̂ lies exactly in the trunk span. With float32 inputs the identity was off by up to 0.014. Moving the least squares to float64 only reduced this to 1.3e-5, and changing `rcond` had no effect, so the basis's conditioning was not the cause (this was first misdiagnosed as conditioning).
- **Cause:** catastrophic cancellation. Δq̂ is only about 1% of q̂, so computing q̂_ρ − q̂_0 from float32 model outputs leaves a ~1e-4 relative error, and that error lies outside the span. With the model run in float64, Δq̂ lies in the span to ~1e-13 (checked per field on seed 0).
- **Fix:** `scripts/experiment_1.py` and `scripts/experiment_1_responses.py` now run the model in float64 and save float64 trunk bases and Δq̂. The largest identity residual is now 2e-10 across all seeds, shapes, regions and fields. The velocity response error on the corner region matches `ood_polygrid_s2_n180_rounding.csv` to 1.8e-4; the remaining difference is the float32 error in that CSV.
