# Corner rounding: near-corner error vs rounding radius

[← Back to README](../README.md)

**Status:** preliminary (2026-11-07). The first grid's rounding range was too small to count as a real shift, and its corner region shrank as ρ grew. Both have been fixed; see [Update: fixed region and wider ρ range](#update-fixed-region-and-wider-ρ-range). The original analysis below is kept for the record, but its numbers are superseded.

## Update: fixed region and wider ρ range

- **Metric (current):** RMSE of the velocity vector divided by $u_\mathrm{in}$, over fluid cells within `CORNER_R` = 0.5 $r_0$ of the **sharp** polygon's vertices. Only cells that are fluid for every ρ of the same polygon and orientation are used, so every ρ is scored on identical cells ([issues log #11](issues-log.md)).
- **Data:** `ood_polygrid_s2_n120` (ρ up to 0.075) and `ood_polygrid_s2_n180` (ρ ∈ {0, 0.05, 0.1, 0.15, 0.2, 0.3}). Both have 3, 4 and 6 sides at 10 orientations.

Near-corner velocity RMSE / $u_\mathrm{in}$ on `ood_polygrid_s2_n180`, mean over 5 seeds (std across seeds ≤ 0.006):

| | ρ = 0 | 0.05 | 0.1 | 0.15 | 0.2 | 0.3 |
|---|---|---|---|---|---|---|
| Triangle | 0.324 | 0.305 | 0.270 | 0.235 | 0.202 | 0.135 |
| Square | 0.164 | 0.161 | 0.144 | 0.132 | 0.123 | 0.100 |
| Hexagon | 0.053 | 0.049 | 0.049 | 0.047 | 0.046 | 0.039 |

The ρ = 0 values match those from `ood_polygrid_s2_n120`, which contains the same sharp shapes.

- **Rounding now lowers the error clearly.** Triangle error falls by 58% from ρ = 0 to 0.3, square by 39% and hexagon by 26%. Rounding by ρ = 0.3 matters about as much as one extra side: a triangle rounded with ρ = 0.3 (0.135) is better than a sharp square (0.164), but still worse than a square rounded with ρ = 0.3 (0.100).
- **The drop comes from the true flow, not from the model.** Rounding a triangle to ρ = 0.3 changes the true near-corner flow by 0.232 $u_\mathrm{in}$, but the model's prediction by only 0.019 $u_\mathrm{in}$. Meanwhile the 32 × 64 branch input keeps the full SDF change (0.786 vs 0.787 $r_0^2$ at native resolution). So the model receives a clearly different input and barely responds to it. This suggests the limit is the model's insensitivity to the input rather than the input's resolution ([issues log #12](issues-log.md), including a caveat about the global SDF norm).
- **The model responds in roughly the right direction, but far too weakly.** The alignment $C(\rho) = \langle \Delta q, \Delta\hat q\rangle / (\|\Delta q\|\|\Delta\hat q\|)$ between the true and predicted near-corner velocity changes is 0.46–0.64 for every side count and ρ (std across seeds ≤ 0.05). That is clearly positive, though some individual shapes go as low as −0.39. The component of the response along the true change, $C\,\|\Delta\hat q\| / \|\Delta q\|$, is about 0.04 for triangles at ρ = 0.3, 0.05 for squares and 0.18 for hexagons. So the model reproduces only about 4–18% of the true change. This looks like a damped response rather than a wrong one, which is relevant to the spectral-bias (NTK) phase of the [experiment design](experiment_design.md).
- **Figure:** `plotting.plot_rounding_sensitivity` shows (a) the SDF change at native and branch resolution, (b) the true flow change vs the prediction change, (c) the near-corner error and (d) the alignment $C(\rho)$, all against ρ. The data comes from `uv run python -m scripts.rounding_sensitivity ood_polygrid_s2_n180`.

## Setup

- **Data:** `ood_polygrid_s2_n120`, made of regular triangles, squares and hexagons. Corners are rounded with radius $\rho \in \{0, 0.025, 0.05, 0.075\}\,r_0$ at 10 orientations each, and every shape has area $\pi r_0^2$. The LBM runs at s = 2, so $r_0 = 20$ cells.
- **Models:** the five s = 2 seeds (0, 123, 456, 789, 2026).
- **Metric:** `vel_corner_rmse` in `scripts/evaluate.py`. It is the RMSE of the velocity vector $(u_x, u_y)$ divided by $u_\mathrm{in}$, over fluid cells within `CORNER_R` = 0.3 $r_0$ of a rounding centre (`geometry.polygon_corners`).
- **Figure:** `plotting.plot_corner_rmse`, one curve per side count with a ±1 std band across seeds.

## Result

Near-corner velocity RMSE / $u_\mathrm{in}$, mean over 5 seeds (std across seeds ≤ 0.008):

| | ρ = 0 | 0.025 | 0.05 | 0.075 |
|---|---|---|---|---|
| Triangle | 0.388 | 0.397 | 0.395 | 0.391 |
| Square | 0.206 | 0.208 | 0.214 | 0.208 |
| Hexagon | 0.071 | 0.066 | 0.068 | 0.070 |

**The number of sides sets the error, and rounding over this range does not change it.** From triangle to square the error roughly halves, and from square to hexagon it falls about 3× more.

## Why the curves are flat

These checks used the seed 0 model only. Each value is the mean over the 10 orientations.

| ρ = 0 → 0.075 | Triangle | Square | Hexagon |
|---|---|---|---|
| Near-corner RMSE / $u_\mathrm{in}$, fixed region | 0.435 → 0.386 | 0.227 → 0.209 | 0.077 → 0.070 |
| Change in the model's prediction (RMSE / $u_\mathrm{in}$) | 0.004 | 0.002 | 0.005 |
| Change in the true flow (RMSE / $u_\mathrm{in}$) | 0.069 | 0.025 | 0.017 |
| Change in the SDF input: rounding vs rotating the shape | 0.85 vs 16 | 0.49 vs 7.7 | 0.45 vs 2.8 |

The "fixed region" uses the same cells at every ρ: those within 0.3 $r_0$ of a corner that are fluid at all four ρ values.

1. **The measurement region shrinks with ρ, and that hid a small trend.** The rounded body fills more of the region, so for triangles it goes from 280 to 221 fluid cells. On a fixed region, triangle error falls by 11% as ρ goes from 0 to 0.075.
2. **The model ignores the rounding.** One pixel of the 32 × 64 branch SDF is 0.156 $r_0$ across, about twice the largest ρ. Rounding changes the input about 20× less than rotating the shape does, and the prediction moves by at most 0.005 $u_\mathrm{in}$. So the fall seen in (1) isn't the model getting better: the true flow becomes smoother and moves toward the model's already smooth prediction.
3. **The rounding radii are too small to matter physically.** At ρ = 0.075 the true near-corner flow changes by at most 0.07 $u_\mathrm{in}$, well below the triangle error of about 0.4.
4. **Corner angle vs whole-body shape is unresolved.** Side count changes both the corner angle and the overall body shape, so this grid can't separate the two.

## Next steps

- **Fix the region:** `corner_mask` should use cells that are fluid for every ρ of the same polygon and orientation, so the region doesn't depend on ρ.
- **Extend ρ to about 0.3 $r_0$:** that's at least two branch-SDF pixels and 6 LBM cells. It needs a new `polygon_grid` dataset, but the s = 2 solver is enough.
- **Separate angle from shape:** for example, use shapes with one sharp corner on an otherwise smooth convex body.
