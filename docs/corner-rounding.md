# Corner rounding: near-corner error vs rounding radius

[← Back to README](../README.md)

**Status:** preliminary (2026-11-07). The rounding range tested is too small to count as a real shift; see [Next steps](#next-steps).

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
