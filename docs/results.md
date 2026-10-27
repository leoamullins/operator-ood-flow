# In-distribution results

[← Back to README](../README.md)

Trained for 1000 epochs with `WALL_FRAC=0.5`. The test set is 200 new convex shapes from the same distribution, generated with a different seed and never used during training or model selection:

| Relative $L^2$ (mean over shapes) | $u_x$ | $u_y$ | $p$ |
|---|---|---|---|
| Test, all fluid cells | 0.011 | 0.034 | 0.013 |
| Test, wall band ($\|\text{sdf}\| < 0.3\,r_0$) | 0.071 | 0.091 | 0.040 |
| Validation, all fluid cells | 0.010 | 0.033 | 0.014 |
| Validation, wall band | 0.067 | 0.089 | 0.041 |

Test error is close to validation error, so choosing the checkpoint by validation error did not overfit the validation shapes.

$u_y$ has the highest relative error mostly because it is close to zero over most of the domain. The wall band is harsher still, because no-slip drives the velocity towards zero there.

![Validation error during training](figures/training_curve.png)

The predicted fields for a typical validation shape (median error) match the LBM closely. What error remains is concentrated in a thin layer at the body surface:

![Median validation shape](figures/median_validation_shape.png)

Error grows steadily as shapes become less convex, even within the training distribution (test set shown; the trend is somewhat stronger on the validation set, with Spearman $\rho$ of −0.60, −0.70 and −0.61):

![Error vs convexity margin](figures/error_vs_convexity.png)

On a non-convex trefoil (convexity margin −2.86, far below the training minimum of −0.2), the model still gets the overall flow right: the wake, the acceleration around the body, and the stagnation pressure. But it smooths over the concave notches, and the error is about 4–5× the test-set median (`ux` 0.048, `uy` 0.154, `p` 0.052):

![Non-convex trefoil](figures/nonconvex_trefoil.png)

## Reproducing

Generate the test set, compute per-shape errors, and rebuild the figures:

```bash
uv run python scripts/generate_dataset.py --seed 1 --n-shapes 200 --out test_convex
uv run python scripts/evaluate.py checkpoints/best.pt test_convex               # -> data/test_convex_eval.csv
uv run python scripts/evaluate.py checkpoints/best.pt train_convex --val        # validation shapes only
uv run python scripts/plot_convexity.py data/test_convex_eval.csv               # -> docs/figures/error_vs_convexity.png
uv run python scripts/plot_trefoil.py checkpoints/best.pt                       # -> docs/figures/nonconvex_trefoil.png
```

`evaluate.py` writes one row per shape: index, convexity margin, and relative $L^2$ for $u_x$, $u_y$ and $p$ over all fluid cells and over the wall band. It takes the SDF resolution, normalisation and validation split from the checkpoint, so it uses exactly the preprocessing the model was trained with. `plot_trefoil.py` solves the trefoil with the LBM, which takes a minute or two on a CPU.

Next: [OOD generalisation: lobed shapes](ood-lobed.md)
