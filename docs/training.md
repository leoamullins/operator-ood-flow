# Model and training

[← Back to README](../README.md)

## Model

[src/model.py](../src/model.py) implements a DeepONet with three outputs:

- **Branch net:** an MLP on the SDF, downsampled to $32 \times 64$ and flattened.
- **Trunk net:** an MLP on the query coordinates $(x, y)$, in units of $r_0$ relative to the body centre.

The output for each field is the dot product of the branch and trunk features, plus a bias.

## Training

[src/train.py](../src/train.py) works as follows:

- Crops the domain to $[-2r_0, 8r_0] \times [-2.5r_0, 2.5r_0]$ around the body.
- Measures pressure relative to the upstream edge.
- Normalises each field with mean and std from the training fluid cells.
- Holds out 200 shapes for validation.
- Trains with AdamW (peak LR $5 \times 10^{-4}$, gradients clipped to norm 1), linear warmup and cosine decay. Each batch uses 32 shapes and 2048 random query points, and the MSE is masked to fluid cells.
- Optionally oversamples query points near the body: `WALL_FRAC` of each batch's points come from cells within `WALL_BAND` of a training body's surface.
- Every 10 epochs, reports the relative $L^2$ error on the validation set for each field, over all fluid cells and over the wall band ($|\text{sdf}| < 0.3\,r_0$).

```bash
EPOCHS=1000 WALL_FRAC=0.5 uv run python src/train.py
```

## Checkpoints

The best model is saved to `checkpoints/best.pt`, together with its model config, normalisation stats, data config and train/val split. The loss history goes to `checkpoints/history.json`. To reload the model:

```python
ckpt = torch.load("checkpoints/best.pt")
model = DeepONet(**ckpt["model_cfg"])
model.load_state_dict(ckpt["model"])
```

Next: [In-distribution results](results.md)
