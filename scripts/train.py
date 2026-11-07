from pathlib import Path

import numpy as np

# ray tune imports
import torch
from ray import tune
from torch.utils.data import DataLoader, TensorDataset

from operator_ood_flow.model import DeepONet
from operator_ood_flow.preprocessing import EVAL_BAND, downsample_sdf, load_data, rel_l2

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data" / "train_convex_s2_n1200_seed0"
OUT = ROOT / "checkpoints"
SEED = 0
N_VAL = 200
SDF_RES = (128, 256)  # branch input resolution (downsampled SDF)
BATCH = 32  # shapes per batch
N_POINTS = 2048  # random query points per batch
EPOCHS = 50
LR, WEIGHT_DECAY = 5e-4, 1e-4
P_DIM, WIDTH = 128, 256

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

search_space = {
    "lr": tune.loguniform(1e-4, 1e-3),
    "width": tune.choice([256, 512]),
    "p_dim": tune.choice([128, 256]),
    "weight_decay": tune.choice([0.0, 1e-4]),
    "epochs": 1000,
}

scheduler = tune.schedulers.ASHAScheduler(
    metric="val_rel", mode="min", max_t=1000, grace_period=50
)


@torch.no_grad()
def evaluate(model, loader, Q, mean, std):
    """Mean relative L2 per field over all fluid cells and over the wall band."""
    model.eval()
    rel, wall = [], []
    for xb, yb, mb, wb in loader:
        xb, yb, mb, wb = xb.to(device), yb.to(device), mb.to(device), wb.to(device)
        pred, true = model(xb, Q) * std + mean, yb * std + mean
        rel.append(rel_l2(pred, true, mb))
        wall.append(rel_l2(pred, true, wb))
    return torch.cat(rel).mean(0).cpu().numpy(), torch.cat(wall).mean(0).cpu().numpy()


def load(data_dir=DATA, seed=SEED):
    """Load, split and normalise a dataset into the tensors training needs."""
    sdf, masks, fields, coords = load_data(data_dir)
    N, H, W = masks.shape
    print(f"{N} shapes, crop {H} x {W}, device {device}")

    perm = np.random.default_rng(seed).permutation(N)
    val_idx, tr_idx = perm[:N_VAL], perm[N_VAL:]

    # normalise each field with training fluid cells only
    f_tr = fields[tr_idx][~masks[tr_idx]]
    mean, std = f_tr.mean(0), f_tr.std(0)

    X = downsample_sdf(sdf, SDF_RES)
    Y = torch.tensor(((fields - mean) / std).reshape(N, H * W, 3), dtype=torch.float32)
    M = torch.tensor(~masks.reshape(N, -1), dtype=torch.float32)
    Wm = M * torch.tensor(np.abs(sdf.reshape(N, -1)) < EVAL_BAND, dtype=torch.float32)

    Q = torch.tensor(coords)
    mean_t = torch.tensor(mean, dtype=torch.float32)
    std_t = torch.tensor(std, dtype=torch.float32)

    return {
        "X": X,
        "Y": Y,
        "M": M,
        "Wm": Wm,
        "Q": Q,
        "mean": mean_t,
        "std": std_t,
        "tr_idx": tr_idx,
        "val_idx": val_idx,
    }


def train(config, data, final=False):
    """Train one DeepONet with config; report val error to Ray each epoch, or with
    final=True print it and save the best checkpoint instead."""
    torch.manual_seed(SEED)

    # loaders from the shared tensors
    ds = lambda idx: TensorDataset(
        data["X"][idx], data["Y"][idx], data["M"][idx], data["Wm"][idx]
    )
    train_loader = DataLoader(
        ds(data["tr_idx"]), batch_size=BATCH, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(ds(data["val_idx"]), batch_size=BATCH)
    Q, mean, std = (data[k].to(device) for k in ("Q", "mean", "std"))
    n_cells = data["Y"].shape[1]

    # model + optimiser from config
    model_cfg = {
        "m": data["X"].shape[1],
        "p": config["p_dim"],
        "width": config["width"],
        "n_out": 3,
    }
    model = DeepONet(**model_cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    best = float("inf")
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        for xb, yb, mb, _ in train_loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            pts = torch.randint(0, n_cells, (N_POINTS,), device=device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                pred = model(xb, Q[pts])
                fluid = mb[:, pts]
                loss = (((pred - yb[:, pts]) ** 2).mean(-1) * fluid).sum() / fluid.sum()

            opt.zero_grad()
            loss.backward()
            opt.step()

        rel, wall = evaluate(model, val_loader, Q, mean, std)
        if not final:
            tune.report(
                {
                    "val_rel": float(rel.mean()),
                    "val_rel_wall": float(wall.mean()),
                    "epoch": epoch,
                }
            )
            continue

        print(
            f"epoch {epoch:4d}  val ux {rel[0]:.3f} uy {rel[1]:.3f} p {rel[2]:.3f}  "
            f"wall {wall[0]:.3f} {wall[1]:.3f} {wall[2]:.3f}"
        )
        if rel.mean() < best:
            best = rel.mean()
            OUT.mkdir(exist_ok=True)
            # keys read by scripts/evaluate.py and the plotting scripts
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_cfg": model_cfg,  # rebuild with DeepONet(**model_cfg)
                    "mean": mean.cpu(),
                    "std": std.cpu(),
                    "data_cfg": {
                        "data": str(DATA.relative_to(ROOT)),
                        "sdf_res": SDF_RES,
                        "tr_idx": data["tr_idx"].tolist(),
                        "val_idx": data["val_idx"].tolist(),
                    },
                    "config": config,
                    "epoch": epoch,
                    "val_rel": rel.tolist(),
                    "val_rel_wall": wall.tolist(),
                },
                OUT / "best.pt",
            )


if __name__ == "__main__":
    data = load()
    tuner = tune.Tuner(
        tune.with_resources(tune.with_parameters(train, data=data), {"gpu": 0.25}),
        param_space=search_space,
        tune_config=tune.TuneConfig(scheduler=scheduler, num_samples=20),
    )
    best = tuner.fit().get_best_result("val_rel", "min")
    print(best.config, best.metrics)
    train(
        best.config, data, final=True
    )  # retrain the winner and save checkpoints/best.pt
