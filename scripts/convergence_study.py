from operator_ood_flow.dataset_generation import Generator

SETS = {
    "convex": {"sampler": "convex", "n_shapes": 20, "seed": 10},
    "lobed": {
        "sampler": "lobed",
        "n_shapes": 20,
        "seed": 11,
        "shape_kw": {"amp": [0.1, 0.4], "lobes": [5]},
    },
    "polygon": {
        "sampler": "polygon",
        "n_shapes": 10,
        "seed": 12,
        "shape_kw": {"sides": [3, 8]},
    },
}
for s in (1.5, 2, 3):
    for fam, kw in SETS.items():
        g = Generator(s=s, max_steps=int(60_000 * s / 1.5), **kw)  # steps scale with r0
        g.run(f"conv_{fam}_s{s}_n{g.n_shapes}_seed{g.seed}")
