import argparse
from pathlib import Path

from operator_ood_flow.dataset_generation import SAMPLERS, Generator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate a flow-past-a-body dataset.")
    parser.add_argument("--sampler", choices=list(SAMPLERS), default="convex")
    parser.add_argument("--amp", type=float, nargs=2, help="lobed: amplitude range")
    parser.add_argument("--sides", type=int, nargs="+", help="polygon: side counts")
    parser.add_argument("--n-shapes", type=int, default=Generator.n_shapes)
    parser.add_argument("--batch", type=int, default=Generator.batch)
    parser.add_argument("--seed", type=int, default=Generator.seed)
    parser.add_argument(
        "--scale", type=float, default=Generator.s, help="lattice scale s: r0 = 10 s cells"
    )
    parser.add_argument(
        "--max-steps", type=int, default=Generator.max_steps, help="LBM step cap; scale with s"
    )
    parser.add_argument(
        "--config",
        help="build from a dataset config or saved dataset .json (ignores other flags)",
    )
    parser.add_argument(
        "--out", help="name in data/ (default: config file name, else train_<sampler>)"
    )
    args = parser.parse_args()

    if args.config:
        gen = Generator.from_json(args.config)
    else:
        shape_kw = {}
        if args.amp:
            shape_kw["amp"] = args.amp
        if args.sides:
            shape_kw["sides"] = args.sides
        gen = Generator(
            sampler=args.sampler,
            n_shapes=args.n_shapes,
            batch=args.batch,
            seed=args.seed,
            s=args.scale,
            max_steps=args.max_steps,
            shape_kw=shape_kw,
        )

    default_out = Path(args.config).stem if args.config else f"train_{gen.sampler}"
    gen.run(args.out or default_out)
