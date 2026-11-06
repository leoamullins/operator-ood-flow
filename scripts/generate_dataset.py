import argparse

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
        "--config", help="rebuild from a saved dataset .json (ignores other flags)"
    )
    parser.add_argument("--out", help="name in data/ (default: train_<sampler>)")
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
            shape_kw=shape_kw,
        )

    gen.run(args.out or f"train_{gen.sampler}")
