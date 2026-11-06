import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import argparse

from generator import Generator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate a flow-past-a-body dataset.")
    parser.add_argument("--sampler", choices=["convex", "lobed"], default="convex")
    parser.add_argument("--amp", type=float, nargs=2, help="lobed: amplitude range")
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
        shape_kw = {"amp": args.amp} if args.amp else {}
        gen = Generator(
            sampler=args.sampler,
            n_shapes=args.n_shapes,
            batch=args.batch,
            seed=args.seed,
            shape_kw=shape_kw,
        )

    gen.run(args.out or f"train_{gen.sampler}")
