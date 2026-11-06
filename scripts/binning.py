import numpy as np

d = np.genfromtxt("data/ood_lobed_eval.csv", delimiter=",", names=True)
counts, edges = np.histogram(d["margin"], bins=[-30, -3, -1, -0.5, -0.2, 1])
for lo, hi, c in zip(edges[:-1], edges[1:], counts):
    print(f"[{lo:>5}, {hi:>4}): {c}")
