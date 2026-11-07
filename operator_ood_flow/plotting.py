import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D


def plot_heatmap(grid, metric):
    """
    requires a grid, with the heatmap axis groupbed by metric. Example usage in notebooks/plots.ipynb
    """
    fig, ax = plt.subplots(figsize=(6, 2.8))
    sns.heatmap(
        grid,
        ax=ax,
        cmap="Blues",
        annot=True,
        fmt=".3f",
        linewidths=2,
        linecolor="white",
        cbar_kws={"label": f"Relative $L^2$ ({metric})"},
    )
    ax.set_xticklabels([f"{a:.2f}" for a in grid.columns])
    ax.set_xlabel("Lobe amplitude $a$")
    ax.set_ylabel("Lobe count $k$")
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    plt.show()


def plot_corner_rmse(data, metric="vel_corner_rmse"):
    """
    Near-corner error vs rounding radius, one curve per side count. data is the raw
    polygon grid eval CSV; each point is the mean over orientations per seed, then the
    mean over seeds, with a +-1 std band across seeds. Example usage in notebooks/plots.ipynb
    """
    per_seed = data.groupby(["sides", "rho", "seed"])[metric].mean().reset_index()
    names = {3: "Triangle", 4: "Square", 6: "Hexagon"}
    per_seed["shape"] = per_seed["sides"].map(names)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.lineplot(
        data=per_seed,
        x="rho",
        y=metric,
        hue="shape",
        hue_order=list(names.values()),
        palette=["#2a78d6", "#eb6834", "#1baf7a"],  # fixed order: n = 3, 4, 6
        errorbar="sd",
        marker="o",
        markersize=7,
        linewidth=2,
        ax=ax,
    )
    ax.set_xticks(sorted(per_seed["rho"].unique()))
    ax.set_xlabel(r"Corner rounding radius $\rho$ ($r_0$)")
    ax.set_ylabel(r"Near-corner velocity RMSE / $u_\mathrm{in}$")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title="", ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    plt.tight_layout()
    plt.show()


def plot_rounding_sensitivity(rounding, evals):
    """
    Three panels against rounding radius rho, one colour per side count:
    (a) SDF distance from the sharp shape, native (solid) vs branch resolution (dashed);
    (b) change from the sharp shape near the corners, true LBM flow (solid) vs model
        prediction (dashed);
    (c) near-corner prediction error;
    (d) alignment (cosine similarity) between the true and predicted flow changes.
    rounding is data/eval/<dataset>_rounding.csv (scripts.rounding_sensitivity), evals the
    matching *_seeds.csv. Model quantities are averaged per seed, then over seeds (band =
    +-1 std across seeds). Example usage in notebooks/plots.ipynb
    """
    names = {3: "Triangle", 4: "Square", 6: "Hexagon"}
    colors = dict(zip(names, ["#2a78d6", "#eb6834", "#1baf7a"]))  # fixed order: n = 3, 4, 6
    cols = ["sdf_native", "sdf_branch", "flow_change", "pred_change", "alignment"]
    r = rounding.groupby(["sides", "rho", "seed"])[cols].mean().reset_index()
    e = evals.groupby(["sides", "rho", "seed"])["vel_corner_rmse"].mean().reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    axes = axes.flat
    panels = [
        (axes[0], r, [("sdf_native", "-", "native"), ("sdf_branch", "--", "32×64 branch")],
         r"SDF distance $\|z_\rho - z_0\|_2$ ($r_0^2$)", "(a) Input change"),
        (axes[1], r, [("flow_change", "-", "true LBM flow"), ("pred_change", "--", "model prediction")],
         r"Near-corner velocity change / $u_\mathrm{in}$", "(b) Flow vs prediction change"),
        (axes[2], e, [("vel_corner_rmse", "-", None)],
         r"Near-corner velocity RMSE / $u_\mathrm{in}$", "(c) Prediction error"),
        (axes[3], r[r["rho"] > 0], [("alignment", "-", None)],
         r"Alignment $C(\rho)$ of true and predicted change", "(d) Direction of the response"),
    ]
    for ax, data, series, ylabel, title in panels:
        for n, name in names.items():
            for col, style, _ in series:
                sns.lineplot(
                    data=data[data["sides"] == n], x="rho", y=col, ax=ax,
                    color=colors[n], linestyle=style, errorbar="sd",
                    marker="o", markersize=6, linewidth=2,
                )
        ax.set_xticks(sorted(data["rho"].unique()))
        ax.set_xlabel(r"Corner rounding radius $\rho$ ($r_0$)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylim(*((0, 1) if title.startswith("(d)") else (0, None)))
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        # line-style key for the two-series panels
        if len(series) > 1:
            ax.legend(
                [Line2D([], [], color="0.3", linestyle=s, linewidth=2) for _, s, _ in series],
                [label for _, _, label in series], frameon=False, fontsize=8, loc="upper left",
            )

    fig.legend(
        [Line2D([], [], color=colors[n], marker="o", linewidth=2) for n in names],
        list(names.values()), ncol=3, loc="upper center", frameon=False,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()


def plot_response_error(rounding):
    """
    Relative response error ||dq_hat - dq|| / ||dq|| vs rounding radius, one curve per
    side count: how much of the rounding-induced near-corner flow change the model
    misses. The dashed line at 1 is what predicting no change at all would score.
    rounding is data/eval/<dataset>_rounding.csv (scripts.rounding_sensitivity); mean
    over orientations per seed, then over seeds (band = +-1 std across seeds).
    Example usage in notebooks/plots.ipynb
    """
    names = {3: "Triangle", 4: "Square", 6: "Hexagon"}
    r = rounding[rounding["rho"] > 0]
    r = r.groupby(["sides", "rho", "seed"])["response_err"].mean().reset_index()
    r["shape"] = r["sides"].map(names)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.axhline(1, color="0.5", linestyle="--", linewidth=1)
    ax.text(r["rho"].max(), 1.02, "no response", ha="right", va="bottom", fontsize=8, color="0.4")
    sns.lineplot(
        data=r, x="rho", y="response_err", hue="shape", hue_order=list(names.values()),
        palette=["#2a78d6", "#eb6834", "#1baf7a"],  # fixed order: n = 3, 4, 6
        errorbar="sd", marker="o", markersize=7, linewidth=2, ax=ax,
    )
    ax.set_xticks(sorted(r["rho"].unique()))
    ax.set_xlabel(r"Corner rounding radius $\rho$ ($r_0$)")
    ax.set_ylabel(r"Response error $\|\Delta\hat q - \Delta q\| / \|\Delta q\|$")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title="", ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    plt.tight_layout()
    plt.show()
