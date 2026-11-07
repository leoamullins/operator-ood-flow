import matplotlib.pyplot as plt
import seaborn as sns


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
