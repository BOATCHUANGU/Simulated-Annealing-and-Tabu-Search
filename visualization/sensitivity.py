"""任务四：参数敏感性与邻域对比图。"""
from pathlib import Path
import matplotlib.pyplot as plt


def draw_factor(axes, summary, groups, factor, title):
    rows = [r for r in summary if r["factor"] == factor]
    if not rows:
        return
    labels = [r["level"] for r in rows]
    x = list(range(len(labels)))
    levels = [r for r in groups if r["factor"] == factor]
    for vehicles in sorted({r["vehicles"] for r in levels}):
        values = {r["level"]: r["mean_distance"] for r in levels if r["vehicles"] == vehicles}
        axes[0].plot(x, [values.get(label, float("nan")) for label in labels], marker="o", label=f"{vehicles} vehicles")
    axes[0].set_ylabel("Mean distance")
    axes[0].set_title(title)
    axes[0].set_xticks(x, labels)
    axes[0].legend()
    axes[1].plot(x, [r["best_vehicles"] for r in rows], marker="o", color="darkred")
    axes[1].set_ylabel("Best vehicle count")
    axes[1].set_xticks(x, labels)
    axes[1].set_xlabel("Level")
    for ax in axes:
        ax.grid(alpha=0.25)


def plot_study(output, summary, groups, instance):
    output = Path(output)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    draw_factor(axes[:, 0], summary, groups, "tabu_tenure", "Tabu tenure")
    draw_factor(axes[:, 1], summary, groups, "candidate_count", "Candidate count")
    fig.suptitle(f"{instance}: parameter sensitivity")
    fig.savefig(output / "parameters.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(7, 7), constrained_layout=True)
    draw_factor(axes, summary, groups, "neighborhoods", "Neighborhood combinations")
    fig.suptitle(instance)
    fig.savefig(output / "neighborhoods.png", dpi=150)
    plt.close(fig)
