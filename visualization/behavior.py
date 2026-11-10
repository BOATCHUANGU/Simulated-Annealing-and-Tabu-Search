"""绘制指定实例和种子的各算法收敛轨迹。"""
import matplotlib.pyplot as plt


def plot_convergence(records, output, instance, seed):
    selected = [r for r in records if r["seed"] == seed]
    if not selected:
        raise ValueError(f"未找到种子 {seed}")
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, constrained_layout=True)
    for r in selected:
        axes[0].plot([h[0] for h in r["history"]], label=r["algorithm"], alpha=0.8)
        axes[1].plot([h[1] for h in r["history"]], label=r["algorithm"], alpha=0.8)
    axes[0].set(title=f"{instance}: convergence, seed {seed}", ylabel="Best vehicles")
    axes[1].set(xlabel="Iteration", ylabel="Distance of best solution")
    axes[0].legend()
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.savefig(output, dpi=150)
    plt.close(fig)
