"""绘制 Solomon 客户分布图。"""
from pathlib import Path
import matplotlib.pyplot as plt
from problems.vrptw import VRPTW


def plot_distribution(data_dir, output):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, name in zip(axes, ("C101", "R101", "RC101")):
        problem = VRPTW.from_solomon(Path(data_dir) / f"{name}.txt")
        depot = problem.customers[0]
        customers = [c for c in problem.customers.values() if c.id != 0]
        ax.scatter([c.x for c in customers], [c.y for c in customers], s=16, label="Customers")
        ax.scatter(depot.x, depot.y, color="red", marker="s", label="Depot")
        ax.set(title=name, xlabel="X", ylabel="Y", aspect="equal")
        ax.legend()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
