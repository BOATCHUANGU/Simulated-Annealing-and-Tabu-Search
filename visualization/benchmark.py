"""任务三：收敛曲线与最优路径图。"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
from config.config import DATA_DIR
from problems.vrptw import VRPTW
from experiments.statistics import load_runs


def proper_intersections(problem, routes):
    segments = []
    for route in routes:
        path = [0] + route + [0]
        segments += list(zip(path, path[1:]))
    def cross(a, b, c):
        return (b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x)
    count = 0
    for i, (a, b) in enumerate(segments):
        for c, d in segments[i+1:]:
            if len({a,b,c,d}) < 4:
                continue
            p, q, r, s = (problem.customers[x] for x in (a,b,c,d))
            if cross(p,q,r)*cross(p,q,s) < -1e-9 and cross(r,s,p)*cross(r,s,q) < -1e-9:
                count += 1
    return count


def make_figures(output):
    output = Path(output)
    records = load_runs(output)
    if not records:
        return
    manifest = json.loads((output / "config.json").read_text(encoding="utf-8"))
    instance = manifest["instances"][0]
    selected = [r for r in records if r["instance"] == instance]
    best = min(selected, key=lambda r: (r["vehicles"], r["distance"]))
    seed = best["seed"]
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, constrained_layout=True)
    for row in selected:
        if row["seed"] != seed:
            continue
        axes[0].plot([p[0] for p in row["history"]], label=row["algorithm"])
        axes[1].plot([p[1] for p in row["history"]], label=row["algorithm"])
    axes[0].set(ylabel="Vehicles", title=f"{instance} convergence, seed {seed}")
    axes[1].set(xlabel="Iteration", ylabel="Distance")
    axes[0].legend()
    fig.savefig(output / "convergence.png", dpi=150)
    plt.close(fig)

    problem = VRPTW.from_solomon(DATA_DIR / (instance + ".txt"), manifest["customers"])
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    colors = plt.get_cmap("turbo")
    for i, route in enumerate(best["routes"]):
        nodes = [0] + route + [0]
        ax.plot([problem.customers[n].x for n in nodes], [problem.customers[n].y for n in nodes],
                marker=".", linewidth=1, color=colors(i/max(1, len(best["routes"])-1)))
    depot = problem.customers[0]
    ax.scatter([depot.x], [depot.y], color="black", marker="s", s=50, label="Depot", zorder=5)
    ax.set(title=f"{instance} best: {best['algorithm']}, {best['vehicles']} vehicles, {best['distance']:.2f}", xlabel="X", ylabel="Y", aspect="equal")
    ax.legend()
    fig.savefig(output / "best_route.png", dpi=150)
    plt.close(fig)
    details = {"instance": instance, "algorithm": best["algorithm"], "seed": best["seed"],
               "vehicles": best["vehicles"], "distance": best["distance"],
               "proper_crossings": proper_intersections(problem, best["routes"]),
               "crossing_definition": "strict geometric crossings; shared endpoints and collinear overlaps excluded"}
    (output / "best_route.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
