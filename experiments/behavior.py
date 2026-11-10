"""利用任务三原始记录分析历史最优更新和计算开销。"""
from collections import defaultdict
from pathlib import Path
from statistics import mean
from experiments.statistics import load_runs, write_csv


def analyze_behavior(benchmark_dir, output, instance="R101", seed=0):
    records = [r for r in load_runs(benchmark_dir) if r["instance"] == instance]
    if not records:
        raise ValueError(f"任务三结果中没有 {instance}")
    groups = defaultdict(list)
    for record in records:
        groups[record["algorithm"]].append(record)
    rows = []
    for algorithm, runs in sorted(groups.items()):
        updates = [sum(a != b for a, b in zip(r["history"], r["history"][1:])) for r in runs]
        rows.append({"instance": instance, "algorithm": algorithm, "runs": len(runs),
                     "improved_runs": sum(count > 0 for count in updates),
                     "mean_best_updates": mean(updates),
                     "best_vehicles": min(r["vehicles"] for r in runs),
                     "mean_runtime": mean(r["runtime"] for r in runs),
                     "mean_iterations": mean(r["iterations"] for r in runs)})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "behavior.csv", list(rows[0]), rows)
    from visualization.behavior import plot_convergence
    available = {r["seed"] for r in records}
    plot_convergence(records, output / "convergence.png", instance, seed if seed in available else min(available))
    return rows
