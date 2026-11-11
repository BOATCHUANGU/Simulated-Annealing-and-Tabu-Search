"""任务四：参数敏感性与邻域对比。实验不修改算法实现。"""
import argparse
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean, stdev
from algorithms.common import SearchConfig
from config.config import DATA_DIR, RESULTS_DIR
from experiments.runner import run, prepare_output, append_record
from experiments.statistics import write_csv
from problems.vrptw import VRPTW

# 每次仅改变一个变量，基准配置写入 config.json。
CASES = (
    ("tabu_tenure", "5", {"tabu_tenure": 5}),
    ("tabu_tenure", "10", {"tabu_tenure": 10}),
    ("tabu_tenure", "20", {"tabu_tenure": 20}),
    ("candidate_count", "10", {"candidate_count": 10}),
    ("candidate_count", "30", {"candidate_count": 30}),
    ("candidate_count", "60", {"candidate_count": 60}),
    ("neighborhoods", "relocate", {"neighborhoods": ("relocate",)}),
    ("neighborhoods", "relocate+swap", {"neighborhoods": ("relocate", "swap")}),
    ("neighborhoods", "all", {"neighborhoods": ("relocate", "swap", "2opt")}),
)


def run_study(output, instance, algorithm, repetitions, base, customers=100):
    if repetitions < 1 or customers < 1:
        raise ValueError("重复次数和客户数量必须为正")
    base.validate()
    output = Path(output)
    manifest = {"instance": instance, "algorithm": algorithm, "repetitions": repetitions,
                "customers": customers, "base": asdict(base), "cases": CASES}
    raw, records = prepare_output(output, manifest)
    completed = {(r["factor"], r["level"], r["seed"]) for r in records}
    problem = VRPTW.from_solomon(DATA_DIR / (instance + ".txt"), customers)
    for factor, level, changes in CASES:
        for seed in range(base.seed, base.seed + repetitions):
            if (factor, level, seed) in completed:
                continue
            config = replace(base, seed=seed, **changes)
            result = run(problem, algorithm, config)
            check = problem.evaluate(result.solution.routes)
            if not check.feasible or abs(check.distance-result.distance) > 1e-6:
                raise RuntimeError(f"无效解: {factor}={level} seed={seed}")
            record = {"factor": factor, "level": level, "seed": seed,
                      "vehicles": result.vehicles, "distance": result.distance,
                      "runtime": result.runtime, "iterations": result.iterations,
                      "history": result.history}
            append_record(raw, record)
            records.append(record)
            print(f"{factor}={level} seed={seed}: {result.vehicles} vehicles, {result.distance:.2f}, {result.runtime:.2f}s", flush=True)
    return records


def summarize(output, records):
    """距离也按车辆数分组，避免把不同车辆数当成同一比较口径。"""
    groups = defaultdict(list)
    for r in records:
        groups[r["factor"], r["level"]].append(r)
    summary, by_vehicle = [], []
    for factor, level, _ in CASES:
        rows = groups.get((factor, level), [])
        if not rows:
            continue
        summary.append({"factor": factor, "level": level, "runs": len(rows),
                        "best_vehicles": min(r["vehicles"] for r in rows),
                        "mean_distance_all": mean(r["distance"] for r in rows),
                        "mean_runtime": mean(r["runtime"] for r in rows),
                        "mean_iterations": mean(r["iterations"] for r in rows),
                        "vehicle_counts": ";".join(f"{v}:{sum(r['vehicles']==v for r in rows)}" for v in sorted({r["vehicles"] for r in rows}))})
        for vehicles in sorted({r["vehicles"] for r in rows}):
            distances = [r["distance"] for r in rows if r["vehicles"] == vehicles]
            by_vehicle.append({"factor": factor, "level": level, "vehicles": vehicles,
                               "runs": len(distances), "mean_distance": mean(distances),
                               "std_distance": stdev(distances) if len(distances) > 1 else ""})
    output = Path(output)
    write_csv(output / "summary.csv", ["factor", "level", "runs", "best_vehicles", "mean_distance_all", "mean_runtime", "mean_iterations", "vehicle_counts"], summary)
    write_csv(output / "distance_by_vehicles.csv", ["factor", "level", "vehicles", "runs", "mean_distance", "std_distance"], by_vehicle)
    return summary, by_vehicle


def main():
    parser = argparse.ArgumentParser(description="R101/R104 参数敏感性及邻域实验")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "sensitivity")
    parser.add_argument("--instance", choices=("R101", "R104"), default="R101")
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--customers", type=int, default=100)
    parser.add_argument("--benchmark", type=Path, default=RESULTS_DIR / "benchmark_pilot", help="任务三结果目录，用于行为分析")
    args = parser.parse_args()
    config = SearchConfig(seed=args.seed, iterations=args.iterations, time_limit=args.seconds)
    records = run_study(args.out, args.instance, "hybrid1", args.repetitions, config, args.customers)
    summary, groups = summarize(args.out, records)
    from visualization.sensitivity import plot_study
    plot_study(args.out, summary, groups, args.instance)
    if (args.benchmark / "runs.jsonl").exists():
        from experiments.statistics import load_runs
        benchmark_runs = load_runs(args.benchmark)
        if any(r["instance"] == args.instance for r in benchmark_runs):
            from experiments.behavior import analyze_behavior
            analyze_behavior(args.benchmark, args.out, args.instance, args.seed)
        else:
            print("任务三结果不包含该实例，跳过行为对比。")
    else:
        print("未找到任务三结果，跳过行为对比。")


if __name__ == "__main__":
    main()
