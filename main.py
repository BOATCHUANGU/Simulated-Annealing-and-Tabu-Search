"""任务一：生成实例分布图并验证 C101 十客户子实例。"""
import argparse
import json
from pathlib import Path
from config.config import DATA_DIR, RESULTS_DIR
from algorithms.common import SearchConfig
from experiments.runner import run, SOLVERS
from problems.vrptw import VRPTW


def main():
    parser = argparse.ArgumentParser(description="VRPTW 任务一验证与任务二算法演示")
    parser.add_argument("--algorithm", choices=SOLVERS, help="不指定时运行任务一验证")
    parser.add_argument("--instance", default="C101", help="Solomon 实例名")
    parser.add_argument("--customers", type=int, default=10)
    parser.add_argument("--iterations", type=int, help="迭代上限，不限时默认 200")
    parser.add_argument("--fleet-repair", action="store_true", help="停滞时用软约束退火尝试减车")
    parser.add_argument("--target-vehicles", type=int, help="达到目标车数后优先缩短路线")
    parser.add_argument("--target-distance", type=float, help="达到指定车辆数和距离时提前结束")
    parser.add_argument("--output", type=Path, help="将结果保存为 JSON")
    parser.add_argument("--seconds", type=float, help="墙钟时间上限（秒）")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.algorithm:
        problem = VRPTW.from_solomon(DATA_DIR / (args.instance + ".txt"), args.customers)
        iterations = args.iterations
        if iterations is None:
            iterations = 200 if args.seconds is None else 2**31 - 1
        result = run(problem, args.algorithm,
                     SearchConfig(seed=args.seed, iterations=iterations, time_limit=args.seconds,
                                  fleet_repair=args.fleet_repair, target_vehicles=args.target_vehicles,
                                  target_distance=args.target_distance))
        print(f"{args.algorithm}: {result.vehicles} 辆, 距离 {result.distance:.6f}, 耗时 {result.runtime:.3f} 秒")
        print(f"迭代 {result.iterations}，停止原因 {result.stop_reason}")
        for method, (vehicles, distance, feasible) in result.initial_solutions.items():
            print(f"初始解 {method}：{vehicles} 辆，距离 {distance:.6f}，可行 {feasible}")
        for route in result.solution.routes:
            print("0 -> " + " -> ".join(map(str, route)) + " -> 0")
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "instance": args.instance, "customers": args.customers, "algorithm": args.algorithm,
                "seed": args.seed, "time_limit": args.seconds, "iteration_limit": iterations,
                "fleet_repair": args.fleet_repair, "target_vehicles": args.target_vehicles,
                "target_distance": args.target_distance, "vehicles": result.vehicles,
                "distance": result.distance, "runtime": result.runtime, "iterations": result.iterations,
                "stop_reason": result.stop_reason, "initial_solutions": result.initial_solutions,
                "routes": result.solution.routes,
            }
            args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return
    from experiments.verify import exact_check
    from visualization.distribution import plot_distribution

    RESULTS_DIR.mkdir(exist_ok=True)
    output = RESULTS_DIR / "distribution.png"
    plot_distribution(DATA_DIR, output)
    count, distance, routes, total = exact_check(DATA_DIR / "C101.txt")
    print(f"图表：{output}")
    print(f"C101 前 10 客户：{count} 辆，距离 {distance:.6f}，枚举可行路径 {total} 条")
    for route in routes:
        print("路径：0 -> " + " -> ".join(map(str, route)) + " -> 0")
    print("精确模型和路径评价一致。")


if __name__ == "__main__":
    main()
