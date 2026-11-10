"""任务一：生成实例分布图并验证 C101 十客户子实例。"""
import argparse
from config.config import DATA_DIR, RESULTS_DIR
from algorithms.common import SearchConfig
from experiments.runner import run, SOLVERS
from problems.vrptw import VRPTW
from experiments.verify import exact_check
from visualization.distribution import plot_distribution


def main():
    parser = argparse.ArgumentParser(description="VRPTW 任务一验证与任务二算法演示")
    parser.add_argument("--algorithm", choices=SOLVERS, help="不指定时运行任务一验证")
    parser.add_argument("--instance", default="C101", help="Solomon 实例名")
    parser.add_argument("--customers", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.algorithm:
        problem = VRPTW.from_solomon(DATA_DIR / (args.instance + ".txt"), args.customers)
        result = run(problem, args.algorithm, SearchConfig(seed=args.seed, iterations=args.iterations))
        print(f"{args.algorithm}: {result.vehicles} 辆, 距离 {result.distance:.6f}, 耗时 {result.runtime:.3f} 秒")
        for route in result.solution.routes:
            print("0 -> " + " -> ".join(map(str, route)) + " -> 0")
        return
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
