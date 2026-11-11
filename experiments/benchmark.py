"""任务三：可续跑的基准实验。python -m experiments.benchmark --help"""
import argparse
from dataclasses import asdict, replace
from pathlib import Path
from config.config import DATA_DIR, RESULTS_DIR
from problems.vrptw import VRPTW
from algorithms.common import SearchConfig
from experiments.runner import run, SOLVERS, prepare_output, append_record

INSTANCES = ("C101", "R101", "RC101", "C201", "R201", "RC201")
REFERENCES = {"C101": (10, 828.94), "R101": (19, 1650.80), "RC101": (14, 1696.95),
              "C201": (3, 591.56), "R201": (4, 1252.37), "RC201": (4, 1406.94)}


def run_benchmark(output, instances, algorithms, repetitions, config, customers=100):
    if repetitions < 1 or customers < 1:
        raise ValueError("重复次数与客户数必须为正")
    config.validate()
    output = Path(output)
    manifest = {"instances": list(instances), "algorithms": list(algorithms), "repetitions": repetitions,
                "customers": customers, "parameters": asdict(config), "seed_range": [config.seed, config.seed + repetitions - 1]}
    raw, records = prepare_output(output, manifest)
    completed = {(r["instance"], r["algorithm"], r["seed"]) for r in records if r.get("status") == "ok"}
    for instance in instances:
        problem = VRPTW.from_solomon(DATA_DIR / (instance + ".txt"), customers)
        for algorithm in algorithms:
            for seed in range(config.seed, config.seed + repetitions):
                if (instance, algorithm, seed) in completed:
                    continue
                result = run(problem, algorithm, replace(config, seed=seed))
                checked = problem.evaluate(result.solution.routes)
                if not checked.feasible or abs(checked.distance-result.distance) > 1e-6:
                    raise RuntimeError(f"无效结果：{instance} {algorithm} {seed}")
                record = {"status": "ok", "instance": instance, "algorithm": algorithm, "seed": seed,
                          "vehicles": result.vehicles, "distance": result.distance, "runtime": result.runtime,
                          "iterations": result.iterations, "routes": result.solution.routes, "history": result.history}
                append_record(raw, record)
                print(f"{instance} {algorithm} seed={seed}: {result.vehicles} vehicles, {result.distance:.2f}, {result.runtime:.2f}s", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description="VRPTW 基准实验；相同输出目录与参数可续跑")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "benchmark")
    parser.add_argument("--instances", nargs="+", choices=INSTANCES, default=list(INSTANCES))
    parser.add_argument("--algorithms", nargs="+", choices=list(SOLVERS), default=list(SOLVERS))
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seconds", type=float, default=1.0, help="每次运行的相同时间上限（秒）")
    parser.add_argument("--candidates", type=int, default=30)
    parser.add_argument("--customers", type=int, default=100)
    args = parser.parse_args()
    config = SearchConfig(seed=args.seed, iterations=args.iterations, time_limit=args.seconds, candidate_count=args.candidates)
    output = run_benchmark(args.out, args.instances, args.algorithms, args.repetitions, config, args.customers)
    from experiments.statistics import summarize
    summarize(output)
    from visualization.benchmark import make_figures
    make_figures(output)


if __name__ == "__main__":
    main()
