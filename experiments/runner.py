"""五套算法的统一调用入口。"""
from algorithms import sa, ts, hybrid

SOLVERS = {
    "sa": sa.solve,
    "ts": ts.solve,
    "hybrid1": hybrid.ts_metropolis,
    "hybrid2": hybrid.sa_ts_inner,
    "hybrid3": hybrid.sa_warm_ts,
}


def run(problem, algorithm, parameters=None):
    if algorithm not in SOLVERS:
        raise ValueError(f"未知算法: {algorithm}")
    return SOLVERS[algorithm](problem, parameters)
