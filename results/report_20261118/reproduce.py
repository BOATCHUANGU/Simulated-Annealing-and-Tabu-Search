
import sys, json, os, platform
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
sys.path.insert(0, os.getcwd())
from config.config import DATA_DIR
from algorithms.common import SearchConfig
from experiments.runner import SOLVERS, run
from experiments.benchmark import INSTANCES
from experiments.sensitivity import CASES
from problems.vrptw import VRPTW

def improvements(history_times, history, runtime):
    """Retain the timestamps needed to reproduce each incumbent step curve."""
    trace = []
    for t, (k, d) in zip(history_times, history):
        if not trace or (k, d) != tuple(trace[-1][1:]):
            trace.append((t, k, d))
    if runtime > trace[-1][0]:
        trace.append((runtime, *trace[-1][1:]))
    return trace

def job(item):
    study, instance, algorithm, seed, seconds, factor, level, settings = item
    p = VRPTW.from_solomon(DATA_DIR / (instance + ".txt"), 100)
    r = run(p, algorithm, SearchConfig(seed=seed, iterations=100000, time_limit=seconds, **settings))
    checked = p.evaluate(r.solution.routes)
    assert checked.feasible and abs(checked.distance - r.distance) < 1e-8
    assert len(r.history) == r.iterations + 1 == len(r.history_times)
    return dict(study=study, instance=instance, algorithm=algorithm, seed=seed, seconds=seconds,
                factor=factor, level=level, settings=settings, vehicles=r.vehicles, distance=r.distance,
                runtime=r.runtime, iterations=r.iterations, stop_reason=r.stop_reason,
                initial_solutions=r.initial_solutions, routes=r.solution.routes,
                history=improvements(r.history_times,r.history,r.runtime))

if __name__ == "__main__":
    out=Path(sys.argv[1])
    out.mkdir(exist_ok=True, parents=True)
    seeds=(0,8,24)
    tasks=[("benchmark",instance,a,s,10,None,None,{}) for instance,a,s in product(INSTANCES,SOLVERS,seeds)]
    cases=list(CASES)+[("neighborhoods", "updated", {})]
    tasks.extend(("sensitivity","R101","hybrid3",s,2,f,l,params) for (f,l,params),s in product(cases,seeds))
    # Interleave job groups for wall time sampling across the batch.
    import random
    random.Random(18).shuffle(tasks)
    with ProcessPoolExecutor(max_workers=8) as pool, (out/"runs.jsonl").open("x",encoding="utf8") as stream:
        for j, result in enumerate(pool.map(job,tasks,chunksize=1),1):
            stream.write(json.dumps(result, ensure_ascii=False)+"\n")
            stream.flush()
            print(j,result["study"],result["instance"],result["algorithm"],result["seed"],result["vehicles"],round(result["distance"],2),flush=True)
    (out/"config.json").write_text(json.dumps(dict(python=sys.version,platform=platform.platform(),workers=8,seeds=seeds,iterations=100000,benchmark_seconds=10,sensitivity_seconds=2,instances=INSTANCES,algorithms=list(SOLVERS),sensitivity_cases=cases),ensure_ascii=False,indent=2),encoding="utf8")

