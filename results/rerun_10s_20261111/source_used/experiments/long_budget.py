"""Reproducible 10-second rerun with isolated result files and wall-time traces.

Run from the project root: python -m experiments.long_budget
Algorithms retain their original cooling/temperature stops. Thus 10 seconds is
an upper bound, not a promise to keep each algorithm alive for 10 seconds.
"""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_name]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]

def hashes():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for d in ('algorithms','problems','experiments','visualization','config')
            for p in (ROOT/d).glob('*.py')}

def execute(job):
    from algorithms.common import SearchConfig
    from experiments.runner import run
    from config.config import DATA_DIR
    from problems.vrptw import VRPTW
    kind,instance,algorithm,seed,factor,level,changes,seconds=job
    config=SearchConfig(seed=seed,iterations=100000,time_limit=seconds,**changes)
    problem=VRPTW.from_solomon(DATA_DIR/(instance+'.txt'),100)
    result=run(problem,algorithm,config)
    checked=problem.evaluate(result.solution.routes)
    if not checked.feasible or checked.vehicles!=result.vehicles or abs(checked.distance-result.distance)>1e-7:
        raise RuntimeError(f'Invalid solution {job}')
    if len(result.history)!=len(result.history_times) or len(result.history)!=result.iterations+1:
        raise RuntimeError('Trace lengths differ')
    if any(a>b for a,b in zip(result.history_times,result.history_times[1:])):
        raise RuntimeError('Nonmonotone timestamps')
    row={'status':'ok','instance':instance,'algorithm':algorithm,'seed':seed,
         'vehicles':result.vehicles,'distance':result.distance,'runtime':result.runtime,
         'iterations':result.iterations,'routes':result.solution.routes,
         'history':result.history,'history_times':result.history_times,
         'stop_reason':result.stop_reason}
    if kind=='sensitivity':row.update(factor=factor,level=level)
    return kind,row

def identity(kind,row):
    return (kind,row['instance'],row['algorithm'],row['seed'],row.get('factor'),row.get('level'))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=ROOT/'results/rerun_10s_20261111')
    parser.add_argument('--seconds',type=float,default=10.)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    from algorithms.common import SearchConfig
    from experiments.benchmark import INSTANCES
    from experiments.runner import SOLVERS,prepare_output,append_record
    from experiments.sensitivity import CASES
    args.out.mkdir(exist_ok=True,parents=True)
    base=SearchConfig(iterations=100000,time_limit=args.seconds)
    source_hashes=hashes()
    metadata={'python':platform.python_version(),'platform':platform.platform(),
              'logical_cpu_count':os.cpu_count(),'workers':args.workers,
              'time_budget_seconds':args.seconds,'iterations':100000,'seeds':[0,29],
              'cooling_unchanged':True,'source_sha256':source_hashes,
              'parallelism_note':'Eight independent processes by default; per-run wall time includes CPU contention. Prior serial pilot is not a controlled speed comparison.'}
    meta=args.out/'metadata.json'
    if meta.exists() and json.loads(meta.read_text(encoding='utf8'))!=metadata:
        raise ValueError('Different source/environment/configuration; use a new directory')
    meta.write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf8')
    for f in source_hashes:
        target=args.out/'source_used'/f;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/f,target)
    benchmark=args.out/'benchmark';sensitivity=args.out/'sensitivity'
    paths={};completed=set()
    for kind,out,manifest in [
        ('benchmark',benchmark,{'instances':list(INSTANCES),'algorithms':list(SOLVERS),'repetitions':30,'customers':100,'parameters':asdict(base),'seed_range':[0,29]}),
        ('sensitivity',sensitivity,{'instance':'R101','algorithm':'hybrid1','repetitions':30,'customers':100,'base':asdict(base),'cases':CASES})]:
        path,records=prepare_output(out,manifest);paths[kind]=path
        completed.update(identity(kind,r) for r in records)
    jobs=[]
    # Interleave algorithms/seeds to avoid allocating one algorithm a special
    # phase of machine load. ProcessPool gives each run its own RNG and instance.
    for seed in range(30):
        for instance in INSTANCES:
            for algorithm in SOLVERS:
                job=('benchmark',instance,algorithm,seed,None,None,{},args.seconds)
                if job[:6] not in completed:jobs.append(job)
        for factor,level,changes in CASES:
            job=('sensitivity','R101','hybrid1',seed,factor,level,changes,args.seconds)
            if job[:6] not in completed:jobs.append(job)
    start=time.perf_counter();total=1170;count=len(completed)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(execute,job) for job in jobs]
        for future in as_completed(futures):
            kind,row=future.result();append_record(paths[kind],row);count+=1
            progress={'completed':count,'total':total,'elapsed_seconds':round(time.perf_counter()-start,1),
                      'last':{k:row[k] for k in ('instance','algorithm','seed','vehicles','distance','stop_reason')}}
            (args.out/'progress.json').write_text(json.dumps(progress,indent=2),encoding='utf8')
            if count%30==0:print(json.dumps(progress),flush=True)
    if hashes()!=source_hashes:raise RuntimeError('Source changed during run; preserve this snapshot and audit before using results')
    from experiments.statistics import summarize,load_runs
    from experiments.sensitivity import summarize as summarize_sensitivity
    from visualization.benchmark import make_figures
    from visualization.sensitivity import plot_study
    from experiments.behavior import analyze_behavior
    summarize(benchmark);make_figures(benchmark)
    sr=[json.loads(s) for s in paths['sensitivity'].read_text(encoding='utf8').splitlines()]
    summary,groups=summarize_sensitivity(sensitivity,sr)
    plot_study(sensitivity,summary,groups,'R101')
    analyze_behavior(benchmark,sensitivity,'R101',8)
    rows=load_runs(benchmark)
    assert len(rows)==900 and len(sr)==270
    assert set(Counter((r['instance'],r['algorithm']) for r in rows).values())=={30}
    assert set(Counter((r['factor'],r['level']) for r in sr).values())=={30}
    make_time_figures(benchmark,rows,args.seconds)
    done={'status':'complete','benchmark_runs':len(rows),'sensitivity_runs':len(sr),
          'all_routes_checked':True,'source_unchanged':True,'elapsed_seconds':time.perf_counter()-start,
          'stops':dict(Counter(r['stop_reason'] for r in rows+sr))}
    (args.out/'completion.json').write_text(json.dumps(done,indent=2),encoding='utf8')
    print(json.dumps(done),flush=True)

def make_time_figures(out,rows,seconds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'sa':'tab:blue','ts':'tab:orange','hybrid1':'tab:green','hybrid2':'tab:red','hybrid3':'tab:purple'}
    for instance in sorted({r['instance'] for r in rows}):
        fig,axes=plt.subplots(2,1,figsize=(9,7),sharex=True,layout='constrained')
        for row in rows:
            if row['instance']!=instance or row['seed']!=8:continue
            # Include initialization at its actual completion time. No invented
            # time-zero samples, and no extension beyond actual termination.
            x=row['history_times']+[row['runtime']];h=row['history']+[row['history'][-1]]
            label=f"{row['algorithm']} ({row['stop_reason']})"
            for ax,column in zip(axes,(0,1)):
                y=[z[column] for z in h]
                ax.step(x,y,where='post',label=label,color=colors[row['algorithm']],alpha=.8)
                ax.plot(x[-1],y[-1],marker='o',ms=4,color=colors[row['algorithm']])
        axes[0].set(title=f'{instance} best-so-far, fixed seed 8; upper budget {seconds:g}s',ylabel='Vehicles')
        axes[1].set(xlabel='Elapsed wall time (s; includes initialization)',ylabel='Distance',xlim=(0,seconds*1.01))
        axes[0].legend(fontsize=8);axes[0].grid(alpha=.2);axes[1].grid(alpha=.2)
        fig.savefig(out/f'{instance}_seed8_walltime.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
