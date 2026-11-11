"""五套算法的统一调用入口。"""
import json
from pathlib import Path
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


def prepare_output(output, manifest):
    """验证续跑参数并读取已有运行记录。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(json.dumps(manifest))  # 元组与 JSON 列表保持一致
    path = output / "config.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("输出目录已有不同实验参数，请使用新目录")
    else:
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    raw = output / "runs.jsonl"
    records = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines()] if raw.exists() else []
    return raw, records


def append_record(path, record):
    """完成一次运行立即保存，便于中断后续跑。"""
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
