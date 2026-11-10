"""任务三的汇总与配对统计；按车辆数分组距离。"""
import csv
import json
import math
from collections import defaultdict
from statistics import mean, stdev
from pathlib import Path
from scipy.stats import ttest_rel
from experiments.benchmark import REFERENCES


def load_runs(output):
    path = Path(output) / "runs.jsonl"
    if not path.exists():
        return []
    return [r for line in path.read_text(encoding="utf-8").splitlines() if (r := json.loads(line)).get("status") == "ok"]


def write_csv(path, fields, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def paired_comparison(a, b):
    """按种子配对且两次车辆数相同；不同车数组分开。"""
    matches = defaultdict(list)
    by_seed = {r["seed"]: r for r in b}
    for left in a:
        right = by_seed.get(left["seed"])
        if right and left["vehicles"] == right["vehicles"]:
            matches[left["vehicles"]].append((left["distance"], right["distance"]))
    results = []
    for vehicles, pairs in sorted(matches.items()):
        diffs = [x-y for x, y in pairs]
        statistic = pvalue = ""
        note = "paired t-test; alpha=0.05"
        if len(pairs) < 2:
            note = "insufficient pairs"
        elif max(diffs)-min(diffs) < 1e-12:
            if abs(diffs[0]) < 1e-12:
                note = "identical pairs; t-test undefined"
            else:
                note = "constant nonzero differences; t-test undefined"
        else:
            test = ttest_rel([x for x, _ in pairs], [y for _, y in pairs])
            statistic, pvalue = float(test.statistic), float(test.pvalue)
            if not math.isfinite(statistic) or not math.isfinite(pvalue):
                statistic = pvalue = ""
                note = "nonfinite test statistic"
        results.append({"vehicles": vehicles, "pairs": len(pairs), "mean_difference": mean(diffs),
                        "statistic": statistic, "p_value": pvalue, "note": note})
    return results


def summarize(output):
    output = Path(output)
    runs = load_runs(output)
    manifest = json.loads((output / "config.json").read_text(encoding="utf-8"))
    groups = defaultdict(list)
    for r in runs:
        groups[r["instance"], r["algorithm"]].append(r)
    summary, by_vehicle, tests = [], [], []
    for (instance, algorithm), records in sorted(groups.items()):
        best = min(records, key=lambda r: (r["vehicles"], r["distance"]))
        ref = REFERENCES.get(instance) if manifest["customers"] == 100 else None
        gap = (best["distance"]-ref[1])/ref[1]*100 if ref and best["vehicles"] == ref[0] else ""
        summary.append({"instance": instance, "algorithm": algorithm, "runs": len(records),
                        "best_vehicles": best["vehicles"], "best_distance": best["distance"],
                        "best_seed": best["seed"], "gap_percent": gap, "mean_runtime": mean(r["runtime"] for r in records),
                        "mean_iterations": mean(r["iterations"] for r in records),
                        "vehicle_counts": ";".join(f"{v}:{sum(r['vehicles']==v for r in records)}" for v in sorted({r["vehicles"] for r in records}))})
        for vehicles in sorted({r["vehicles"] for r in records}):
            subset = [r["distance"] for r in records if r["vehicles"] == vehicles]
            by_vehicle.append({"instance": instance, "algorithm": algorithm, "vehicles": vehicles,
                               "runs": len(subset), "mean_distance": mean(subset),
                               "std_distance": stdev(subset) if len(subset)>1 else ""})
    for instance in manifest["instances"]:
        for hybrid in ("hybrid1", "hybrid2", "hybrid3"):
            for baseline in ("sa", "ts"):
                if (instance, hybrid) not in groups or (instance, baseline) not in groups:
                    continue
                rows = paired_comparison(groups[instance, hybrid], groups[instance, baseline])
                if not rows:
                    rows = [{"vehicles": "", "pairs": 0, "mean_difference": "", "statistic": "", "p_value": "", "note": "no matched vehicle counts"}]
                for row in rows:
                    tests.append({"instance": instance, "algorithm": hybrid, "baseline": baseline, **row})
    write_csv(output / "summary.csv", ["instance","algorithm","runs","best_vehicles","best_distance","best_seed","gap_percent","mean_runtime","mean_iterations","vehicle_counts"], summary)
    write_csv(output / "distance_by_vehicles.csv", ["instance","algorithm","vehicles","runs","mean_distance","std_distance"], by_vehicle)
    write_csv(output / "statistical_tests.csv", ["instance","algorithm","baseline","vehicles","pairs","mean_difference","statistic","p_value","note"], tests)
    return summary, by_vehicle, tests
