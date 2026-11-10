import tempfile
import unittest
from pathlib import Path
from algorithms.common import SearchConfig
from experiments.sensitivity import CASES, run_study, summarize
from visualization.sensitivity import plot_study
from experiments.behavior import analyze_behavior
import json


class TestSensitivity(unittest.TestCase):
    def test_cases_vary_one_parameter(self):
        for factor, _, changes in CASES:
            self.assertEqual(list(changes), [factor])

    def test_study_resume_and_summaries(self):
        with tempfile.TemporaryDirectory() as temp:
            config = SearchConfig(iterations=2, candidate_count=30, seed=3)
            records = run_study(temp, "R101", "hybrid1", 2, config, customers=10)
            again = run_study(temp, "R101", "hybrid1", 2, config, customers=10)
            self.assertEqual(len(records), 18)
            self.assertEqual(len(again), 18)
            summary, groups = summarize(temp, records)
            self.assertEqual(len(summary), 9)
            self.assertTrue(all(r["runs"] == 2 for r in summary))
            plot_study(temp, summary, groups, "R101")
            self.assertTrue((Path(temp)/"parameters.png").exists())
            self.assertTrue((Path(temp)/"neighborhoods.png").exists())
            with self.assertRaises(ValueError):
                run_study(temp, "R101", "hybrid1", 3, config, customers=10)

    def test_behavior_summary(self):
        record = {"instance": "R101", "algorithm": "sa", "seed": 0, "status": "ok",
                  "vehicles": 1, "distance": 9, "runtime": 0.1, "iterations": 1,
                  "history": [[1, 10], [1, 9]]}
        with tempfile.TemporaryDirectory() as temp:
            benchmark = Path(temp) / "benchmark"
            benchmark.mkdir()
            (benchmark / "runs.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
            output = Path(temp) / "out"
            rows = analyze_behavior(benchmark, output)
            self.assertEqual(rows[0]["improved_runs"], 1)
            self.assertEqual(rows[0]["mean_best_updates"], 1)
            self.assertTrue((output / "convergence.png").exists())

    def test_mixed_vehicle_count_is_grouped(self):
        records = [{"factor": "tabu_tenure", "level": "5", "vehicles": v,
                    "distance": d, "runtime": 1, "iterations": 1}
                   for v, d in [(2, 20), (3, 10)]]
        with tempfile.TemporaryDirectory() as temp:
            summary, grouped = summarize(temp, records)
            self.assertEqual(summary[0]["mean_distance_all"], 15)
            self.assertEqual({r["vehicles"] for r in grouped}, {2, 3})


if __name__ == "__main__":
    unittest.main()
