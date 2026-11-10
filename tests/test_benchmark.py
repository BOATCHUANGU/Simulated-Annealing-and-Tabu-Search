import json
import tempfile
import unittest
from pathlib import Path
from algorithms.common import SearchConfig
from experiments.benchmark import run_benchmark
from experiments.statistics import paired_comparison, summarize, load_runs
from visualization.benchmark import make_figures


class TestBenchmark(unittest.TestCase):
    def test_statistics_requires_matching_vehicle_count(self):
        left = [{"seed": 0, "vehicles": 2, "distance": 10},
                {"seed": 1, "vehicles": 3, "distance": 11}]
        right = [{"seed": 0, "vehicles": 2, "distance": 10},
                 {"seed": 1, "vehicles": 2, "distance": 20}]
        rows = paired_comparison(left, right)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["pairs"], 1)
        self.assertEqual(rows[0]["p_value"], "")

    def test_paired_t_test(self):
        left = [{"seed": i, "vehicles": 2, "distance": v} for i, v in enumerate([10, 12, 14])]
        right = [{"seed": i, "vehicles": 2, "distance": 9} for i in range(3)]
        row = paired_comparison(left, right)[0]
        self.assertEqual(row["pairs"], 3)
        self.assertAlmostEqual(row["statistic"], 2.598076211, places=6)
        self.assertGreater(row["p_value"], 0.05)
        self.assertLess(row["p_value"], 0.2)

    def test_batch_resume_and_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = SearchConfig(seed=5, iterations=3, candidate_count=2, time_limit=1)
            for _ in range(2):
                run_benchmark(tmp, ["C101"], ["sa", "ts"], 2, config, customers=10)
            runs = load_runs(tmp)
            self.assertEqual(len(runs), 4)
            self.assertTrue(all(len(r["history"]) == r["iterations"]+1 for r in runs))
            summary, groups, _ = summarize(tmp)
            self.assertEqual(len(summary), 2)
            self.assertTrue(all(row["gap_percent"] == "" for row in summary))
            make_figures(tmp)
            self.assertTrue((Path(tmp) / "convergence.png").exists())
            self.assertTrue((Path(tmp) / "best_route.png").exists())
            with self.assertRaises(ValueError):
                run_benchmark(tmp, ["C101"], ["sa"], 2, config, customers=10)


if __name__ == "__main__":
    unittest.main()
