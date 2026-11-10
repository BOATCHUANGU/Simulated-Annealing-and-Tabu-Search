import random
import unittest
from config.config import DATA_DIR
from problems.vrptw import VRPTW
from algorithms.initial import greedy_order
from algorithms.common import SearchConfig, SearchState, key
from algorithms.neighborhoods import generate
from algorithms import sa, ts, hybrid
from experiments.verify import exact_check


class TestAlgorithms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problem = VRPTW.from_solomon(DATA_DIR / "C101.txt", 10)
        cls.optimal = exact_check(DATA_DIR / "C101.txt")

    def test_incremental_matches_full_decode(self):
        state = SearchState(self.problem, SearchConfig())
        rng = random.Random(7)
        for kind in ("relocate", "swap", "2opt"):
            for _ in range(200):
                move = generate(state.order, state.current, kind, rng)
                if move is None:
                    continue
                partial = self.problem.decode_neighbor(move.order, state.current, move.changed_at)
                full = self.problem.decode(move.order)
                self.assertEqual(partial, full)
                if full.feasible:
                    state.update(move.order, full)

    def test_five_algorithms_against_exact(self):
        funcs = (sa.solve, ts.solve, hybrid.ts_metropolis, hybrid.sa_ts_inner, hybrid.sa_warm_ts)
        config = SearchConfig(seed=3, iterations=80, candidate_count=8, inner_iterations=5, restart_after=10)
        initial = self.problem.decode(greedy_order(self.problem))
        optimal_count, optimal_distance = self.optimal[:2]
        for solve in funcs:
            with self.subTest(algorithm=solve.__module__ + '.' + solve.__name__):
                result = solve(self.problem, config)
                again = solve(self.problem, config)
                self.assertTrue(self.problem.evaluate(result.solution.routes).feasible)
                self.assertLessEqual(key(result.solution), key(initial))
                self.assertGreaterEqual(result.vehicles, optimal_count)
                if result.vehicles == optimal_count:
                    self.assertGreaterEqual(result.distance + 1e-6, optimal_distance)
                self.assertEqual(result.history, again.history)
                self.assertEqual(len(result.history), result.iterations + 1)
                self.assertEqual(result.history[-1], key(result.solution))
                self.assertTrue(all(a >= b for a, b in zip(result.history, result.history[1:])))

    def test_invalid_config(self):
        with self.assertRaises(ValueError):
            sa.solve(self.problem, SearchConfig(cooling=0))


if __name__ == '__main__':
    unittest.main()
