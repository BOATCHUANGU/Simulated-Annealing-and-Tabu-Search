import random
import unittest
from dataclasses import replace
from config.config import DATA_DIR
from problems.vrptw import Customer, VRPTW
from algorithms.initial import greedy_order
from algorithms.common import SearchConfig, SearchState, key
from algorithms.fleet_repair import recover_fleet
from algorithms.neighborhoods import generate
from algorithms.routes import RouteLabels
from algorithms.tail_polish import polish_tails
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
                order = state.order
                i, j = sorted(rng.sample(range(len(order)), 2))
                if kind == "relocate":
                    order.insert(j, order.pop(i))
                elif kind == "swap":
                    order[i], order[j] = order[j], order[i]
                else:
                    order[i:j + 1] = reversed(order[i:j + 1])
                partial = self.problem.decode_neighbor(order, state.current, i)
                full = self.problem.decode(order)
                self.assertEqual(partial, full)
                if full.feasible:
                    state.update(full)


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

    def test_insertion_labels_match_full_route_checks(self):
        problem = VRPTW.from_solomon(DATA_DIR / "RC101.txt")
        state = SearchState(problem, SearchConfig(iterations=0))
        for route in state.current.routes:
            labels = RouteLabels(problem, route)
            before = problem.check_route(route).distance
            for node in problem.customers.keys() - {0, *route}:
                feasible = dict(labels.insertions(node))
                for pos in range(len(route) + 1):
                    trial = problem.check_route(route[:pos] + [node] + route[pos:])
                    self.assertEqual(pos in feasible, trial.feasible)
                    if trial.feasible:
                        self.assertAlmostEqual(feasible[pos], trial.distance - before)

    def test_vehicle_elimination_can_eject_a_customer(self):
        problem = VRPTW([Customer(i, i, 0, demand, 0, 100, 0)
                         for i, demand in enumerate((0, 6, 2, 6, 2, 4))], 10, 3)
        state = SearchState(problem, SearchConfig(seed=1, neighborhoods=("eliminate",)))
        initial = problem.evaluate([[1, 2], [3, 4], [5]])
        state.update(initial)
        move, result = generate(state)
        self.assertEqual(result, problem.evaluate(result.routes))
        self.assertEqual(result.vehicles, 2)
        self.assertEqual(sorted(move.order), [1, 2, 3, 4, 5])
        self.assertEqual(state.current.routes, [[1, 2], [3, 4], [5]])

        attempt, item = recover_fleet(state, None, 2000)
        self.assertIsNone(attempt)
        self.assertIsNotNone(item)
        self.assertEqual(item.solution, problem.evaluate(item.solution.routes))
        self.assertEqual(item.solution.vehicles, 2)

    def test_explicit_neighbors_match_full_solution_checks(self):
        state = SearchState(self.problem, SearchConfig())
        for kind in state.config.neighborhoods:
            state.config = replace(state.config, neighborhoods=(kind,))
            for _ in range(40):
                item = generate(state)
                if item:
                    move, solution = item
                    self.assertEqual(solution, self.problem.evaluate(solution.routes))
                    state.update(solution)
        self.assertEqual(set(state.initial_solutions),
                         {"insertion", "nearest", "farthest", "earliest_due", "earliest_ready"})

    def test_tail_polish_reassigns_customers_between_routes(self):
        problem = VRPTW([Customer(i, x, 0, int(i != 0), 0, 100, 0)
                         for i, x in enumerate((0, 10, -10, -11, 11))], 2, 2)
        state = SearchState(problem, SearchConfig())
        state.current = state.best = problem.evaluate([[1, 3], [2, 4]])
        _, polished = polish_tails(state)
        self.assertEqual(polished, problem.evaluate([[1, 4], [2, 3]]))
        self.assertEqual(polished.distance, 44)

    def test_matching_the_target_stops_the_search(self):
        result = sa.solve(self.problem, SearchConfig(target_vehicles=1, target_distance=100))
        self.assertEqual(result.stop_reason, "target")
        self.assertEqual(result.iterations, 0)

    def test_reaching_temperature_floor_does_not_stop_annealing(self):
        config = SearchConfig(iterations=20, initial_temperature=1, min_temperature=0.5, cooling=0.1)
        for solve in (sa.solve, hybrid.ts_metropolis, hybrid.sa_ts_inner):
            with self.subTest(algorithm=solve):
                result = solve(self.problem, config)
                self.assertEqual(result.stop_reason, "iterations")
                self.assertEqual(result.iterations, config.iterations)

    def test_invalid_config(self):
        with self.assertRaises(ValueError):
            sa.solve(self.problem, SearchConfig(cooling=0))


if __name__ == '__main__':
    unittest.main()
