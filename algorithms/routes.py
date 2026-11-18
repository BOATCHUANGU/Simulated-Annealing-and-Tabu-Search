"""时间窗标签、路线校验以及邻域复用的插入和逐出操作。"""
from dataclasses import dataclass
from itertools import combinations
from math import inf
from typing import NamedTuple
from problems.vrptw import Solution

EPSILON = 1e-9


class RouteLabels:
    """前向最早离开时间和后向最晚服务时间。"""

    def __init__(self, problem, route):
        self.problem = problem
        self.nodes = (0, *route, 0)
        self.load = sum(problem.customers[node].demand for node in route)
        self.departures = [problem.customers[0].ready]
        for left, right in zip(self.nodes, self.nodes[1:-1]):
            customer = problem.customers[right]
            arrival = self.departures[-1] + problem.dist[left, right]
            self.departures.append(max(arrival, customer.ready) + customer.service)
        self.latest = [problem.customers[0].due + EPSILON] * len(self.nodes)
        for i in range(len(route), 0, -1):
            node = self.nodes[i]
            customer = problem.customers[node]
            self.latest[i] = min(customer.due + EPSILON, self.latest[i + 1]
                                 - problem.dist[node, self.nodes[i + 1]] - customer.service)

    def insertions(self, node):
        """给出所有可行插入位置和距离增量，每个位置只做常数次检查。"""
        problem = self.problem
        customer = problem.customers[node]
        if self.load + customer.demand > problem.capacity + EPSILON:
            return
        dist = problem.dist
        for i, departure in enumerate(self.departures):
            left, right = self.nodes[i:i + 2]
            length_in, length_out = dist[left, node], dist[node, right]
            start = max(departure + length_in, customer.ready)
            if start <= customer.due + EPSILON and start + customer.service + length_out <= self.latest[i + 1]:
                yield i, length_in + length_out - dist[left, right]


@dataclass
class RoutedMove:
    routes: list
    check: tuple
    forbid: tuple

    @property
    def order(self):
        return [node for route in self.routes for node in route]


class Candidate(NamedTuple):
    move: RoutedMove
    solution: Solution


def route_edges(route):
    nodes = (0, *route, 0)
    return {(a, b) for a, b in zip(nodes, nodes[1:])}


def changed_solution(state, replacements):
    """移动保持客户覆盖，用原评价器核验变更路线并复用其余路线。"""
    current = state.current
    routes, details = [], []
    added, removed = set(), set()
    for i, (previous, detail) in enumerate(zip(current.routes, current.details)):
        route = replacements.get(i, previous)
        if route != previous:
            if route:
                detail = state.problem.check_route(route)
                if not detail.feasible:
                    return None
            old_edges = route_edges(previous)
            new_edges = route_edges(route) if route else set()
            added.update(new_edges - old_edges)
            removed.update(old_edges - new_edges)
            if not route:
                continue
        routes.append(route)
        details.append(detail)
    if not added and not removed:
        return None
    solution = Solution(True, routes, sum(detail.distance for detail in details), details)
    move = RoutedMove(routes, tuple(sorted(added - removed)), tuple(sorted(removed - added)))
    return Candidate(move, solution)


def repair(problem, routes, missing, rng, expired, noise=0.0):
    """按后悔值选客户。只在发生插入的路线重新计算可行位置。"""
    labels = [RouteLabels(problem, route) for route in routes]
    options = {node: [best_insertion(label, node, rng, noise) for label in labels]
               for node in missing}
    while options and not expired():
        selected = None
        priority = None
        for node, choices in options.items():
            first, second = (inf, -1, -1), (inf, -1, -1)
            for i, (cost, pos) in enumerate(choices):
                item = (cost, i, pos)
                if item < first:
                    first, second = item, first
                elif item < second:
                    second = item
            if first[0] == inf:
                return list(options)
            score = (second[0] - first[0], -first[0])
            if priority is None or score > priority:
                priority = score
                selected = node, first[1], first[2]
        node, i, pos = selected
        routes[i].insert(pos, node)
        del options[node]
        label = RouteLabels(problem, routes[i])
        for other, choices in options.items():
            choices[i] = best_insertion(label, other, rng, noise)
    return list(options)


def best_insertion(label, node, rng, noise):
    return min(((cost + rng.uniform(-noise, noise) if noise else cost, pos)
                for pos, cost in label.insertions(node)), default=(inf, -1))


def perturb(problem, routes, rng):
    """在部分解上随机做一次可行移位，打破逐出链循环。"""
    a, b = rng.sample(routes, 2) if len(routes) >= 2 else (routes[0], routes[0])
    if len(a) <= 1:
        return
    pos = rng.randrange(len(a))
    node = a[pos]
    target = b if a is not b else a[:pos] + a[pos + 1:]
    positions = list(RouteLabels(problem, target).insertions(node))
    if positions:
        index, _ = rng.choice(positions)
        a.pop(pos)
        b.insert(index, node)


def ejection(problem, routes, node, penalties, rng, variants, count, expired):
    best = None
    choice = None
    for i, route in enumerate(routes):
        if expired():
            return None
        cache_key = (tuple(route), count)
        if cache_key not in variants:
            candidates = []
            original = sum(problem.dist[a, b] for a, b in route_edges(route))
            for kicked in combinations(route, count):
                if expired():
                    return None
                shorter = [other for other in route if other not in kicked]
                label = RouteLabels(problem, shorter)
                saved = original - sum(problem.dist[a, b] for a, b in route_edges(shorter))
                candidates.append((kicked, label, saved))
            variants[cache_key] = candidates
        for kicked, label, saved in variants[cache_key]:
            penalty = sum(penalties[other] for other in kicked)
            for pos, cost in label.insertions(node):
                score = 24 * penalty + 0.4 * (cost - saved) + rng.uniform(0, 32)
                if best is None or score < best:
                    best = score
                    choice = (i, pos, label)
    return choice
