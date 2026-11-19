"""时间窗标签、路线校验以及邻域复用的插入和逐出操作。"""
from dataclasses import dataclass
from itertools import combinations
from math import inf
from typing import NamedTuple
from problems.vrptw import Solution

EPSILON = 1e-9


class RouteLabels:
    """前向最早离开时间和后向最晚服务时间。"""

    def __init__(self, problem, route):  #线性预处理路线载荷和前后向时间窗标签 返回None
        self.problem = problem  #用于查询客户 距离 容量和时间窗的VRPTW实例
        self.nodes = (0, *route, 0)  #首尾均加上车场编号0的路线节点元组
        self.load = sum(problem.customers[node].demand for node in route)  #当前路线全部客户的总需求量
        self.departures = [problem.customers[0].ready]  #前向标签 记录各插入点左侧节点的最早离开时刻
        for left, right in zip(self.nodes, self.nodes[1:-1]):
            customer = problem.customers[right]
            arrival = self.departures[-1] + problem.dist[left, right]
            self.departures.append(max(arrival, customer.ready) + customer.service)
        self.latest = [problem.customers[0].due + EPSILON] * len(self.nodes)  #后向标签 记录各节点的最晚开始服务时刻 末位为回场上限
        for i in range(len(route), 0, -1):
            node = self.nodes[i]
            customer = problem.customers[node]
            self.latest[i] = min(customer.due + EPSILON, self.latest[i + 1]
                                 - problem.dist[node, self.nodes[i + 1]] - customer.service)

    def insertions(self, node):  #枚举符合容量和时间窗的客户插入点 返回generator逐个产出位置与距离增量
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
    routes: list  #此次邻域操作生成的各车辆路线
    added_arcs: tuple  #本次新增的有向弧 用于判断候选是否触发禁忌
    removed_arcs: tuple  #本次删除的有向弧 移动提交后写入禁忌表

    @property
    def order(self):  #将移动后的各车路线展开 返回list[int]客户编号
        return [node for route in self.routes for node in route]


class Candidate(NamedTuple):
    move: RoutedMove  #候选移动 包含变动后的路线和禁忌弧
    solution: Solution  #经过可行性检查的候选路线解及评价结果


def route_arcs(route):  #提取包含出发和回场弧的路线连接 返回set有向弧集合
    nodes = (0, *route, 0)
    return set(zip(nodes, nodes[1:]))


def changed_solution(state, replacements):  #校验变化路线并生成禁忌属性 返回Candidate移动与可行解或None
    """移动保持客户覆盖，用原评价器核验变更路线并复用其余路线。"""
    current = state.current
    routes, details = [], []
    added, removed = set(), set()
    for route_index, (previous, detail) in enumerate(zip(current.routes, current.details)):
        route = replacements.get(route_index, previous)
        if route != previous:
            if route:
                detail = state.problem.check_route(route)
                if not detail.feasible:
                    return None
            old_arcs = route_arcs(previous)
            new_arcs = route_arcs(route) if route else set()
            added.update(new_arcs - old_arcs)
            removed.update(old_arcs - new_arcs)
            if not route:
                continue
        routes.append(route)
        details.append(detail)
    if not added and not removed:
        return None
    solution = Solution(True, routes, sum(detail.distance for detail in details), details)
    move = RoutedMove(routes, tuple(sorted(added - removed)), tuple(sorted(removed - added)))
    return Candidate(move, solution)


def repair(problem, routes, missing, rng, expired, noise=0.0):  #按后悔值把待安置客户插入现有路线 返回list[int]尚未插入的客户编号
    """按后悔值选客户。只在发生插入的路线重新计算可行位置。"""
    labels = [RouteLabels(problem, route) for route in routes]
    options = {node: [best_insertion(label, node, rng, noise) for label in labels]
               for node in missing}
    while options and not expired():
        selected = None
        priority = None
        for node, choices in options.items():
            # 找到该客户在不同车辆上的最优和次优插入方案。
            first, second = (inf, -1, -1), (inf, -1, -1)
            for route_index, (cost, position) in enumerate(choices):
                option = (cost, route_index, position)
                if option < first:
                    first, second = option, first
                elif option < second:
                    second = option

            best_cost, route_index, position = first
            if best_cost == inf:
                return list(options)
            # 越难用次优位置代替，越应该优先插入该客户。
            regret = second[0] - best_cost
            score = (regret, -best_cost)
            if priority is None or score > priority:
                priority = score
                selected = node, route_index, position

        node, route_index, position = selected
        route = routes[route_index]
        route.insert(position, node)
        del options[node]

        # 只有修改过的路线需要重建标签与插入选项。
        label = RouteLabels(problem, route)
        for other, choices in options.items():
            choices[route_index] = best_insertion(label, other, rng, noise)
    return list(options)


def best_insertion(label, node, rng, noise):  #选择评分最低的可行插入位置 返回tuple插入评分和位置无解时为(inf,-1)
    return min(((cost + rng.uniform(-noise, noise) if noise else cost, pos)
                for pos, cost in label.insertions(node)), default=(inf, -1))


def perturb(problem, routes, rng):  #对路线原地执行一次随机可行移位 返回None
    """在部分解上随机做一次可行移位，打破逐出链循环。"""
    source, target = rng.sample(routes, 2) if len(routes) >= 2 else (routes[0], routes[0])
    if len(source) <= 1:
        return
    position = rng.randrange(len(source))
    node = source[position]
    base = target if source is not target else source[:position] + source[position + 1:]
    positions = list(RouteLabels(problem, base).insertions(node))
    if positions:
        insert_at, _ = rng.choice(positions)
        source.pop(position)
        target.insert(insert_at, node)


def ejection(problem, routes, node, penalties, rng, variants, count, expired):  #搜索逐出客户后可插入新客户的最佳方案 返回tuple路线索引插入位置和路线标签或None
    best = None
    choice = None
    for route_index, route in enumerate(routes):
        if expired():
            return None
        cache_key = (tuple(route), count)
        if cache_key not in variants:
            candidates = []
            original = sum(problem.dist[a, b] for a, b in route_arcs(route))
            for kicked in combinations(route, count):
                if expired():
                    return None
                shorter = [other for other in route if other not in kicked]
                label = RouteLabels(problem, shorter)
                saved = original - sum(problem.dist[a, b] for a, b in route_arcs(shorter))
                candidates.append((kicked, label, saved))
            variants[cache_key] = candidates
        for kicked, label, saved in variants[cache_key]:
            penalty = sum(penalties[other] for other in kicked)
            for pos, cost in label.insertions(node):
                score = 24 * penalty + 0.4 * (cost - saved) + rng.uniform(0, 32)
                if best is None or score < best:
                    best = score
                    choice = (route_index, pos, label)
    return choice
