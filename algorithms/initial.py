"""两种初始排列：随机与最近可行客户贪心。"""
import random


def random_order(problem, seed=0):
    order = sorted(set(problem.customers) - {0})
    random.Random(seed).shuffle(order)
    return order


def greedy_order(problem):
    remaining = set(problem.customers) - {0}
    routes = []
    while remaining:
        route = []
        while True:
            feasible = [node for node in remaining if problem.check_route(route + [node]).feasible]
            if not feasible:
                break
            last = route[-1] if route else 0
            node = min(feasible, key=lambda i: (problem.dist[last, i], i))
            route.append(node)
            remaining.remove(node)
        if not route:
            raise ValueError("存在无法由单车服务的客户")
        routes.append(route)
    return [node for route in routes for node in route]


def insertion_order(problem):
    """按时间窗截止时间排序，以最小增量插入已有可行路线。"""
    routes = []
    customers = sorted(set(problem.customers) - {0}, key=lambda i: (problem.customers[i].due, i))
    for node in customers:
        best = None
        for index, route in enumerate(routes):
            before = problem.check_route(route).distance
            for position in range(len(route) + 1):
                trial = route[:position] + [node] + route[position:]
                checked = problem.check_route(trial)
                if checked.feasible:
                    option = (checked.distance - before, index, position)
                    if best is None or option < best:
                        best = option
        if best is None:
            if not problem.check_route([node]).feasible:
                raise ValueError("存在无法由单车服务的客户")
            routes.append([node])
        else:
            _, index, position = best
            routes[index].insert(position, node)
    return [node for route in routes for node in route]
