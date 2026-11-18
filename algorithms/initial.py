#初始解构造
import random
from algorithms.routes import RouteLabels


def random_order(problem, seed=0): #随机排列客户，返回不含车场的客户列表。
    order = sorted(set(problem.customers) - {0})
    random.Random(seed).shuffle(order)
    return order


def greedy_order(problem): #按最近可行客户构造初始解，返回客户排列。
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


def insertion_order(problem): #按时间窗和插入代价构造初始解，返回客户排列。
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


def sequential_routes(problem, seed_key, time_weight, expired):
    """逐条构造路线，以距离增量和后继延迟选择插入点。"""
    remaining = set(problem.customers) - {0}
    routes = []
    while remaining:
        if expired():
            return None
        seed = min(remaining, key=lambda node: (seed_key(node), node))
        route = [seed]
        remaining.remove(seed)
        while remaining:
            label = RouteLabels(problem, route)
            best = None
            for node in sorted(remaining):
                customer = problem.customers[node]
                for position, extra_distance in label.insertions(node):
                    left, right = label.nodes[position:position + 2]
                    new_departure = max(label.departures[position] + problem.dist[left, node],
                                        customer.ready) + customer.service
                    delay = new_departure + problem.dist[node, right] - problem.dist[left, right] - label.departures[position]
                    if position < len(route):
                        delay -= max(0, problem.customers[right].ready - label.departures[position]
                                     - problem.dist[left, right])
                    option = (extra_distance + time_weight * max(0, delay), node, position)
                    if best is None or option < best:
                        best = option
            if best is None:
                break
            _, node, position = best
            route.insert(position, node)
            remaining.remove(node)
        routes.append(route)
    return problem.evaluate(routes)
