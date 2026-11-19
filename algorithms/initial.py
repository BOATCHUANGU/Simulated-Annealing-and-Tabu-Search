"""用最近客户、插入代价和种子规则构造初始解。"""
from algorithms.routes import RouteLabels


def greedy_order(problem):  #按最近可行客户构造初始路线 返回list[int]展开后的客户编号
    remaining = set(problem.customers) - {0}
    routes = []
    while remaining:
        route = []
        while remaining:
            last = route[-1] if route else 0
            feasible = (node for node in remaining if problem.check_route(route + [node]).feasible)
            nearest = min(feasible, key=lambda node: (problem.dist[last, node], node), default=None)
            if nearest is None:
                break
            route.append(nearest)
            remaining.remove(nearest)
        if not route:
            raise ValueError("存在无法由单车服务的客户")
        routes.append(route)
    return [node for route in routes for node in route]


def insertion_order(problem):  #按客户截止时间和插入代价构造初始路线 返回list[int]展开后的客户编号
    routes = []
    # 截止时间早的客户优先安排。
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


def sequential_routes(problem, seed_key, time_weight, expired):  #按种子规则逐车构造并评价路线 返回Solution初始解或None
    """逐条构造路线，以距离增量和后继延迟选择插入点。"""
    remaining = set(problem.customers) - {0}
    routes = []
    while remaining:
        if expired():
            return None
        # 每辆车先选一个种子客户，再尝试把路线填满。
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
                    departure = label.departures[position]
                    old_leg = problem.dist[left, right]
                    arrival = departure + problem.dist[left, node]
                    new_departure = max(arrival, customer.ready) + customer.service
                    delay = new_departure + problem.dist[node, right] - old_leg - departure
                    if position < len(route):
                        # 后继客户原本要等待的时间可以吸收一部分延迟。
                        ready = problem.customers[right].ready
                        delay -= max(0, ready - departure - old_leg)
                    score = extra_distance + time_weight * max(0, delay)
                    option = (score, node, position)
                    if best is None or option < best:
                        best = option
            if best is None:
                break
            _, node, position = best
            route.insert(position, node)
            remaining.remove(node)
        routes.append(route)
    return problem.evaluate(routes)
