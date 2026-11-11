#初始解构造
import random


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
