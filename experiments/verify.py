"""C101 前 10 客户的路径枚举与 PuLP 集合划分精确验证。"""
from math import hypot
import pulp
from problems.vrptw import VRPTW


def exact_check(path):
    problem = VRPTW.from_solomon(path, customer_limit=10)
    customers = problem.customers
    distance = lambda a, b: hypot(customers[a].x-customers[b].x, customers[a].y-customers[b].y)
    routes = []

    def enumerate_routes(route, remaining, current_time, load, length):
        last = route[-1] if route else 0
        if route:
            routes.append((tuple(route), length + distance(last, 0)))
        for node in remaining:
            c = customers[node]
            start = max(current_time + distance(last, node), c.ready)
            if (load+c.demand <= problem.capacity and start <= c.due and
                    start+c.service+distance(node, 0) <= customers[0].due):
                enumerate_routes(route+[node], remaining-{node}, start+c.service,
                                 load+c.demand, length+distance(last, node))

    enumerate_routes([], set(customers)-{0}, customers[0].ready, 0, 0)
    model = pulp.LpProblem("VRPTW_exact", pulp.LpMinimize)
    x = [pulp.LpVariable(f"route_{i}", cat="Binary") for i in range(len(routes))]
    for node in set(customers)-{0}:
        model += pulp.lpSum(x[i] for i, (route, _) in enumerate(routes) if node in route) == 1
    model += pulp.lpSum(x) <= problem.vehicle_limit
    solver = pulp.PULP_CBC_CMD(msg=False)
    model.setObjective(pulp.lpSum(x))
    model.solve(solver)
    if pulp.LpStatus[model.status] != "Optimal":
        raise RuntimeError("车辆数未证明最优")
    count = round(pulp.value(pulp.lpSum(x)))
    model += pulp.lpSum(x) == count
    model.setObjective(pulp.lpSum(x[i]*r[1] for i, r in enumerate(routes)))
    model.solve(solver)
    if pulp.LpStatus[model.status] != "Optimal":
        raise RuntimeError("距离未证明最优")
    selected = [route for i, (route, _) in enumerate(routes) if pulp.value(x[i]) > 0.5]
    exact_distance = pulp.value(model.objective)
    evaluated = problem.evaluate(selected)
    if not evaluated.feasible or evaluated.vehicles != count or abs(evaluated.distance-exact_distance) > 1e-6:
        raise AssertionError("精确解与评价函数不一致")
    return count, exact_distance, selected, len(routes)
