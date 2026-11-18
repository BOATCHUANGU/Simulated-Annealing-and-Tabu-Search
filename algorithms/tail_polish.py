"""在固定车辆数下，枚举可行后缀并重新分配各路线的尾部客户。"""
from time import perf_counter

from algorithms.routes import EPSILON, changed_solution


def suffix_options(problem, prefix, original_suffix, pending, deadline):
    """同一组尾部客户只保留距离最短的可行接续。"""
    last = prefix[-1]
    prefix_check = problem.check_route(prefix)
    depot_due = problem.customers[0].due
    load = sum(problem.customers[node].demand for node in prefix)
    original_mask = sum(bit for node, bit in pending if node in original_suffix)
    original_route = prefix + original_suffix
    options = {original_mask: (problem.check_route(original_route).distance, original_suffix)}
    dist, customers = problem.dist, problem.customers
    pending = [(node, bit, customers[node]) for node, bit in pending]
    visits = 0

    def extend(suffix, previous, departure, used_load, distance, mask):
        nonlocal visits
        visits += 1
        if visits >= 30000 or perf_counter() >= deadline:
            return
        total = distance + dist[previous, 0]
        if mask not in options or total < options[mask][0]:
            options[mask] = total, suffix
        for node, bit, customer in pending:
            if mask & bit or used_load + customer.demand > problem.capacity + EPSILON:
                continue
            leg = dist[previous, node]
            start = max(departure + leg, customer.ready)
            finish = start + customer.service
            if start <= customer.due + EPSILON and finish + dist[node, 0] <= depot_due + EPSILON:
                extend(suffix + (node,), node, finish, used_load + customer.demand, distance + leg, mask | bit)

    extend((), last, prefix_check.start_times[-1] + customers[last].service,
           load, prefix_check.distance - dist[last, 0], 0)
    return options.values()


def polish_tails(state, width=3):
    """每个车选一个后缀，每个取出的客户分配一次。"""
    if len(state.current.routes) != len(state.best.routes):
        return None
    deadline = state.started + state.config.time_limit if state.config.time_limit is not None else float("inf")
    if deadline - perf_counter() < 2:
        return None

    from numpy import ones
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csc_matrix

    routes = state.best.routes
    tails = [(tuple(route[:max(1, len(route) - width)]), tuple(route[max(1, len(route) - width):]))
             for route in routes]
    pending = sorted(node for _, tail in tails for node in tail)
    if not pending:
        return None
    node_rows = {node: i for i, node in enumerate(pending)}
    bits = [(node, 1 << i) for node, i in node_rows.items()]
    enumeration_end = min(deadline - 1.5, perf_counter() + 1.2)
    columns, rows, indices = [], [], []
    for i, (prefix, tail) in enumerate(tails):
        if perf_counter() >= enumeration_end:
            return None
        options = suffix_options(state.problem, prefix, tail, bits, enumeration_end)
        for distance, suffix in options:
            positions = [len(pending) + i] + [node_rows[node] for node in suffix]
            rows.extend(positions)
            indices.extend([len(columns)] * len(positions))
            columns.append((i, distance, prefix + suffix))
    remaining = deadline - perf_counter()
    if remaining <= 0.2:
        return None
    incidence = csc_matrix((ones(len(rows)), (rows, indices)),
                           shape=(len(pending) + len(routes), len(columns)))
    answer = milp([distance for _, distance, _ in columns], integrality=ones(len(columns)),
                  bounds=Bounds(0, 1), constraints=LinearConstraint(incidence, 1, 1),
                  options={"time_limit": min(3.0, remaining - 0.1)})
    if answer.x is None:
        return None
    selected = [(i, route) for (i, _, route), value in zip(columns, answer.x) if value > 0.5]
    replacements = {i: list(route) for i, route in selected}
    if len(selected) != len(routes) or len(replacements) != len(routes):
        return None
    checked = state.problem.evaluate(replacements.values())
    if not checked.feasible or checked.distance >= state.best.distance - EPSILON:
        return None
    return changed_solution(state, replacements)

