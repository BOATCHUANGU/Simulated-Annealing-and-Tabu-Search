"""固定较少车辆数时，用软约束退火修复拆车后的路线。"""
from math import exp
from algorithms.routes import EPSILON, changed_solution


class FleetRepair:
    """暂时允许超载或迟到，输出前仍用原始约束验证。"""

    def __init__(self, state, index):
        self.state = state
        self.routes = [route[:] for route in state.current.routes]
        pending = self.routes.pop(index)
        self.capacity = state.problem.capacity
        self.dist = state.problem.dist
        self.customers = state.problem.customers
        self.metrics = [self.measure(route) for route in self.routes]
        for node in pending:
            _, i, pos = min(
                (self.cost(self.measure(route[:pos] + [node] + route[pos:]), 4)
                 - self.cost(self.metrics[i], 4), i, pos)
                for i, route in enumerate(self.routes)
                for pos in range(len(route) + 1)
            )
            self.routes[i].insert(pos, node)
            self.metrics[i] = self.measure(self.routes[i])
        self.steps = 0
        self.best = None
        self.best_routes = None
        self.remember()

    def measure(self, route):
        depot = self.customers[0]
        current, last = depot.ready, 0
        distance = lateness = load = 0.0
        for node in route:
            customer = self.customers[node]
            leg = self.dist[last, node]
            distance += leg
            start = max(current + leg, customer.ready)
            lateness += max(0, start - customer.due)
            current = min(start, customer.due) + customer.service
            load += customer.demand
            last = node
        back = self.dist[last, 0]
        return distance + back, lateness + max(0, current + back - depot.due), max(0, load - self.capacity)

    @staticmethod
    def cost(metrics, penalty):
        distance, lateness, excess = metrics
        return distance + penalty * (lateness + excess)

    def remember(self):
        violation = sum(lateness + excess for _, lateness, excess in self.metrics)
        score = (violation, sum(distance for distance, _, _ in self.metrics))
        if self.best is None or score < self.best:
            self.best = score
            self.best_routes = [route[:] for route in self.routes]

    def propose(self, i, j, penalty):
        rng = self.state.rng
        a, b = self.routes[i], self.routes[j]
        if not a:
            return {}
        x = rng.randrange(len(a))
        move = rng.random()
        if move < .45:
            length = min(len(a) - x, rng.choice((1, 1, 1, 2, 3)))
            shorter = a[:x] + a[x + length:]
            target = shorter if i == j else b
            y = rng.randrange(len(target) + 1)
            return {i: shorter, j: target[:y] + a[x:x + length] + target[y:]}
        if i != j and move < .88:
            y = rng.randrange(len(b) + 1)
            if move < .68 and y < len(b):
                return {i: a[:x] + [b[y]] + a[x + 1:], j: b[:y] + [a[x]] + b[y + 1:]}
            return {i: a[:x] + b[y:], j: b[:y] + a[x:]}
        if i != j and b:
            y = rng.randrange(len(b))
            return {i: self.best_position(a[:x] + a[x + 1:], b[y], penalty),
                    j: self.best_position(b[:y] + b[y + 1:], a[x], penalty)}
        y = rng.randrange(x + 1, len(a) + 1)
        return {i: a[:x] + a[x:y][::-1] + a[y:]}

    def best_position(self, route, node, penalty):
        return min((route[:pos] + [node] + route[pos:] for pos in range(len(route) + 1)),
                   key=lambda candidate: self.cost(self.measure(candidate), penalty))

    def advance(self, trials):
        """延续上次的部分解，按周期重新升温并交替调整违约权重。"""
        rng = self.state.rng
        for _ in range(trials):
            if self.state.time_expired():
                break
            if self.best[0] <= EPSILON:
                return [route for route in self.best_routes if route]
            cycle, phase = divmod(self.steps, 4000)
            if phase == 0 and cycle % 9 == 0:
                self.routes = [route[:] for route in self.best_routes]
                self.metrics = [self.measure(route) for route in self.routes]
            self.steps += 1
            penalty = (3, 8, 25)[cycle % 3]
            temperature = 3 + 28 * (1 - phase / 4000) ** 2
            i = rng.randrange(len(self.routes))
            if rng.random() < .4:
                bad = [j for j, (_, late, excess) in enumerate(self.metrics) if late + excess > EPSILON]
                if bad:
                    i = rng.choice(bad)
            j = rng.randrange(len(self.routes))
            changes = self.propose(i, j, penalty)
            if changes:
                measured = {k: self.measure(route) for k, route in changes.items()}
                delta = sum(self.cost(value, penalty) - self.cost(self.metrics[k], penalty)
                            for k, value in measured.items())
                if delta <= 0 or rng.random() < exp(-delta / temperature):
                    for k, route in changes.items():
                        self.routes[k], self.metrics[k] = route, measured[k]
                    self.remember()
        if self.best[0] <= EPSILON:
            return [route for route in self.best_routes if route]
        return None


def recover_fleet(state, repair, budget):
    """从拆一辆车的状态继续搜索，找到完整可行解才提交。"""
    routes = state.current.routes
    if len(routes) < 2 or sum(c.demand for c in state.problem.customers.values() if c.id) > (len(routes) - 1) * state.problem.capacity + EPSILON:
        return None, None
    if repair is None or len(repair.routes) + 1 != len(routes) or repair.steps >= 360000:
        shortest = sorted(range(len(routes)), key=lambda i: len(routes[i]))
        repair = FleetRepair(state, state.rng.choice(shortest[:max(1, len(shortest) // 3)]))
    improved = repair.advance(budget)
    if improved is None:
        return repair, None
    replacements = {i: improved[i] if i < len(improved) else [] for i in range(len(routes))}
    return None, changed_solution(state, replacements)

