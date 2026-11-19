"""长预算搜索的两个阶段：先减车，再在固定车数下缩短距离。"""
from math import exp
from time import perf_counter

from algorithms.routes import EPSILON, changed_solution


# 一 减车修复

class FleetRepair:
    """暂时允许超载或迟到，输出前仍用原始约束验证。"""

    def __init__(self, state, index):  #拆除指定路线并构造带惩罚的减车起点 返回None
        self.state = state  #原搜索状态 提供问题实例 随机数和时间上限
        self.routes = [route[:] for route in state.current.routes]  #移除指定车辆后用于软约束搜索的临时路线
        pending = self.routes.pop(index)
        self.capacity = state.problem.capacity  #每辆车允许的最大载荷
        self.dist = state.problem.dist  #从起点和终点编号对查询行驶距离的字典
        self.customers = state.problem.customers  #从客户编号查询客户数据的字典 包含车场0
        self.metrics = [self.measure(route) for route in self.routes]  #与临时路线逐条对应的距离 超时量和超载量
        for node in pending:
            # 先把撤下来的客户全部插入，以惩罚代价增量选位置。
            best = None
            for route_index, route in enumerate(self.routes):
                before = self.cost(self.metrics[route_index], 4)
                for position in range(len(route) + 1):
                    trial = route[:position] + [node] + route[position:]
                    increase = self.cost(self.measure(trial), 4) - before
                    option = (increase, route_index, position)
                    if best is None or option < best:
                        best = option
            if best is None:
                raise ValueError("没有可接收客户的路线")
            _, route_index, position = best
            route = self.routes[route_index]
            route.insert(position, node)
            self.metrics[route_index] = self.measure(route)
        self.steps = 0  #本次软约束修复已尝试的搜索步数
        self.best = None  #修复过程中按违约总量优先比较的最好评分
        self.best_routes = None  #与最好评分对应的路线副本
        self.remember()

    def measure(self, route):  #评估路线的距离超时量和超载量 返回tuple距离超时量和超载量
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
        late_return = max(0, current + back - depot.due)
        overload = max(0, load - self.capacity)
        return distance + back, lateness + late_return, overload

    @staticmethod
    def cost(metrics, penalty):  #计算包含超时和超载惩罚的代价 返回float惩罚代价
        distance, lateness, excess = metrics
        return distance + penalty * (lateness + excess)

    def remember(self):  #按违约量和距离更新修复过程的最优路线 返回None
        violation = sum(lateness + excess for _, lateness, excess in self.metrics)
        score = (violation, sum(distance for distance, _, _ in self.metrics))
        if self.best is None or score < self.best:
            self.best = score
            self.best_routes = [route[:] for route in self.routes]

    def propose(self, src, dst, penalty):  #随机生成一次软约束路线变动 返回dict路线索引到替换路线的映射
        rng = self.state.rng
        source, target = self.routes[src], self.routes[dst]
        if not source:
            return {}

        start = rng.randrange(len(source))
        draw = rng.random()

        # 移位
        if draw < 0.45:
            length = min(len(source) - start, rng.choice((1, 1, 1, 2, 3)))
            shorter = source[:start] + source[start + length:]
            receiving = shorter if src == dst else target
            position = rng.randrange(len(receiving) + 1)
            moved = receiving[:position] + source[start:start + length] + receiving[position:]
            return {src: shorter, dst: moved}

        # 直接交换客户或交换两条路线的后缀。
        if src != dst and draw < 0.88:
            cut = rng.randrange(len(target) + 1)
            if draw < 0.68 and cut < len(target):
                new_source = source[:start] + [target[cut]] + source[start + 1:]
                new_target = target[:cut] + [source[start]] + target[cut + 1:]
                return {src: new_source, dst: new_target}
            return {src: source[:start] + target[cut:], dst: target[:cut] + source[start:]}

        # 跨车时可取出两个客户，再各自找最低惩罚的插入位置。
        if src != dst and target:
            position = rng.randrange(len(target))
            new_source = self.insert_at_best(source[:start] + source[start + 1:], target[position], penalty)
            new_target = self.insert_at_best(target[:position] + target[position + 1:], source[start], penalty)
            return {src: new_source, dst: new_target}

        # 路线内反转
        end = rng.randrange(start + 1, len(source) + 1)
        return {src: source[:start] + source[start:end][::-1] + source[end:]}

    def insert_at_best(self, route, node, penalty):  #选取惩罚代价最低的插入位置 返回list[int]插入后的路线
        trials = (route[:position] + [node] + route[position:] for position in range(len(route) + 1))
        return min(trials, key=lambda trial: self.cost(self.measure(trial), penalty))

    def advance(self, trials):  #推进软约束退火修复 返回list[list[int]]完整可行路线或None
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
            # 偶尔优先从有违约的路线生成移动。
            src = rng.randrange(len(self.routes))
            if rng.random() < 0.4:
                violated = [
                    index for index, (_, late, excess) in enumerate(self.metrics)
                    if late + excess > EPSILON
                ]
                if violated:
                    src = rng.choice(violated)
            dst = rng.randrange(len(self.routes))
            changes = self.propose(src, dst, penalty)
            if not changes:
                continue

            measured = {index: self.measure(route) for index, route in changes.items()}
            delta = sum(self.cost(metrics, penalty) - self.cost(self.metrics[index], penalty)
                        for index, metrics in measured.items())
            if delta <= 0 or rng.random() < exp(-delta / temperature):
                for index, route in changes.items():
                    self.routes[index] = route
                    self.metrics[index] = measured[index]
                self.remember()
        if self.best[0] <= EPSILON:
            return [route for route in self.best_routes if route]
        return None


def recover_fleet(state, repair, budget):  #启动或延续拆车后的修复 返回tuple修复状态和可行候选各自可为None
    """从拆一辆车的状态继续搜索，找到完整可行解才提交。"""
    routes = state.current.routes
    if len(routes) < 2:
        return None, None

    # 先判断少一辆车后的总容量是否够用。
    problem = state.problem
    demand = sum(customer.demand for customer in problem.customers.values() if customer.id)
    capacity_after_removal = (len(routes) - 1) * problem.capacity
    if demand > capacity_after_removal + EPSILON:
        return None, None

    if repair is None or len(repair.routes) + 1 != len(routes) or repair.steps >= 360000:
        by_length = sorted(range(len(routes)), key=lambda index: len(routes[index]))
        short_routes = by_length[:max(1, len(by_length) // 3)]
        remove_index = state.rng.choice(short_routes)
        repair = FleetRepair(state, remove_index)
    improved = repair.advance(budget)
    if improved is None:
        return repair, None
    replacements = {i: improved[i] if i < len(improved) else [] for i in range(len(routes))}
    return None, changed_solution(state, replacements)


# 二 尾部精修

def suffix_options(problem, prefix, original_suffix, pending, deadline):  #枚举固定前缀后的可行后缀并按客户集合保留最短接法 返回dict_values距离和后缀元组的视图
    """同一组尾部客户只保留距离最短的可行接续。"""
    last = prefix[-1]
    prefix_check = problem.check_route(prefix)
    depot_due = problem.customers[0].due
    load = sum(problem.customers[node].demand for node in prefix)
    original_mask = sum(bit for node, bit in pending if node in original_suffix)
    original_route = prefix + original_suffix
    options = {original_mask: (problem.check_route(original_route).distance, original_suffix)}
    dist, customers = problem.dist, problem.customers
    # 每位客户对应一个二进制位，用来记录后缀已选过哪些客户。
    pending_customers = [(node, bit, customers[node]) for node, bit in pending]
    visits = 0

    def extend(suffix, previous, departure, used_load, distance, mask):  #递归扩展可行后缀并更新最短接续选项 返回None
        nonlocal visits
        visits += 1
        if visits >= 30000 or perf_counter() >= deadline:
            return
        total = distance + dist[previous, 0]
        if mask not in options or total < options[mask][0]:
            options[mask] = total, suffix
        for node, bit, customer in pending_customers:
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


def polish_tails(state, width=3):  #保持车数并用整数优化重排路线尾部 返回Candidate改进后的移动与可行解或None
    """每个车选一个后缀，每个取出的客户分配一次。"""
    if len(state.current.routes) != len(state.best.routes):
        return None
    deadline = state.started + state.config.time_limit if state.config.time_limit is not None else float("inf")
    if deadline - perf_counter() < 2:
        return None

    from numpy import ones
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csc_matrix

    # 1. 固定每条路线的前缀，汇总可以重新分配的尾部客户。
    routes = state.best.routes
    segments = []
    for route in routes:
        split = max(1, len(route) - width)
        segments.append((tuple(route[:split]), tuple(route[split:])))
    pending = sorted(node for _, tail in segments for node in tail)
    if not pending:
        return None
    node_rows = {node: i for i, node in enumerate(pending)}
    bits = [(node, 1 << i) for node, i in node_rows.items()]
    enumeration_end = min(deadline - 1.5, perf_counter() + 1.2)
    # 2. 每个可行接法形成一列，同时覆盖一辆车和该后缀的客户。
    route_choices = []
    constraint_rows, choice_columns = [], []
    for route_index, (prefix, tail) in enumerate(segments):
        if perf_counter() >= enumeration_end:
            return None
        options = suffix_options(state.problem, prefix, tail, bits, enumeration_end)
        for distance, suffix in options:
            covered = [len(pending) + route_index] + [node_rows[node] for node in suffix]
            constraint_rows.extend(covered)
            choice_columns.extend([len(route_choices)] * len(covered))
            route_choices.append((route_index, distance, prefix + suffix))
    remaining = deadline - perf_counter()
    if remaining <= 0.2:
        return None
    # 3. 每辆车选一个接法，每个待分配客户恰好覆盖一次。
    num_choices = len(route_choices)
    incidence = csc_matrix(
        (ones(len(constraint_rows)), (constraint_rows, choice_columns)),
        shape=(len(pending) + len(routes), num_choices),
    )
    answer = milp([distance for _, distance, _ in route_choices], integrality=ones(num_choices),
                  bounds=Bounds(0, 1), constraints=LinearConstraint(incidence, 1, 1),
                  options={"time_limit": min(3.0, remaining - 0.1)})
    if answer.x is None:
        return None
    # 4. 对求解器选出的路线做一次完整评价，确认可行且更短。
    selected = [(index, route) for (index, _, route), value in zip(route_choices, answer.x) if value > 0.5]
    replacements = {i: list(route) for i, route in selected}
    if len(selected) != len(routes) or len(replacements) != len(routes):
        return None
    checked = state.problem.evaluate(replacements.values())
    if not checked.feasible or checked.distance >= state.best.distance - EPSILON:
        return None
    return changed_solution(state, replacements)
