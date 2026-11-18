"""SA、TS 和混合方案共用的六种邻域。"""
from algorithms.fleet_repair import recover_fleet
from algorithms.routes import EPSILON, changed_solution, ejection, perturb, repair
from algorithms.tail_polish import polish_tails


# 邻域选择与长预算扩展
SHORT_ROUTE_ELIMINATION_PROBABILITY = 0.20
REBUILD_PROBABILITY = 0.50
FLEET_REPAIR_STALL_STEPS = 40
FLEET_REPAIR_TRIALS = 1500
TAIL_POLISH_STALL_STEPS = 160

# 局部移位与整车消除
RELOCATE_LENGTH_CHOICES = (1, 1, 1, 2, 3)
ELIMINATION_ATTEMPTS = 96
EJECTION_CACHE_LIMIT = 600
ELIMINATION_INSERTION_NOISE = 3
DOUBLE_EJECTION_START = 24
DOUBLE_EJECTION_PROBABILITY = 0.18
EJECTION_PERTURBATIONS = 5

# 多点重组
REBUILD_CUSTOMER_RANGE = (5, 23)
REBUILD_INSERTION_NOISE_CHOICES = (0, 3, 10)
NEARBY_CUSTOMER_NOISE = 25
TIME_WINDOW_CUSTOMER_NOISE = 55
REBUILD_SEGMENT_MAX_LENGTH = 4
REBUILD_SUFFIX_PROBABILITY = 0.65


def generate(state, *, allow_large_moves=True):
    """从当前路线生成一次候选，返回（移动，可行解）或 None。"""
    if not state.current.routes or state.time_expired():
        return None

    config = state.config
    problem, rng = state.problem, state.rng
    routes = state.current.routes
    kinds = config.neighborhoods
    local_kinds = tuple(kind for kind in kinds if kind not in ("eliminate", "rebuild"))
    target = config.target_vehicles
    at_target = target is not None and len(routes) <= target
    can_eliminate = "eliminate" in kinds and (target is None or len(routes) > target)
    steps_without_improvement = state.steps - state.best_step

    # 长预算模式停滞时，先尝试减车修复或尾部精修。
    if (allow_large_moves and config.fleet_repair and "rebuild" in kinds and at_target
            and steps_without_improvement >= TAIL_POLISH_STALL_STEPS and state.tail_base is not state.best):
        state.tail_base = state.best
        polished = polish_tails(state)
        if polished is not None:
            return polished
    if (allow_large_moves and can_eliminate and config.fleet_repair
            and steps_without_improvement >= FLEET_REPAIR_STALL_STEPS):
        state.fleet_attempt, improved = recover_fleet(state, state.fleet_attempt, FLEET_REPAIR_TRIALS)
        if improved is not None:
            return improved

    # 选定邻域，再进入对应的操作板块。
    prefer_short_route = (allow_large_moves and can_eliminate
                          and rng.random() < SHORT_ROUTE_ELIMINATION_PROBABILITY)
    if prefer_short_route:
        kind = "eliminate"
    elif (allow_large_moves and "rebuild" in kinds
          and (not local_kinds or rng.random() < REBUILD_PROBABILITY)):
        kind = "rebuild"
    elif local_kinds:
        kind = rng.choice(local_kinds)
    elif allow_large_moves and can_eliminate:
        kind = "eliminate"
    else:
        return None

    # 局部操作先抽取起始路线。路线内反转只需要这条路线。
    if kind in local_kinds:
        source_index = rng.randrange(len(routes))
        source = routes[source_index]
        if kind != "2opt":
            target_index = rng.randrange(len(routes))
            destination = routes[target_index]
            start = rng.randrange(len(source))

    # 1. 移位：把一段连续客户移到同一条路线或另一条路线。
    if kind == "relocate":
        length = min(len(source) - start, rng.choice(RELOCATE_LENGTH_CHOICES))
        segment = source[start:start + length]
        shortened = source[:start] + source[start + length:]
        receiving_route = shortened if source_index == target_index else destination
        position = rng.randrange(len(receiving_route) + 1)
        extended = receiving_route[:position] + segment + receiving_route[position:]
        replacements = {source_index: shortened}
        replacements[target_index] = extended
        return changed_solution(state, replacements)

    # 2. 交换：互换两个客户，支持同车和跨车。
    if kind == "swap":
        position = rng.randrange(len(destination))
        if source_index == target_index:
            swapped = source[:]
            swapped[start], swapped[position] = swapped[position], swapped[start]
            return changed_solution(state, {source_index: swapped})
        new_source = source[:start] + [destination[position]] + source[start + 1:]
        new_destination = destination[:position] + [source[start]] + destination[position + 1:]
        return changed_solution(state, {source_index: new_source, target_index: new_destination})

    # 3. 路线内 2-opt：反转一条路线的连续片段。
    if kind == "2opt":
        if len(source) < 2:
            return None
        left, right = sorted(rng.sample(range(len(source) + 1), 2))
        reversed_route = source[:left] + source[left:right][::-1] + source[right:]
        return changed_solution(state, {source_index: reversed_route})

    # 4. 跨车 2-opt*：在两条路线各选切点并交换后缀。
    if kind == "2opt_star":
        if source_index == target_index:
            return None
        source_cut = rng.randrange(len(source) + 1)
        target_cut = rng.randrange(len(destination) + 1)
        new_source = source[:source_cut] + destination[target_cut:]
        new_destination = destination[:target_cut] + source[source_cut:]
        return changed_solution(state, {source_index: new_source, target_index: new_destination})

    # 5. 整车消除：腾空一辆车，必要时逐出其他客户再重新插入。
    if kind == "eliminate":
        if prefer_short_route:
            by_length = sorted(range(len(routes)), key=lambda index: len(routes[index]))
            removed_index = rng.choice(by_length[:max(1, len(by_length) // 2)])
        else:
            removed_index = rng.randrange(len(routes))

        trial_routes = [route[:] for route in routes]
        pending = trial_routes[removed_index]
        trial_routes[removed_index] = []
        remaining_routes = [route for index, route in enumerate(trial_routes) if index != removed_index]
        if not remaining_routes:
            return None
        total_demand = sum(customer.demand for customer in problem.customers.values() if customer.id)
        if total_demand > len(remaining_routes) * problem.capacity + EPSILON:
            return None

        penalties = {node: 1 for node in problem.customers}
        variants = {}
        for attempt in range(ELIMINATION_ATTEMPTS):
            if state.time_expired():
                return None
            if len(variants) > EJECTION_CACHE_LIMIT:
                variants.clear()
            pending = repair(problem, remaining_routes, pending, rng, state.time_expired,
                             noise=ELIMINATION_INSERTION_NOISE)
            if not pending:
                return changed_solution(state, dict(enumerate(trial_routes)))

            node = rng.choice(pending)
            penalties[node] += 1
            choice = ejection(problem, remaining_routes, node, penalties, rng, variants, 1, state.time_expired)
            if choice is None or (attempt >= DOUBLE_EJECTION_START
                                  and rng.random() < DOUBLE_EJECTION_PROBABILITY):
                pair = ejection(problem, remaining_routes, node, penalties, rng, variants, 2, state.time_expired)
                if pair is not None:
                    choice = pair
            if choice is None:
                return None

            route_index, position, label = choice
            route = remaining_routes[route_index]
            pending.remove(node)
            pending.extend(ejected for ejected in route if ejected not in label.nodes)
            route[:] = label.nodes[1:position + 1] + (node,) + label.nodes[position + 1:-1]
            for _ in range(EJECTION_PERTURBATIONS):
                perturb(problem, remaining_routes, rng)
        return None

    # 6. 多点重组：选出一组客户，再分配回各条路线。
    if kind == "rebuild":
        trial_routes = [route[:] for route in routes]
        nodes = state.order
        rng.shuffle(nodes)
        count = min(len(nodes), rng.randint(*REBUILD_CUSTOMER_RANGE))
        mode = rng.randrange(5 if at_target else 3)
        if mode == 0:
            anchor = nodes[0]
            nodes.sort(key=lambda node: problem.dist[anchor, node] + rng.uniform(0, NEARBY_CUSTOMER_NOISE))
        elif mode == 1:
            nodes = list(dict.fromkeys(rng.choice(trial_routes) + nodes))
        elif mode == 2:
            # 优先取出时间窗接近的客户。
            ready = problem.customers[nodes[0]].ready
            nodes.sort(key=lambda node: abs(problem.customers[node].ready - ready)
                       + rng.uniform(0, TIME_WINDOW_CUSTOMER_NOISE))
        elif mode == 3:
            # 优先取出多条路线的首段或尾段。模式 4 沿用随机顺序。
            shuffled_routes = trial_routes[:]
            rng.shuffle(shuffled_routes)
            take_suffix = rng.random() < REBUILD_SUFFIX_PROBABILITY
            segments = []
            for route in shuffled_routes:
                length = rng.randint(1, min(len(route), REBUILD_SEGMENT_MAX_LENGTH))
                segments.extend(route[-length:] if take_suffix else route[:length])
            nodes = list(dict.fromkeys(segments + nodes))

        pending = nodes[:count]
        removed = set(pending)
        remaining_routes = [[node for node in route if node not in removed] for route in trial_routes]
        noise = rng.choice(REBUILD_INSERTION_NOISE_CHOICES)
        if repair(problem, remaining_routes, pending, rng, state.time_expired, noise=noise):
            return None
        return changed_solution(state, dict(enumerate(remaining_routes)))

    raise ValueError(f"未知邻域：{kind}")
