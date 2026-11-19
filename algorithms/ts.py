"""禁忌搜索：筛选候选、选择最好候选、提交移动。"""
from algorithms.common import SearchConfig, SearchState, key
from algorithms.neighborhoods import generate


def candidates(state, tabu, iteration):  #过滤不可行和未满足藐视准则的禁忌候选 返回generator逐个产出Candidate
    for attempt in range(state.config.candidate_count):
        if state.time_expired():
            break
        candidate = generate(state, allow_large_moves=attempt == 0)
        if candidate is None:
            continue
        move, solution = candidate
        is_tabu = any(tabu.get(arc, 0) > iteration for arc in move.added_arcs)
        # 能刷新历史最优的候选允许通过禁忌筛选。
        if not is_tabu or key(solution) < key(state.best):
            yield candidate


def commit(state, candidate, tabu, iteration):  #提交候选并记录删除弧的禁忌期限 返回bool是否刷新历史最优
    move, solution = candidate
    expires_at = iteration + state.config.tabu_tenure + 1
    for arc in move.removed_arcs:
        tabu[arc] = expires_at
    return state.update(solution)


def step(state, tabu, iteration):  #选择并提交一轮禁忌搜索中最好的候选 返回bool是否刷新历史最优
    best_candidate = min(candidates(state, tabu, iteration),
                         key=lambda candidate: key(candidate.solution), default=None)
    if best_candidate is None:
        return False
    return commit(state, best_candidate, tabu, iteration)


def solve(problem, parameters=None):  #运行禁忌搜索 返回SearchResult搜索结果
    config = parameters or SearchConfig()
    state = SearchState(problem, config)
    tabu = {}
    for iteration in range(config.iterations):
        if state.stopped():
            break
        step(state, tabu, iteration)
        state.record()
    return state.result()
