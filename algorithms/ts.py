from algorithms.common import SearchConfig, SearchState, key
from algorithms.neighborhoods import generate


def candidates(state, tabu, iteration): #筛选可行且允许的候选，逐个产生 (移动, 解)。
    for attempt in range(state.config.candidate_count):
        if state.time_expired():
            break
        candidate = generate(state, allow_large_moves=attempt == 0)
        if candidate is None:
            continue
        move, solution = candidate
        forbidden = any(tabu.get(attr, 0) > iteration for attr in move.check) #禁忌
        if not forbidden or key(solution) < key(state.best): #藐视
            yield candidate


def commit(state, candidate, tabu, iteration): #SearchState 提交候选解 更新禁忌表 更新当前搜索状态；提交候选并更新禁忌表  返回是否刷新历史最优。
    move, solution = candidate
    for attr in move.forbid:
        tabu[attr] = iteration + state.config.tabu_tenure + 1 #SearchConfig
    return state.update(solution)


def step(state, tabu, iteration): #进行一次搜索；执行一轮禁忌搜索 返回是否刷新历史最优。
    best_candidate = min(candidates(state, tabu, iteration),
                         key=lambda candidate: key(candidate.solution), default=None)
    if best_candidate is None:
        return False
    return commit(state, best_candidate, tabu, iteration)


def solve(problem, parameters=None): #运行禁忌搜索 返回包含最优解和运行记录的结果。
    config = parameters or SearchConfig()
    state = SearchState(problem, config)
    tabu = {}
    for iteration in range(config.iterations):
        if state.stopped():
            break
        step(state, tabu, iteration)
        state.record()
    return state.result()
