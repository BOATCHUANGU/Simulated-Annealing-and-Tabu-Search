from algorithms.common import SearchConfig, SearchState, key
from algorithms.neighborhoods import candidate


def candidates(state, tabu, iteration): #筛选可行且允许的候选，逐个产生 (移动, 解)。
    for _ in range(state.config.candidate_count):
        item = candidate(state)
        if item is None:
            continue
        move, solution = item
        forbidden = any(tabu.get(attr, 0) > iteration for attr in move.check) #禁忌
        if not forbidden or key(solution) < key(state.best): #藐视
            yield item


def commit(state, item, tabu, iteration): #SearchState 提交候选解 更新禁忌表 更新当前搜索状态；提交候选并更新禁忌表  返回是否刷新历史最优。
    move, solution = item
    for attr in move.forbid:
        tabu[attr] = iteration + state.config.tabu_tenure + 1 #SearchConfig
    return state.update(move.order, solution)


def step(state, tabu, iteration): #进行一次搜索；执行一轮禁忌搜索 返回是否刷新历史最优。
    choices = list(candidates(state, tabu, iteration))
    if not choices:
        return False
    return commit(state, min(choices, key=lambda item: key(item[1])), tabu, iteration)


def solve(problem, parameters=None): #运行禁忌搜索 返回包含最优解和运行记录的结果。
    config = parameters or SearchConfig()
    state = SearchState(problem, config)
    tabu = {}
    for iteration in range(config.iterations):
        if state.time_expired():
            break
        step(state, tabu, iteration)
        state.record()
    return state.result()
