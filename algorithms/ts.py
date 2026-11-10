"""禁忌搜索及可复用的单步搜索。"""
from algorithms.common import SearchConfig, SearchState, key
from algorithms.neighborhoods import candidate


def candidates(state, tabu, iteration):
    for _ in range(state.config.candidate_count):
        item = candidate(state)
        if item is None:
            continue
        move, solution = item
        forbidden = any(tabu.get(attr, 0) > iteration for attr in move.check)
        if not forbidden or key(solution) < key(state.best):
            yield item


def commit(state, item, tabu, iteration):
    move, solution = item
    for attr in move.forbid:
        tabu[attr] = iteration + state.config.tabu_tenure + 1
    return state.update(move.order, solution)


def step(state, tabu, iteration):
    choices = list(candidates(state, tabu, iteration))
    if not choices:
        return False
    return commit(state, min(choices, key=lambda item: key(item[1])), tabu, iteration)


def solve(problem, parameters=None):
    config = parameters or SearchConfig()
    state = SearchState(problem, config)
    tabu = {}
    for iteration in range(config.iterations):
        if state.time_expired():
            break
        step(state, tabu, iteration)
        state.record()
    return state.result()
