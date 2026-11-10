"""纯模拟退火。"""
from algorithms.common import SearchConfig, SearchState, accept
from algorithms.neighborhoods import candidate


def solve(problem, parameters=None):
    config = parameters or SearchConfig()
    state = SearchState(problem, config)
    temperature = config.initial_temperature
    for _ in range(config.iterations):
        if state.time_expired() or temperature < config.min_temperature:
            break
        item = candidate(state)
        if item and accept(problem, state.current, item[1], temperature, state.rng):
            state.update(item[0].order, item[1])
        state.record()
        temperature *= config.cooling
    return state.result()
