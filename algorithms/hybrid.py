"""三种 SA + TS 混合框架，统一返回 SearchResult。"""
from algorithms.common import SearchConfig, SearchState, accept
from algorithms.neighborhoods import candidate
from algorithms.ts import candidates, commit, step


def ts_metropolis(problem, parameters=None):
    """方案一：禁忌筛选后按 Metropolis 接受候选。"""
    config = parameters or SearchConfig()
    state = SearchState(problem, config)
    tabu = {}
    temperature = config.initial_temperature
    for iteration in range(config.iterations):
        if state.time_expired() or temperature < config.min_temperature:
            break
        for item in candidates(state, tabu, iteration):
            if accept(problem, state.current, item[1], temperature, state.rng):
                commit(state, item, tabu, iteration)
                break
        state.record()
        temperature *= config.cooling
    return state.result()


def sa_ts_inner(problem, parameters=None):
    """方案二：每个温度层进行 TS 精搜，层间按 Metropolis 决定是否转移。"""
    config = parameters or SearchConfig()
    state = SearchState(problem, config)
    tabu = {}
    temperature = config.initial_temperature
    iteration = 0
    while iteration < config.iterations and temperature >= config.min_temperature and not state.time_expired():
        start_order, start_solution = state.order[:], state.current
        for _ in range(min(config.inner_iterations, config.iterations - iteration)):
            if state.time_expired():
                break
            step(state, tabu, iteration)
            state.record()
            iteration += 1
        if not accept(problem, start_solution, state.current, temperature, state.rng):
            state.order, state.current = start_order, start_solution
        temperature *= config.cooling
    return state.result()


def _anneal(state, steps):
    """执行一段 SA，用于暖启动或停滞后的扰动。"""
    config = state.config
    temperature = config.initial_temperature
    for _ in range(steps):
        if state.time_expired():
            break
        item = candidate(state)
        if item and accept(state.problem, state.current, item[1], temperature, state.rng):
            state.update(item[0].order, item[1])
        state.record()
        temperature = max(config.min_temperature, temperature * config.cooling)


def sa_warm_ts(problem, parameters=None):
    """方案三 b：SA 暖启动、TS 精修，停滞后从最优解 SA 扰动重启。"""
    config = parameters or SearchConfig()
    state = SearchState(problem, config)
    tabu = {}
    _anneal(state, min(config.inner_iterations, config.iterations))
    stale = 0
    while state.steps < config.iterations and not state.time_expired():
        improved = step(state, tabu, state.steps)
        stale = 0 if improved else stale + 1
        state.record()
        if stale >= config.restart_after and state.steps < config.iterations:
            state.order, state.current = state.best_order[:], state.best
            tabu.clear()
            _anneal(state, min(config.inner_iterations, config.iterations - state.steps))
            stale = 0
    return state.result()
