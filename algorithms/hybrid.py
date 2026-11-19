"""三种模拟退火与禁忌搜索的混合方案。"""
from algorithms.common import SearchConfig, SearchState, accept
from algorithms.sa import anneal_step
from algorithms.ts import candidates, commit, step


def ts_metropolis(problem, parameters=None):  #执行先禁忌筛选再退火接受的方案一 返回SearchResult搜索结果
    config = parameters or SearchConfig()
    state = SearchState(problem, config)
    tabu = {}
    for iteration in range(config.iterations):
        if state.stopped():
            break
        temperature = state.temperature(iteration)
        for candidate in candidates(state, tabu, iteration):
            if accept(state.current, candidate.solution, temperature, state.rng):
                commit(state, candidate, tabu, iteration)
                break
        state.record()
    return state.result()


def sa_ts_inner(problem, parameters=None):  #执行外层退火和内层禁忌搜索的方案二 返回SearchResult搜索结果
    config = parameters or SearchConfig()
    state = SearchState(problem, config)
    tabu = {}
    while state.steps < config.iterations and not state.stopped():
        # 一个温度层包含若干步 TS，接受判断针对整段搜索的终点。
        layer = state.steps // config.inner_iterations
        temperature = state.temperature(layer)
        start_solution = state.current
        layer_steps = min(config.inner_iterations, config.iterations - state.steps)
        for _ in range(layer_steps):
            if state.stopped():
                break
            step(state, tabu, state.steps)
            state.record()
        if not accept(start_solution, state.current, temperature, state.rng):
            state.current = start_solution
    return state.result()


def _warm_anneal(state, steps):  #在现有状态上执行指定步数的退火暖启动或重启 返回None
    config = state.config
    temperature = config.initial_temperature
    for _ in range(steps):
        if state.stopped():
            break
        anneal_step(state, temperature)
        state.record()
        temperature = max(config.min_temperature, temperature * config.cooling)


def sa_warm_ts(problem, parameters=None):  #执行退火暖启动和禁忌精修的方案三 返回SearchResult搜索结果
    config = parameters or SearchConfig()
    state = SearchState(problem, config)
    tabu = {}
    _warm_anneal(state, min(config.inner_iterations, config.iterations))
    steps_without_improvement = 0
    while state.steps < config.iterations and not state.stopped():
        improved = step(state, tabu, state.steps)
        steps_without_improvement = 0 if improved else steps_without_improvement + 1
        state.record()
        if steps_without_improvement >= config.restart_after and state.steps < config.iterations:
            state.current = state.best
            tabu.clear()
            _warm_anneal(state, min(config.inner_iterations, config.iterations - state.steps))
            steps_without_improvement = 0
    return state.result()
