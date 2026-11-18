#三种方案
from algorithms.common import SearchConfig, SearchState, accept
from algorithms.neighborhoods import generate
from algorithms.ts import candidates, commit, step


def ts_metropolis(problem, parameters=None): #方案一：禁忌筛选后按退火准则接受，返回搜索结果。 	SA 和 TS 每一步一起用
    config = parameters or SearchConfig()
    state = SearchState(problem, config)
    tabu = {}
    for iteration in range(config.iterations):
        if state.stopped():
            break
        temperature = state.temperature(iteration)
        for candidate in candidates(state, tabu, iteration):
            if accept(problem, state.current, candidate.solution, temperature, state.rng):
                commit(state, candidate, tabu, iteration)
                break
        state.record()
    return state.result()


def sa_ts_inner(problem, parameters=None): #方案二：外层退火、内层禁忌搜索，返回搜索结果。 	SA 控制外层，TS 在每一层连续搜索
    config = parameters or SearchConfig()
    state = SearchState(problem, config)
    tabu = {}
    iteration = 0
    while iteration < config.iterations and not state.stopped():
        temperature = state.temperature(iteration // config.inner_iterations)
        start_solution = state.current
        for _ in range(min(config.inner_iterations, config.iterations - iteration)):
            if state.stopped():
                break
            step(state, tabu, iteration)
            state.record()
            iteration += 1
        if not accept(problem, start_solution, state.current, temperature, state.rng): #Metropolis
            state.current = start_solution
    return state.result()


def _anneal(state, steps): #执行一段退火并更新状态；无返回值。 不重新构造初始解，利用之前搜索得到的解，继续进行模拟退火
    """暖启动"""
    config = state.config
    temperature = config.initial_temperature
    for _ in range(steps):
        if state.stopped():
            break
        candidate = generate(state)
        if candidate is not None:
            _, solution = candidate
            if accept(state.problem, state.current, solution, temperature, state.rng):
                state.update(solution)
        state.record()
        temperature = max(config.min_temperature, temperature * config.cooling)


def sa_warm_ts(problem, parameters=None): #方案三：退火暖启动、禁忌精修和重启，返回搜索结果  先 SA、后 TS，卡住后再用 SA 扰动
    config = parameters or SearchConfig()
    state = SearchState(problem, config)
    tabu = {}
    _anneal(state, min(config.inner_iterations, config.iterations))
    stale = 0
    while state.steps < config.iterations and not state.stopped():
        improved = step(state, tabu, state.steps)
        stale = 0 if improved else stale + 1
        state.record()
        if stale >= config.restart_after and state.steps < config.iterations:
            state.current = state.best
            tabu.clear()
            _anneal(state, min(config.inner_iterations, config.iterations - state.steps))
            stale = 0
    return state.result()
