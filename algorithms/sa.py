from algorithms.common import SearchConfig, SearchState, accept
from algorithms.neighborhoods import generate


def solve(problem, parameters=None): #运行模拟退火，返回包含最优解和运行记录的结果。
    config = parameters or SearchConfig()
    state = SearchState(problem, config)
    for iteration in range(config.iterations):
        if state.stopped():
            break
        temperature = state.temperature(iteration)
        candidate = generate(state)
        if candidate is not None:
            _, solution = candidate
            if accept(problem, state.current, solution, temperature, state.rng):
                state.update(solution)
        state.record()
    return state.result()
