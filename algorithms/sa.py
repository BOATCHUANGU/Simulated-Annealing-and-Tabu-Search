"""模拟退火的单步操作与主循环。"""
from algorithms.common import SearchConfig, SearchState, accept
from algorithms.neighborhoods import generate


def anneal_step(state, temperature):  #生成一个候选并按退火准则决定是否更新 返回None
    candidate = generate(state)
    if candidate is None:
        return

    solution = candidate.solution
    if accept(state.current, solution, temperature, state.rng):
        state.update(solution)


def solve(problem, parameters=None):  #运行模拟退火搜索 返回SearchResult搜索结果
    config = parameters or SearchConfig()
    state = SearchState(problem, config)
    for iteration in range(config.iterations):
        if state.stopped():
            break
        temperature = state.temperature(iteration)
        anneal_step(state, temperature)
        state.record()
    return state.result()
