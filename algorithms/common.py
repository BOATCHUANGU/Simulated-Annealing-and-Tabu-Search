"""搜索共享的参数、状态和目标比较。"""
from dataclasses import dataclass, field
from time import perf_counter
from math import exp
import random
from algorithms.initial import greedy_order, insertion_order


@dataclass
class SearchConfig:
    """一次搜索使用的参数；可通过 seed 和 iterations 控制复现。"""
    seed: int = 0
    iterations: int = 500 
    candidate_count: int = 30
    tabu_tenure: int = 10
    initial_temperature: float = 100.0
    cooling: float = 0.98
    min_temperature: float = 0.001
    inner_iterations: int = 10
    restart_after: int = 50
    time_limit: float = None
    neighborhoods: tuple = ("relocate", "swap", "2opt")

    def validate(self):
        if self.iterations < 0 or self.candidate_count < 1 or self.tabu_tenure < 1 or self.inner_iterations < 1 or self.restart_after < 1:
            raise ValueError("迭代、候选数和禁忌参数无效")
        if self.time_limit is not None and self.time_limit <= 0:
            raise ValueError("时间预算必须为正数")
        if self.initial_temperature <= 0 or self.min_temperature <= 0 or not 0 < self.cooling <= 1:
            raise ValueError("温度参数无效")
        if not self.neighborhoods or set(self.neighborhoods) - {"relocate", "swap", "2opt"}:
            raise ValueError("邻域名称无效")


@dataclass
class SearchResult:
    """搜索结束后的最优解、历史记录和运行信息。"""
    solution: object
    history: list
    runtime: float
    iterations: int
    seed: int

    @property
    def vehicles(self):
        return self.solution.vehicles

    @property
    def distance(self):
        return self.solution.distance


def key(solution):
    """先比较车辆数，再比较距离。"""
    return (solution.vehicles, solution.distance)


def energy(problem, solution):
    # 任意解距离不超过 (客户数 + 路径数) × 最大边长，保证车数优先。
    n = len(problem.customers) - 1
    weight = 2 * n * max(problem.dist.values(), default=0) + 1
    return weight * solution.vehicles + solution.distance


def accept(problem, current, candidate, temperature, rng):
    delta = energy(problem, candidate) - energy(problem, current)
    return delta <= 0 or rng.random() < exp(-delta / temperature)


@dataclass
class SearchState:
    """order/current 为当前解；best_order/best 为历史最优解。"""
    problem: object
    config: SearchConfig

    order: list = field(init=False)
    current: object = field(init=False)

    best_order: list = field(init=False)
    best: object = field(init=False)
    
    history: list = field(init=False)
    rng: object = field(init=False)
    started: float = field(init=False)
    steps: int = 0

    def __post_init__(self):
        self.config.validate()
        self.started = perf_counter()
        self.rng = random.Random(self.config.seed)
        orders = [insertion_order(self.problem), greedy_order(self.problem)]
        feasible = [(order, self.problem.decode(order)) for order in orders]
        feasible = [(order, solution) for order, solution in feasible if solution.feasible]
        if not feasible:
            raise ValueError("初始解不可行：请调整构造方法或检查实例")
        self.order, self.current = min(feasible, key=lambda pair: key(pair[1]))
        self.best_order, self.best = self.order[:], self.current
        self.history = [(self.best.vehicles, self.best.distance)]

    def update(self, order, solution):
        self.order, self.current = order, solution
        if key(solution) < key(self.best):
            self.best_order, self.best = order[:], solution
            return True
        return False

    def record(self):
        self.steps += 1
        self.history.append((self.best.vehicles, self.best.distance))

    def time_expired(self):
        return self.config.time_limit is not None and perf_counter() - self.started >= self.config.time_limit

    def result(self):
        return SearchResult(self.best, self.history, perf_counter()-self.started, self.steps, self.config.seed)
