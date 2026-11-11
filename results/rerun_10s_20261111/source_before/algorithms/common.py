#公共
from dataclasses import dataclass, field
from time import perf_counter
from math import exp
import random
from algorithms.initial import greedy_order, insertion_order


@dataclass
class SearchConfig: #算法参数
    seed: int = 0 
    iterations: int = 500 
    candidate_count: int = 30 #候选数量
    tabu_tenure: int = 10 #禁忌长度
    initial_temperature: float = 100.0
    cooling: float = 0.98
    min_temperature: float = 0.001
    inner_iterations: int = 10 #方案二每温度层的 TS 步数；方案三每段暖启动／重启退火的步数。
    restart_after: int = 50 #方案三 TS 阶段连续多少步未刷新历史最优后触发重启。
    time_limit: float = None
    neighborhoods: tuple = ("relocate", "swap", "2opt")

    # def validate(self):
    #     if self.iterations < 0 or self.candidate_count < 1 or self.tabu_tenure < 1 or self.inner_iterations < 1 or self.restart_after < 1:
    #         raise ValueError("迭代、候选数和禁忌参数无效")
    #     if self.time_limit is not None and self.time_limit <= 0:
    #         raise ValueError("时间预算必须为正数")
    #     if self.initial_temperature <= 0 or self.min_temperature <= 0 or not 0 < self.cooling <= 1:
    #         raise ValueError("温度参数无效")
    #     if not self.neighborhoods or set(self.neighborhoods) - {"relocate", "swap", "2opt"}:
    #         raise ValueError("邻域名称无效")


@dataclass
class SearchResult: #算法返回值
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


def key(solution): #按字典排序，车辆优先 返回 (车辆数, 距离)
    return (solution.vehicles, solution.distance)


def energy(problem, solution):#输入问题和解 返回 SA能量
    n = len(problem.customers) - 1
    weight = 2 * n * max(problem.dist.values(), default=0) + 1
    return weight * solution.vehicles + solution.distance


def accept(problem, current, candidate, temperature, rng):#输入问题、当前解、候选解、温度与随机数生成器 按 Metropolis 准则返回布尔值。
    delta = energy(problem, candidate) - energy(problem, current)
    return delta <= 0 or rng.random() < exp(-delta / temperature)


@dataclass
class SearchState:#一次搜索的内部状态
    problem: object
    
    config: SearchConfig

    order: list = field(init=False) #客户排列
    current: object = field(init=False) 

    best_order: list = field(init=False)
    best: object = field(init=False)
    
    history: list = field(init=False)
    rng: object = field(init=False)
    started: float = field(init=False)
    steps: int = 0

    def __post_init__(self):
        # self.config.validate()
        self.started = perf_counter()
        self.rng = random.Random(self.config.seed)
        orders = [insertion_order(self.problem), greedy_order(self.problem)]
        feasible = [(order, self.problem.decode(order)) for order in orders]
        feasible = [(order, solution) for order, solution in feasible if solution.feasible]

        self.order, self.current = min(feasible, key=lambda pair: key(pair[1]))
        self.best_order, self.best = self.order[:], self.current
        self.history = [(self.best.vehicles, self.best.distance)]

    def update(self, order, solution): #刷新历史最优返回1，否则返回0
        self.order, self.current = order, solution
        if key(solution) < key(self.best):
            self.best_order, self.best = order[:], solution
            return True
        return False

    def record(self): #历史最优加入history 无返回
        self.steps += 1
        self.history.append((self.best.vehicles, self.best.distance))

    def time_expired(self): #时间上限 返回true/false
        return self.config.time_limit is not None and perf_counter() - self.started >= self.config.time_limit

    def result(self): #构造 返回SearchResult
        return SearchResult(self.best, self.history, perf_counter()-self.started, self.steps, self.config.seed)
