#公共
from dataclasses import dataclass, field
from time import perf_counter
from math import exp
import random
from algorithms.initial import greedy_order, insertion_order, sequential_routes
from problems.vrptw import Solution


NEIGHBORHOODS = ("relocate", "swap", "2opt", "2opt_star", "eliminate", "rebuild")


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
    neighborhoods: tuple = NEIGHBORHOODS
    fleet_repair: bool = False
    target_vehicles: int = None
    target_distance: float = None

    def validate(self):
        if self.iterations < 0 or self.candidate_count < 1 or self.tabu_tenure < 1 or self.inner_iterations < 1 or self.restart_after < 1:
            raise ValueError("迭代、候选数和禁忌参数无效")
        if self.time_limit is not None and self.time_limit <= 0:
            raise ValueError("时间预算必须为正数")
        if self.initial_temperature <= 0 or self.min_temperature <= 0 or not 0 < self.cooling <= 1:
            raise ValueError("温度参数无效")
        if self.target_vehicles is not None and self.target_vehicles < 1:
            raise ValueError("目标车辆数必须为正数")
        if self.target_distance is not None and (self.target_vehicles is None or self.target_distance < 0):
            raise ValueError("设置距离目标时须同时给定车辆数，且距离不能为负")
        if not self.neighborhoods or set(self.neighborhoods).difference(NEIGHBORHOODS):
            raise ValueError("邻域名称无效")


@dataclass
class SearchResult: #算法返回值
    solution: object
    history: list
    runtime: float
    iterations: int
    seed: int
    history_times: list = field(default_factory=list)
    stop_reason: str = ""
    initial_solutions: dict = field(default_factory=dict)

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
    if candidate.vehicles != current.vehicles:
        return candidate.vehicles < current.vehicles
    delta = candidate.distance - current.distance
    return delta <= 0 or rng.random() < exp(-delta / temperature)


@dataclass
class SearchState:#一次搜索的内部状态
    problem: object
    
    config: SearchConfig

    current: Solution = field(init=False)
    best: Solution = field(init=False)
    
    history: list = field(init=False)
    rng: object = field(init=False) #按 seed 初始化的 random.Random
    started: float = field(init=False)# 计时起点。
    history_times: list = field(init=False) #历史记录对应的运行时间
    steps: int = 0
    best_step: int = field(default=0, init=False)
    fleet_attempt: object = field(default=None, init=False)
    tail_base: object = field(default=None, init=False)

    def __post_init__(self):
        self.config.validate()
        self.started = perf_counter()
        self.rng = random.Random(self.config.seed)
        initial = {
            "insertion": self.problem.decode(insertion_order(self.problem)),
            "nearest": self.problem.decode(greedy_order(self.problem)),
        }
        for name, seed_key, weight in (
            ("farthest", lambda node: -self.problem.dist[0, node], 0.1),
            ("earliest_due", lambda node: self.problem.customers[node].due, 0.25),
            ("earliest_ready", lambda node: self.problem.customers[node].ready, 0.1),
        ):
            solution = sequential_routes(self.problem, seed_key, weight, self.time_expired)
            if solution is None:
                break
            initial[name] = solution
        feasible = [solution for solution in initial.values() if solution.feasible]
        if not feasible:
            raise ValueError("初始解不可行：请调整构造方法或检查实例")
        self.initial_solutions = {name: (sol.vehicles, sol.distance, sol.feasible)
                                  for name, sol in initial.items()}
        self.current = min(feasible, key=key)
        self.best = self.current
        self.history = [(self.best.vehicles, self.best.distance)]
        self.history_times = [perf_counter() - self.started]

    @property
    def order(self):
        """按当前路线展开客户顺序。每次返回一个可以独立修改的列表。"""
        return [node for route in self.current.routes for node in route]

    def update(self, solution):
        """提交当前解，返回是否刷新历史最优。"""
        self.current = solution
        if key(solution) < key(self.best):
            self.best = solution
            self.best_step = self.steps
            return True
        return False

    def record(self): #历史最优加入history 无返回
        self.steps += 1
        self.history.append((self.best.vehicles, self.best.distance))
        self.history_times.append(perf_counter() - self.started)

    def time_expired(self): #时间上限 返回true/false
        return self.config.time_limit is not None and perf_counter() - self.started >= self.config.time_limit

    def temperature(self, stage):
        if self.config.time_limit is not None:
            elapsed = perf_counter() - self.started
            progress = min(1, elapsed / self.config.time_limit)
            stage = max(stage, 1500 * progress)
        ceiling = self.config.initial_temperature
        floor = min(ceiling, self.config.min_temperature)
        if self.config.cooling == 1:
            return ceiling
        # 降温至下限后继续搜索，预算内优先安排较低温度的精修。
        return max(floor, ceiling * self.config.cooling ** stage)

    def reached_target(self):
        vehicles, distance = self.config.target_vehicles, self.config.target_distance
        return distance is not None and key(self.best) <= (vehicles, distance)

    def stopped(self):
        return self.reached_target() or self.time_expired()

    def result(self, reason=None):
        if reason is None:
            if self.reached_target():
                reason = "target"
            elif self.time_expired():
                reason = "time_limit"
            elif self.steps >= self.config.iterations:
                reason = "iterations"
            else:
                reason = "completed"
        return SearchResult(self.best, self.history, perf_counter() - self.started,
                            self.steps, self.config.seed, self.history_times, reason,
                            self.initial_solutions)
