"""搜索配置、状态和停止条件。"""
from dataclasses import dataclass, field
from time import perf_counter
from math import exp
import random
from algorithms.initial import greedy_order, insertion_order, sequential_routes
from problems.vrptw import Solution


NEIGHBORHOODS = ("relocate", "swap", "2opt", "2opt_star", "eliminate", "rebuild")


@dataclass
class SearchConfig:
    seed: int = 0  #随机种子
    iterations: int = 500  #最大迭代轮数
    candidate_count: int = 30  #候选数
    tabu_tenure: int = 10  #禁忌弧保留的轮数
    initial_temperature: float = 100.0  #起始温度
    cooling: float = 0.98  #几何降温系数
    min_temperature: float = 0.001  #温度下限
    inner_iterations: int = 10  #方案二每层的TS步数 也是方案三每次暖启动或重启的SA步数
    restart_after: int = 50  #方案三在TS阶段连续多少步未刷新最优时重启

    time_limit: float = None  #时间上限
    neighborhoods: tuple = NEIGHBORHOODS
    fleet_repair: bool = False  #是否启用长预算的减车修复和尾部精修
    target_vehicles: int = None  #目标车辆数 用于减车和精修阶段的切换
    target_distance: float = None  #达标停止的目标距离 需同时设置目标车辆数

    def validate(self):  #检查搜索参数是否合法 返回None
        if (self.iterations < 0 or self.candidate_count < 1 or self.tabu_tenure < 1
                or self.inner_iterations < 1 or self.restart_after < 1):
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
class SearchResult:
    solution: object  #搜索期间找到的历史最优可行解
    history: list  #按记录顺序保存的历史最优车数和距离 含初始解
    runtime: float  #本次搜索实际使用的墙钟时间 单位秒
    iterations: int  #本次实际完成的迭代轮数
    seed: int  #本次搜索使用的随机种子
    history_times: list = field(default_factory=list)  #与history逐项对应的累计耗时 单位秒
    stop_reason: str = ""  #停止原因 例如达标 超时或达到迭代上限
    initial_solutions: dict = field(default_factory=dict)  #各构造方法的车数 距离和可行性

    @property
    def vehicles(self):  #读取最优解的车辆数 返回int车辆数
        return self.solution.vehicles

    @property
    def distance(self):  #读取最优解的总行驶距离 返回float总距离
        return self.solution.distance


@dataclass
class SearchState:
    problem: object  #待求解的VRPTW实例 提供客户数据 距离 容量和时间窗

    config: SearchConfig  #本次运行的搜索参数

    current: Solution = field(init=False)  #当前参与邻域搜索的可行路线解
    best: Solution = field(init=False)  #至今按车辆数优先找到的最优可行解

    history: list = field(init=False)  #包括初始解在内的历史最优车数和距离记录
    rng: object = field(init=False)  #由随机种子初始化的随机数生成器
    started: float = field(init=False)  #初始化前记录的墙钟计时起点
    history_times: list = field(init=False)  #各历史记录相对计时起点的累计秒数
    steps: int = 0  #本次搜索已经记录的迭代步数
    best_step: int = field(default=0, init=False)  #最近一次刷新历史最优时的步数 用于判断停滞
    fleet_attempt: object = field(default=None, init=False)  #可跨迭代延续的软约束减车修复状态
    tail_base: object = field(default=None, init=False)  #最近一次用于尾部精修尝试的历史最优解

    def __post_init__(self):  #构造初始解并初始化计时随机数和历史记录 返回None
        self.config.validate()
        self.started = perf_counter()
        self.rng = random.Random(self.config.seed)
        # 先比较两种旧构造法，再补上三种种子选择策略。
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
                                  for name, sol in initial.items()}  #按构造方法名称保存车数 距离和可行性
        self.current = min(feasible, key=key)
        self.best = self.current
        self.history = [(self.best.vehicles, self.best.distance)]
        self.history_times = [perf_counter() - self.started]

    @property
    def order(self):  #按当前路线展开客户顺序 返回list[int]客户编号
        """按当前路线展开客户顺序。每次返回一个可以独立修改的列表。"""
        return [node for route in self.current.routes for node in route]

    def update(self, solution):  #提交当前解并尝试刷新历史最优 返回bool是否刷新历史最优
        """提交当前解，返回是否刷新历史最优。"""
        self.current = solution
        if key(solution) < key(self.best):
            self.best = solution
            self.best_step = self.steps
            return True
        return False

    def record(self):  #增加迭代计数并记录历史最优与耗时 返回None
        self.steps += 1
        self.history.append((self.best.vehicles, self.best.distance))
        self.history_times.append(perf_counter() - self.started)

    def time_expired(self):  #检查是否达到墙钟时间上限 返回bool是否超时
        return self.config.time_limit is not None and perf_counter() - self.started >= self.config.time_limit

    def temperature(self, stage):  #按搜索进度计算并限制退火温度 返回float当前温度
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

    def reached_target(self):  #按车辆数和距离的字典序检查是否达标 返回bool是否达标
        vehicles, distance = self.config.target_vehicles, self.config.target_distance
        return distance is not None and key(self.best) <= (vehicles, distance)

    def stopped(self):  #检查是否达标或超时 返回bool是否应停止搜索
        return self.reached_target() or self.time_expired()

    def result(self, reason=None):  #汇总最优解耗时历史记录和停止原因 返回SearchResult搜索结果
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


def key(solution):  #生成车辆数优先的比较键 返回tuple车辆数和总距离
    return solution.vehicles, solution.distance


def accept(current, candidate, temperature, rng):  #按车辆优先和Metropolis准则判断候选是否接受 返回bool是否接受
    if candidate.vehicles != current.vehicles:
        return candidate.vehicles < current.vehicles
    delta = candidate.distance - current.distance
    return delta <= 0 or rng.random() < exp(-delta / temperature)
