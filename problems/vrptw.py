"""Solomon 实例与 VRPTW 路径评价，不包含搜索算法。"""

from dataclasses import dataclass
from math import hypot
from pathlib import Path


@dataclass(frozen=True)
class Customer:
    id: int
    x: float
    y: float
    demand: float
    ready: float
    due: float
    service: float


@dataclass
class RouteResult:
    feasible: bool
    distance: float
    start_times: list
    reason: str = ""


@dataclass
class Solution:
    feasible: bool
    routes: list
    distance: float
    details: list
    reason: str = ""

    @property
    def vehicles(self):
        return len(self.routes)


class VRPTW:
    def __init__(self, customers, capacity, vehicle_limit):
        self.customers = {c.id: c for c in customers}
        if len(self.customers) != len(customers) or 0 not in self.customers:
            raise ValueError("客户编号重复或缺少车场 0")
        self.capacity = capacity
        self.vehicle_limit = vehicle_limit
        self.dist = {
            (a.id, b.id): hypot(a.x - b.x, a.y - b.y)
            for a in customers
            for b in customers
        }

    @classmethod
    def from_solomon(cls, path, customer_limit=None):
        lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
        try:
            v = next(
                i
                for i, line in enumerate(lines)
                if "NUMBER" in line and "CAPACITY" in line
            )
            vehicle_limit, capacity = map(float, lines[v + 1].split())
            start = next(i for i, line in enumerate(lines) if "CUST NO." in line) + 1
            rows = [line.split() for line in lines[start:] if line.strip()]
            if customer_limit is not None:
                rows = rows[: customer_limit + 1]
            customers = [Customer(int(r[0]), *map(float, r[1:])) for r in rows]
            if len(customers) < 2 or capacity <= 0 or vehicle_limit <= 0:
                raise ValueError("缺少客户或车辆参数无效")
            if customer_limit is not None and len(customers) != customer_limit + 1:
                raise ValueError("客户数量不足")
            return cls(customers, capacity, int(vehicle_limit))
        except (StopIteration, IndexError, ValueError) as exc:
            raise ValueError(f"Solomon 文件格式无效: {path}: {exc}") from exc

    def check_route(self, route):
        depot = self.customers[0]
        time, load, distance, last = depot.ready, 0.0, 0.0, 0
        starts = []
        for node in route:
            if node == 0 or node not in self.customers:
                return RouteResult(False, distance, starts, "客户编号无效")
            c = self.customers[node]
            load += c.demand
            distance += self.dist[last, node]
            time = max(time + self.dist[last, node], c.ready)
            starts.append(time)
            if load > self.capacity + 1e-9:
                return RouteResult(False, distance, starts, "超出容量")
            if time > c.due + 1e-9:
                return RouteResult(False, distance, starts, "超过客户时间窗")
            time += c.service
            last = node
        distance += self.dist[last, 0]
        if time + self.dist[last, 0] > depot.due + 1e-9:
            return RouteResult(False, distance, starts, "返回车场超时")
        return RouteResult(True, distance, starts)

    def evaluate(self, routes):
        routes = [list(route) for route in routes]
        details = [self.check_route(route) for route in routes]
        visited = [node for route in routes for node in route]
        expected = set(self.customers) - {0}
        reason = ""
        if any(not route for route in routes):
            reason = "空路径"
        elif len(routes) > self.vehicle_limit:
            reason = "车辆数超限"
        elif len(visited) != len(expected) or set(visited) != expected:
            reason = "客户重复、缺失或编号无效"
        elif any(not r.feasible for r in details):
            reason = next(r.reason for r in details if not r.feasible)
        return Solution(
            not reason, routes, sum(r.distance for r in details), details, reason
        )

    def decode(self, order):
        """按排列顺序依次加入当前车辆，不可行则新开车辆。"""
        routes, route = [], []
        for node in order:
            if route and not self.check_route(route + [node]).feasible:
                routes.append(route)
                route = []
            route.append(node)
        if route:
            routes.append(route)
        return self.evaluate(routes)
