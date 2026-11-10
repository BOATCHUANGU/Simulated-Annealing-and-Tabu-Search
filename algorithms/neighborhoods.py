"""排列邻域和移动禁忌属性。"""
from dataclasses import dataclass


@dataclass
class Move:
    order: list
    changed_at: int
    check: tuple
    forbid: tuple


def edge(a, b):
    return ("edge", min(a, b), max(a, b))


def generate(order, solution, kind, rng):
    n = len(order)
    if n < 2:
        return None
    candidate = order[:]
    if kind == "swap":
        i, j = rng.sample(range(n), 2)
        a, b = order[i], order[j]
        candidate[i], candidate[j] = candidate[j], candidate[i]
        pair = ("swap", min(a, b), max(a, b))
        return Move(candidate, min(i, j), (pair,), (pair,))
    if kind == "relocate":
        i, j = rng.sample(range(n), 2)
        node = candidate.pop(i)
        old_prev = order[i-1] if i else 0
        candidate.insert(j, node)
        new_prev = candidate[j-1] if j else 0
        return Move(candidate, min(i, j), (("insert", node, new_prev),), (("insert", node, old_prev),))
    if kind == "2opt":
        choices, offset = [], 0
        for route in solution.routes:
            if len(route) >= 2:
                choices.append((offset, route))
            offset += len(route)
        if not choices:
            return None
        offset, route = rng.choice(choices)
        a, b = sorted(rng.sample(range(len(route)), 2))
        left = route[a-1] if a else 0
        right = route[b+1] if b+1 < len(route) else 0
        first, last = route[a], route[b]
        candidate[offset+a:offset+b+1] = reversed(candidate[offset+a:offset+b+1])
        return Move(candidate, offset+a,
                    (edge(left, last), edge(first, right)),
                    (edge(left, first), edge(last, right)))
    raise ValueError("未知邻域")


def candidate(state):
    kind = state.rng.choice(state.config.neighborhoods)
    move = generate(state.order, state.current, kind, state.rng)
    if move is None:
        return None
    solution = state.problem.decode_neighbor(move.order, state.current, move.changed_at)
    if not solution.feasible:
        return None
    return move, solution
