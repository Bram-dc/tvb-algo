import math
import random
from typing import Callable, Generator


def em_color(
    f: Callable[[int, list[list[float]]], list[list[float]]],
    g: Callable[[int, list[list[float]]], float],
    dt: float,
    lam: float,
    x0: list[list[float]],
) -> Generator[tuple[list[list[float]], list[list[float]]], None, None]:
    n = len(x0)
    dim = len(x0[0])

    sigma = math.sqrt(g(0, x0) * lam)
    e = [[sigma * random.gauss(0, 1) for _ in range(dim)] for _ in range(n)]
    E = math.exp(-lam * dt)
    i = 0

    while True:
        yield x0, e
        i += 1
        f_val = f(i, x0)
        for r in range(n):
            for d in range(dim):
                x0[r][d] += dt * (f_val[r][d] + e[r][d])

        sigma2 = math.sqrt(g(i, x0) * lam * (1 - E * E))
        h = [[sigma2 * random.gauss(0, 1) for _ in range(dim)] for _ in range(n)]

        for r in range(n):
            for d in range(dim):
                e[r][d] = e[r][d] * E + h[r][d]
