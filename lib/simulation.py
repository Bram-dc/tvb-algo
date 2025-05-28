import math
import random
from . import network, deint


def simulate(
    W: list[list[float]],
    D: list[list[float]],
    dt: float,
    tf: float,
    k: float,
    speed: float,
    freq: float,
) -> tuple[list[float], list[list[list[float]]]]:
    n = len(W)

    def pre(xi: list[float], xj: list[float]) -> list[float]:
        return [xj[0] - 1.0]

    def post(gx: float) -> float:
        return k * gx

    D_speed = [[D[r][c] / speed for c in range(n)] for r in range(n)]

    prop = network.wm_ring(W, D_speed, dt, pre, post, ncv=1, cut=0.0)

    def f(i: int, X: list[list[float]]) -> list[list[float]]:
        x_vals = [X[r][0] for r in range(n)]
        y_vals = [X[r][1] for r in range(n)]
        inp = [[x] for x in x_vals]
        c_out = prop(i, inp)
        c_list = [c_out[r][0] for r in range(n)]

        dx = [0.0] * n
        dy = [0.0] * n
        for r in range(n):
            x, y, c = x_vals[r], y_vals[r], c_list[r]
            dx[r] = freq * (x - x**3 / 3 + y) * 3.0
            dy[r] = freq * (1.01 - x + c) / 3.0

        return [[dx[r], dy[r]] for r in range(n)]

    def g(i: int, X: list[list[float]]) -> float:
        return math.sqrt(1e-9)

    steps = int(tf / dt)
    X = [[0.0, 0.0] for _ in range(n)]
    Xs: list[list[list[float]]] = []
    gen = deint.em_color(f, g, dt, lam=1e-1, x0=X)

    for t in range(steps):
        x, _ = next(gen)
        if t == 0:
            for r in range(n):
                x[r][0] = x[r][1] = -1.0
        elif t == 1:
            for r in range(n):
                x[r][0] = random.random() / 5 + 1.0
                x[r][1] = random.random() / 5 - 0.6
        Xs.append([x[r].copy() for r in range(n)])

    T = [t * dt for t in range(steps)]
    return T, Xs
