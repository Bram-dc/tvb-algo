from typing import Generator
from jit import network


def f(
    n: int,
    i: int,
    freq: float,
    k: float,
    X: list[list[float]],
    H: int,
    hist: list[list[list[float]]],
    active_nodes: list[int],
    adj: list[list[tuple[int, float, int]]],
) -> tuple[list[float], list[float]]:
    x_vals = [X[r][0] for r in range(n)]
    y_vals = [X[r][1] for r in range(n)]
    inp = [[x] for x in x_vals]
    c_out = network.step(n, i, inp, H, hist, active_nodes, adj, k, ncv=1)
    c_list = [c_out[r][0] for r in range(n)]

    dx = [0.0] * n
    dy = [0.0] * n
    for r in range(n):
        x, y, c = x_vals[r], y_vals[r], c_list[r]
        dx[r] = freq * (x - x**3 / 3 + y) * 3.0
        dy[r] = freq * (1.01 - x + c) / 3.0

    return dx, dy


def em_color(
    freq: float,
    k: float,
    H: int,
    hist: list[list[list[float]]],
    active_nodes: list[int],
    adj: list[list[tuple[int, float, int]]],
    dt: float,
    x0: list[list[float]],
) -> Generator[list[list[float]], None, None]:
    n = len(x0)

    i = 0
    while True:
        yield x0
        i += 1
        f_val_x, f_val_y = f(n, i, freq, k, x0, H, hist, active_nodes, adj)

        for r in range(n):
            x0[r][0] += dt * f_val_x[r]
            x0[r][1] += dt * f_val_y[r]
