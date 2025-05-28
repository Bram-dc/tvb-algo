import math
import random
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
from tvb_algo import data, network, deint
import time
from tqdm import tqdm

W, D = data.tvb76_weights_lengths()
n = len(W)


def simulate(
    dt: float = 0.05,
    tf: float = 150.0,
    k: float = 0.0,
    speed: float = 1.0,
    freq: float = 1.0,
) -> tuple[list[float], list[list[list[float]]]]:
    n = len(W)

    pre: Callable[[list[float], list[float]], list[float]] = lambda xi, xj: [
        xj[0] - 1.0
    ]
    post: Callable[[float], float] = lambda gx: k * gx

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


dt = 0.05


plt.figure(figsize=(12, 6))
elapsed = 0.0
for i, speed in tqdm(enumerate([1.0, 2.0, 10.0])):
    tic = time.time()
    T, Xs = simulate(dt, 150.0, k=1e-3, speed=speed)
    elapsed += time.time() - tic

    D_np = np.array(D)
    W_np = np.array(W)
    T_np = np.array(T)
    Xs_np = np.array(Xs)

    delays = (D[W != 0] / speed).flatten()

    plt.subplot(2, 3, i + 1)
    plt.plot(T_np[::5], Xs_np[::5, :, 0], "k", alpha=0.3)
    plt.grid(True, axis="x")
    plt.xlim(0, T_np[-1])
    plt.title(f"Speed = {speed} mm/ms")
    plt.xlabel("time (ms)")
    plt.ylabel("X(t)")

    plt.subplot(2, 3, i + 4)
    plt.hist(delays, bins=100)
    plt.grid(True)
    plt.xlabel("delay (ms)")
    plt.ylabel("# delay")
    plt.xlim(0, T_np[-1])

plt.tight_layout()
print(f"{elapsed:.3f}s elapsed")
plt.show()
