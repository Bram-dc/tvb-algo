import time
from typing import Any

# from matplotlib.pylab import rand
from . import deint, network
import numpy as np
import math


def simulate(
    W: Any,
    D: Any,
    dt: float,
    tf: float,
    k: float,
    speed: float,
    freq: float,
):
    n = W.shape[0]
    pre = lambda i, j: j - 1.0
    post = lambda gx: k * gx
    prop = network.wm_ring(W, D / speed, dt, pre, post, 1)

    def f(i, X):  # monostable
        x, y = X.T
        (c,) = prop(i, x.reshape((-1, 1))).T
        dx = freq * (x - x**3 / 3 + y) * 3.0
        dy = freq * (1.01 - x + c) / 3.0
        return np.array([dx, dy]).T

    def g(i, X):  # additive linear noise
        return math.sqrt(1e-9)

    X = np.zeros((n, 2))
    Xs = np.zeros((int(tf / dt),) + X.shape)
    T = np.r_[: Xs.shape[0]]

    start = time.time()
    for t, (x, _) in zip(T, deint.em_color(f, g, dt, 1e-1, X)):
        if t == 0:
            x[:] = -1.0
        if t == 1:
            # r = rand(n, 2)
            r = np.array([[0.5, 0.5]] * n)
            x[:] = r / 5 + np.r_[1.0, -0.6]
        Xs[t] = x
    end = time.time()

    duration = end - start

    return T, Xs, duration
