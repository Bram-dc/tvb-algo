import math
from typing import Callable
from matplotlib.pylab import rand
import matplotlib.pyplot as plt
from tvb_algo import data, deint, network, helpers
from time import time
from tqdm import tqdm
import numpy as np

W, D = data.tvb76_weights_lengths()

from typing import Callable, List


def sim(
    dt: float = 0.05,
    tf: float = 150.0,
    k: float = 0.0,
    speed: float = 1.0,
    freq: float = 1.0,
):
    n = len(W[0])
    pre: Callable[[float, float], float] = lambda i, j: j - 1.0
    post: Callable[[float], float] = lambda gx: k * gx
    prop = network.wm_ring(W, helpers.divide_2d(D, speed), dt, pre, post, 1)

    def f(i: int, X: list[list[float]]):  # monostable
        result: list[list[float]] = []

        for i, (x, y) in enumerate(X):
            # TODO: Outside of the loop, we can compute c once
            c = prop(i, [[xi] for xi in X[i]])[0][0]

            dx = freq * (x - x**3 / 3 + y) * 3.0
            dy = freq * (1.01 - x + c) / 3.0

            result.append([dx, dy])

        return result

    def g(i: int, X: list[list[float]]):  # additive linear noise
        return math.sqrt(1e-9)

    X = helpers.zeros_2d(n, 2)
    Xs = np.zeros((int(tf / dt),) + X.shape)
    T = np.r_[: Xs.shape[0]]
    for t, (x, _) in zip(T, deint.em_color(f, g, dt, 1e-1, X)):
        if t == 0:
            x[:] = -1.0
        if t == 1:
            x[:] = rand(n, 2) / 5 + np.r_[1.0, -0.6]
        Xs[t] = x
    return T, Xs


dt = 0.05
plt.figure(figsize=(12, 6))


elapsed = 0.0
speeds = [1.0, 2.0, 10.0]
for i, speed in enumerate(tqdm(speeds)):
    tic = time()
    t, x = sim(dt, 150.0, 1e-3, speed)
    elapsed += time() - tic
    plt.subplot(2, 3, i + 1)
    plt.plot(t[::5] * dt, x[::5, :, 0] + 0 * np.r_[: W.shape[0]], "k", alpha=0.3)
    plt.grid(True, axis="x")
    plt.xlim([0, t[-1] * dt])
    plt.title("Speed = %g mm/ms" % (speed,))
    plt.xlabel("time (ms)")
    plt.ylabel("X(t)")
    plt.subplot(2, 3, i + 4)
    plt.hist((D[W != 0] / speed).flat[:], 100, color="k")
    plt.xlim([0, t[-1] * dt])
    plt.grid(True)
    plt.xlabel("delay (ms)")
    plt.ylabel("# delay")

plt.tight_layout()

print("%.3fs elapsed" % (elapsed,))
plt.show()
