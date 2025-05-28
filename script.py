import math
from typing import Callable, Generator, Tuple
from matplotlib.pylab import rand
from tvb_algo import data
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

print("downloading weights")
W, D = data.tvb76_weights_lengths()

print(W.shape, D.shape)
# (76, 76) (76, 76)


def em_color(
    f: Callable[[int, NDArray[np.float64]], np.ndarray],
    g: Callable[[int, np.ndarray], float],
    dt: float,
    lam: float,
    x: np.ndarray,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Euler-Maruyama for colored noise."""
    i: int = 0
    nd = x.shape
    e: np.ndarray = np.sqrt(g(i, x) * lam) * np.random.randn(*nd)
    E: float = np.exp(-lam * dt)
    while True:
        yield x, e
        i += 1
        x += dt * (f(i, x) + e)
        h: np.ndarray = np.sqrt(g(i, x) * lam * (1 - E**2)) * np.random.randn(*nd)
        e = e * E + h


def wm_ring(
    W: np.ndarray,
    D: np.ndarray,
    dt: float,
    pre: Callable[[np.ndarray, np.ndarray], np.ndarray],
    post: Callable[[np.ndarray], np.ndarray],
    ncv: int,
    cut: float = 0.0,
    icf: Callable[[np.ndarray], np.ndarray] = lambda h: h,
) -> Callable[[int, np.ndarray], np.ndarray]:
    """Build white matter connectome model with sparse weights, ring buffer."""
    n: int = W.shape[0]
    mask: np.ndarray = W > cut
    w: np.ndarray = W[mask]
    d: np.ndarray = D[mask]
    di: np.ndarray = (d / dt).astype(int)
    r, c = np.nonzero(mask)
    lri: np.ndarray = np.unique(r, return_index=True)[1]
    nzr: np.ndarray = np.unique(r)
    H: int = di.max() + 1
    hist: np.ndarray = icf(np.zeros((H, n, ncv), dtype=float))

    def step(i: int, xi: np.ndarray) -> np.ndarray:
        hist[i % H] = xi
        xj: np.ndarray = hist[(i - di) % H, c]
        gx: np.ndarray = np.add.reduceat((w * pre(xi[c], xj).T).T, lri)
        out: np.ndarray = np.zeros_like(xi)
        out[nzr] = post(gx)
        return out

    return step


def sim(
    dt: float = 0.05,
    tf: float = 150.0,
    k: float = 0.0,
    speed: float = 1.0,
    freq: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the neural mass simulation with given parameters."""
    n: int = W.shape[0]
    # pre- and post-synaptic coupling functions
    pre: Callable[[np.ndarray, np.ndarray], np.ndarray] = lambda i, j: j - 1.0
    post: Callable[[np.ndarray], np.ndarray] = lambda gx: k * gx
    prop: Callable[[int, np.ndarray], np.ndarray] = wm_ring(
        W, D / speed, dt, pre, post, ncv=1
    )

    def f(i: int, X: np.ndarray) -> np.ndarray:
        X_list = X.tolist()
        x_list = [pair[0] for pair in X_list]
        y_list = [pair[1] for pair in X_list]

        x_input = [[x] for x in x_list]
        c_arr = prop(i, np.array(x_input))
        c_list = c_arr.flatten().tolist()

        dx_list = []
        dy_list = []
        for x, y, c in zip(x_list, y_list, c_list):
            dx = freq * (x - x**3 / 3 + y) * 3.0
            dy = freq * (1.01 - x + c) / 3.0
            dx_list.append(dx)
            dy_list.append(dy)

        out = np.empty((len(dx_list), 2), dtype=float)
        for idx, (dx, dy) in enumerate(zip(dx_list, dy_list)):
            out[idx, 0] = dx
            out[idx, 1] = dy
        return out

    def g(i: int, X: NDArray[np.float64]) -> float:
        return math.sqrt(1e-9)

    X_init: NDArray[np.float64] = np.zeros((n, 2))
    Xs: NDArray[np.float64] = np.zeros((int(tf / dt), n, 2))
    T: NDArray[np.float64] = np.arange(Xs.shape[0])

    for t, (x, _) in zip(T, em_color(f, g, dt, lam=1e-1, x=X_init)):
        if t == 0:
            x[:] = -1.0
        elif t == 1:
            x[:] = rand(n, 2) / 5 + np.array([1.0, -0.6])
        Xs[t] = x
    return T, Xs


dt = 0.05
plt.figure(figsize=(12, 6))
from time import time
from tqdm import tqdm

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
