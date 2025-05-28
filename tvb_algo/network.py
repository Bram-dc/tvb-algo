"""Network implementation including weights & delays."""

from typing import Callable
import numpy as np
from .helpers import zeros_3d


def wm_ring(
    W: list[list[float]],
    D: list[list[float]],
    dt: float,
    pre: Callable[[float, float], float],
    post: Callable[[float], float],
    ncv: float,
):
    """Build white matter connectome model with sparse weights, ring buffer."""
    n = len(W[0])
    di = (D / dt).astype("i")  # delays in time steps
    r, c = np.argwhere(m).T  # non-zero row & col indices
    (lri,) = np.argwhere(np.diff(np.r_[-1, r])).T  # local reduction indices
    nzr = np.unique(r)  # rows with non-zeros
    H = di.max() + 1
    hist = zeros_3d(H, n, ncv)

    def step(i: int, xi):
        hist[i % H] = xi
        xj = hist[(i - di) % H, c]
        gx = np.add.reduceat((W * pre(xi[c], xj).T).T, lri)
        out = np.zeros_like(xi)
        out[nzr] = post(gx)
        return out

    return step
