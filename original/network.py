"""Network implementation including weights & delays."""

import numpy as np


def wm_ring(W, D, dt, pre, post, ncv, cut=0, icf=lambda h: h):
    """Build white matter connectome model with sparse weights, ring buffer."""
    n = W.shape[0]
    m = W > cut  # non-zero mask
    w = W[m]  # non-zero weights
    d = D[m]  # non-zero delays
    di = (d / dt).astype("i")  # non-zero delays in time steps
    r, c = np.argwhere(m).T  # non-zero row & col indices
    (lri,) = np.argwhere(np.diff(np.r_[-1, r])).T  # local reduction indices
    nzr = np.unique(r)  # rows with non-zeros
    H = di.max() + 1
    hist = icf(np.zeros((H, n, ncv), "f"))

    def step(i, xi):
        hist[i % H] = xi
        xj = hist[(i - di) % H, c]
        gx = np.add.reduceat((w * pre(xi[c], xj).T).T, lri)
        out = np.zeros_like(xi)
        out[nzr] = post(gx)
        return out

    return step
