"""Network implementation including weights & delays."""

from typing import Callable, List, Tuple
from tvb_algo import helpers


def extract_connections(
    W: List[List[float]], D: List[List[float]], cut: float
) -> Tuple[List[float], List[float], List[int], List[int]]:
    """Return flat lists of weights, delays, row-indices and col-indices for W[i][j] > cut."""
    w: List[float] = []
    d: List[float] = []
    rows: List[int] = []
    cols: List[int] = []

    for i, row in enumerate(W):
        for j, val in enumerate(row):
            if val > cut:
                w.append(val)
                d.append(D[i][j])
                rows.append(i)
                cols.append(j)

    return w, d, rows, cols


def wm_ring(
    W: List[List[float]],  # 2D weight matrix
    D: List[List[float]],  # 2D delay matrix
    dt: float,
    pre: Callable[[float, float], float],
    post: Callable[[float], float],
    ncv: int,
    cut: float = 0.0,
) -> Callable[[int, List[List[float]]], List[List[float]]]:
    """Build white‐matter connectome model with sparse weights, ring buffer."""
    n = len(W[0])
    w, d, rows, cols = extract_connections(W, D, cut)
    di = [int(v / dt) for v in d]
    H = (max(di) + 1) if di else 1
    hist = helpers.zeros_3d(H, n, ncv)

    def step(step_idx: int, xi: List[List[float]]) -> List[List[float]]:
        buf_idx = step_idx % H
        hist[buf_idx] = [row.copy() for row in xi]

        acc = helpers.zeros_2d(n, ncv)
        for k in range(len(w)):
            src = cols[k]
            tgt = rows[k]
            delay_idx = di[k]
            past = hist[(step_idx - delay_idx) % H][src]
            curr = xi[src]
            wt = w[k]
            for v in range(ncv):
                acc[tgt][v] += wt * pre(curr[v], past[v])

        out = helpers.zeros_2d(n, ncv)
        for r in set(rows):
            for v in range(ncv):
                out[r][v] = post(acc[r][v])

        return out

    return step
