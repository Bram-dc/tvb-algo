from typing import Callable


def wm_ring(
    W: list[list[float]],
    D: list[list[float]],
    dt: float,
    pre: Callable[[list[float], list[float]], list[float]],
    post: Callable[[float], float],
    ncv: int = 1,
    cut: float = 0.0,
):
    n = len(W)
    adj: list[list[tuple[int, float, int]]] = []
    max_delay = 0
    for r in range(n):
        row: list[tuple[int, float, int]] = []
        for c in range(n):
            w = W[r][c]
            if w > cut:
                delay = int(D[r][c] / dt)
                row.append((c, w, delay))
                if delay > max_delay:
                    max_delay = delay
        adj.append(row)

    H = max_delay + 1
    hist: list[list[list[float]]] = [[[0.0] * ncv for _ in range(n)] for _ in range(H)]

    active_nodes = [r for r, outs in enumerate(adj) if outs]

    def step(i: int, xi: list[list[float]]) -> list[list[float]]:
        hist[i % H] = xi
        gx = [0.0] * n
        for r in active_nodes:
            total = 0.0
            for c, w_rc, delay in adj[r]:
                xj = hist[(i - delay) % H][c]
                xi_r = xi[r]
                pre_val = pre(xi_r, xj)
                total += w_rc * pre_val[0]
            gx[r] = post(total)

        out = [[0.0] * ncv for _ in range(n)]
        for r in active_nodes:
            out[r][0] = gx[r]
        return out

    return step
