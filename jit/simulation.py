from typing import Generator


def pre(xi: list[float], xj: list[float]) -> list[float]:
    return [xj[0] - 1.0]


def post(gx: float, k: float) -> float:
    return k * gx


def wm_ring_params(
    W: list[list[float]],
    D: list[list[float]],
    dt: float,
    ncv: int = 1,
    cut: float = 0.0,
) -> tuple[int, list[list[list[float]]], list[int], list[list[tuple[int, float, int]]]]:
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
    hist = [[[0.0] * ncv for _ in range(n)] for _ in range(H)]
    active_nodes = [r for r, outs in enumerate(adj) if outs]

    return H, hist, active_nodes, adj


def step(
    n: int,
    i: int,
    xi: list[list[float]],
    H: int,
    hist: list[list[list[float]]],
    active_nodes: list[int],
    adj: list[list[tuple[int, float, int]]],
    k: float,
    ncv: int = 1,
) -> list[list[float]]:
    hist[i % H] = xi
    gx = [0.0] * n
    for r in active_nodes:
        total = 0.0
        for c, w_rc, delay in adj[r]:
            xj = hist[(i - delay) % H][c]
            xi_r = xi[r]
            pre_val = pre(xi_r, xj)
            total += w_rc * pre_val[0]
        gx[r] = post(total, k)

    out = [[0.0] * ncv for _ in range(n)]
    for r in active_nodes:
        out[r][0] = gx[r]

    return out


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
    c_out = step(n, i, inp, H, hist, active_nodes, adj, k, ncv=1)
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

    D_speed = [[D[r][c] / speed for c in range(n)] for r in range(n)]

    H, hist, active_nodes, adj = wm_ring_params(W, D_speed, dt, ncv=1, cut=0.0)

    steps = int(tf / dt)
    x0 = [[0.0, 0.0] for _ in range(n)]
    gen = em_color(freq, k, H, hist, active_nodes, adj, dt, x0)

    Xs: list[list[list[float]]] = []

    for t in range(steps):
        x = next(gen)
        if t == 0:
            for r in range(n):
                x[r][0] = x[r][1] = -1.0
        elif t == 1:
            for r in range(n):
                r1 = 0.5
                r2 = 0.5

                x[r][0] = r1 / 5 + 1.0
                x[r][1] = r2 / 5 - 0.6

        Xs.append([x[r].copy() for r in range(n)])

    T = [t * dt for t in range(steps)]

    return T, Xs
