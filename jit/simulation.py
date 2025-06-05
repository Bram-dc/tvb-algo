from typing import Generator


# Pre-synaptic function: computes input from neuron j to neuron i
def pre(xi: list[float], xj: list[float]) -> list[float]:
    return [xj[0] - 1.0]


# Post-synaptic function: scales input by coupling constant k
def post(gx: float, k: float) -> float:
    return k * gx


# Prepare adjacency and delay structures for the network
def wm_ring_params(
    W: list[list[float]],
    D: list[list[float]],
    dt: float,
    ncv: int,
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

    H = max_delay + 1  # History length needed for delays
    hist = [[[0.0] * ncv for _ in range(n)] for _ in range(H)]
    active_nodes = [r for r, outs in enumerate(adj) if outs]

    return H, hist, active_nodes, adj


# Single integration step for the network
def step(
    n: int,
    i: int,
    xi: list[list[float]],
    H: int,
    hist: list[list[list[float]]],
    active_nodes: list[int],
    adj: list[list[tuple[int, float, int]]],
    k: float,
    ncv: int,
) -> list[list[float]]:

    hist[i % H] = xi  # Store current state in history buffer

    out = [[0.0] * ncv for _ in range(n)]

    for r in active_nodes:
        total = 0.0

        for c, w_rc, delay in adj[r]:
            xj = hist[(i - delay) % H][c]
            pre_val = pre(xi[r], xj)
            total += w_rc * pre_val[0]

        out[r][0] = post(total, k)

    return out


def compute_derivative(
    x: float, y: float, c: float, freq: float
) -> tuple[float, float]:
    dx = freq * (x - x**3 / 3 + y) * 3.0
    dy = freq * (1.01 - x + c) / 3.0

    return dx, dy


# Computes derivatives for the network
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
    ncv: int,
) -> tuple[list[float], list[float]]:

    x_vals = [X[r][0] for r in range(n)]

    inp = [[x] * ncv for x in x_vals]
    c_out = step(n, i, inp, H, hist, active_nodes, adj, k, ncv)
    c_list = [c_out[r][0] for r in range(n)]

    dx = [0.0] * n
    dy = [0.0] * n
    for r in range(n):
        x = X[r][0]
        y = X[r][1]
        c = c_list[r]

        dx[r], dy[r] = compute_derivative(x, y, c, freq)

    return dx, dy


# Generator for Euler-Maruyama integration of the network
def em_color(
    freq: float,
    k: float,
    H: int,
    hist: list[list[list[float]]],
    active_nodes: list[int],
    adj: list[list[tuple[int, float, int]]],
    dt: float,
    x0: list[list[float]],
    ncv: int,
) -> Generator[list[list[float]], None, None]:
    n = len(x0)

    i = 0
    while True:
        yield x0
        i += 1
        dx, dy = f(n, i, freq, k, x0, H, hist, active_nodes, adj, ncv)

        for r in range(n):
            x0[r][0] += dt * dx[r]
            x0[r][1] += dt * dy[r]


# Main simulation function
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

    ncv = 1  # Number of control variables

    gen = em_color(freq, k, H, hist, active_nodes, adj, dt, x0, ncv)

    Xs: list[list[list[float]]] = []

    for t in range(steps):
        x = next(gen)

        # Set initial conditions for first two steps
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
