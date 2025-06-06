from concurrent.futures import ThreadPoolExecutor
import time
from typing import Generator
import numba  # type: ignore

max_workers = 8
threading_enabled = True


# Prepare adjacency and delay structures for the network
def wm_ring_params(
    W: list[list[float]],
    D: list[list[float]],
    dt: float,
    cut: float = 0.0,
) -> tuple[int, list[list[float]], list[bool], list[list[tuple[int, float, int]]]]:
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
    hist = [[0.0 for _ in range(n)] for _ in range(H)]

    active_nodes = [r for r, outs in enumerate(adj) if outs]

    is_node_active = [True if r in active_nodes else False for r in range(n)]

    return H, hist, is_node_active, adj


# Pre-synaptic function: computes input from neuron j to neuron i
@numba.njit  # type: ignore
def pre(xi: float, xj: float) -> float:
    return xj - 1.0


# Post-synaptic function: scales input by coupling constant k
@numba.njit  # type: ignore
def post(gx: float, k: float) -> float:
    return k * gx


@numba.njit  # type: ignore
def compute_derivatives(
    x: float, y: float, coupling: float, freq: float
) -> tuple[float, float]:
    dx = freq * (x - x**3 / 3 + y) * 3.0
    dy = freq * (1.01 - x + coupling) / 3.0

    return dx, dy


# Computes derivatives for the network
def f(
    n: int,
    i: int,
    freq: float,
    k: float,
    X: list[list[float]],
    H: int,
    hist: list[list[float]],
    is_node_active: list[bool],
    adj: list[list[tuple[int, float, int]]],
) -> tuple[list[float], list[float]]:

    hist[i % H] = [X[r][0] for r in range(n)]  # Store current state in history buffer

    dx = [0.0] * n
    dy = [0.0] * n

    def compute_node(r: int):
        x = X[r][0]
        y = X[r][1]

        total = 0.0
        if is_node_active[r]:
            for c, w_rc, delay in adj[r]:
                xj = hist[(i - delay) % H][c]
                total += w_rc * pre(X[r][0], xj)

        coupling = post(total, k)

        return compute_derivatives(x, y, coupling, freq)

    if threading_enabled:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for r, future in enumerate(
                executor.submit(compute_node, r) for r in range(n)
            ):
                dx[r], dy[r] = future.result()
    else:
        for r in range(n):
            dx[r], dy[r] = compute_node(r)

    return dx, dy


# Generator for Euler-Maruyama integration of the network
def em_color(
    freq: float,
    k: float,
    H: int,
    hist: list[list[float]],
    is_node_active: list[bool],
    adj: list[list[tuple[int, float, int]]],
    dt: float,
    x0: list[list[float]],
) -> Generator[list[list[float]], None, None]:
    n = len(x0)

    i = 0
    while True:
        yield x0
        i += 1
        dx, dy = f(n, i, freq, k, x0, H, hist, is_node_active, adj)

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
) -> tuple[list[float], list[list[list[float]]], float]:
    n = len(W)

    D_speed = [[D[r][c] / speed for c in range(n)] for r in range(n)]

    H, hist, is_node_active, adj = wm_ring_params(W, D_speed, dt, cut=0.0)

    steps = int(tf / dt)
    x0 = [[0.0, 0.0] for _ in range(n)]

    gen = em_color(freq, k, H, hist, is_node_active, adj, dt, x0)

    Xs: list[list[list[float]]] = []

    # Compile the jit functions
    pre(1.0, 1.0)
    post(1.0, 1.0)
    compute_derivatives(1.0, 1.0, 1.0, 1.0)

    start = time.time()
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
    end = time.time()

    T = [t * dt for t in range(steps)]

    duration = end - start

    return T, Xs, duration
