import time
import numba  # type: ignore
import numpy as np


# Prepare adjacency and delay structures for the network
def wm_ring_params(
    W: np.ndarray,
    D: np.ndarray,
    dt: float,
    cut: float = 0.0,
) -> tuple[
    np.ndarray,  # bool array for active nodes
    np.ndarray,  # float array for adj_w
    np.ndarray,  # int array for adj_delay
    int,  # max_delay
]:
    n = len(W)

    adj_w = np.zeros((n, n), dtype=float)
    adj_delay = np.zeros((n, n), dtype=int)
    max_delay = 0

    for r in range(n):
        for c in range(n):
            w = W[r, c]
            if w > cut:
                delay = int(D[r, c] / dt)

                adj_w[r, c] = w
                adj_delay[r, c] = delay

                if delay > max_delay:
                    max_delay = delay

    is_node_active = np.any(adj_w > cut, axis=1)

    return is_node_active, adj_w, adj_delay, max_delay


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


@numba.njit
def compute_node(
    n: int,
    i: int,
    freq: float,
    k: float,
    H: int,
    x: float,
    y: float,
    hist: np.ndarray,  # History as a numpy array
    is_active: bool,
    adj_w_r: np.ndarray,  # float array
    adj_delay_r: np.ndarray,  # int array
) -> tuple[float, float]:
    total = 0.0
    if is_active:
        for c in range(n):
            w = adj_w_r[c]
            delay = adj_delay_r[c]

            xj = hist[(i - delay) % H, c]
            total += w * pre(x, xj)

    coupling = post(total, k)

    return compute_derivatives(x, y, coupling, freq)


# Generator for Euler-Maruyama integration of the network
@numba.njit  # type: ignore
def step(
    n: int,
    i: int,
    freq: float,
    k: float,
    dt: float,
    x0: np.ndarray,
    y0: np.ndarray,
    H: int,
    hist: np.ndarray,  # History as a numpy array
    is_node_active: np.ndarray,  # bool array
    adj_w: np.ndarray,  # float array
    adj_delay: np.ndarray,  # int array
) -> tuple[np.ndarray, np.ndarray]:
    dx = np.zeros(n)
    dy = np.zeros(n)

    for r in range(n):
        dx[r], dy[r] = compute_node(
            n,
            i,
            freq,
            k,
            H,
            x0[r],
            y0[r],
            hist,
            is_node_active[r],
            adj_w[r],
            adj_delay[r],
        )

    x0 += dt * dx
    y0 += dt * dy

    return x0, y0


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

    W = np.array(W, dtype=float)
    D = np.array(D, dtype=float)

    D_speed = D / speed  # Element-wise division of numpy arrays

    is_node_active, adj_w, adj_delay, max_delay = wm_ring_params(
        W, D_speed, dt, cut=0.0
    )

    H = max_delay + 1  # History length needed for delays
    hist = np.zeros((H, n))

    steps = int(tf / dt)
    x0 = np.zeros(n)
    y0 = np.zeros(n)

    Xs = np.zeros((steps, n, 2))

    # Compile the jit functions
    pre(1.0, 1.0)
    post(1.0, 1.0)
    compute_derivatives(1.0, 1.0, 1.0, 1.0)
    compute_node(
        n,
        0,
        freq,
        k,
        H,
        x0[0],
        y0[0],
        hist,
        is_node_active[0],
        adj_w[0],
        adj_delay[0],
    )
    step(
        n,
        0,
        freq,
        k,
        dt,
        x0,
        y0,
        H,
        hist,
        is_node_active,
        adj_w,
        adj_delay,
    )

    start = time.time()
    for i in range(steps):
        x, y = step(
            n,
            i,
            freq,
            k,
            dt,
            x0,
            y0,
            H,
            hist,
            is_node_active,
            adj_w,
            adj_delay,
        )

        # Set initial conditions for first two steps
        if i == 0:
            x[:] = -1.0
            y[:] = -1.0
        elif i == 1:
            r1, r2 = 0.5, 0.5
            x[:] = r1 / 5 + 1.0
            y[:] = r2 / 5 - 0.6

        # +1 weird implementation detail
        hist[(i + 1) % H] = x0.copy()

        Xs[i] = np.column_stack((x, y))
    end = time.time()

    T = np.arange(steps) * dt

    duration = end - start

    return T.tolist(), Xs.tolist(), duration
