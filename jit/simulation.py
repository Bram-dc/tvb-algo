from . import network, deint
import numba


@numba.jit(nopython=True)
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

    H, hist, active_nodes, adj = network.wm_ring_params(W, D_speed, dt, ncv=1, cut=0.0)

    steps = int(tf / dt)
    X = [[0.0, 0.0] for _ in range(n)]
    Xs: list[list[list[float]]] = []
    gen = deint.em_color(freq, k, H, hist, active_nodes, adj, dt, x0=X)

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
