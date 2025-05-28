import numpy as np


def plot_traj(ax, T: list[float], Xs: list[list[list[float]]], speed: float):
    T_np = np.array(T)
    Xs_np = np.array(Xs)

    ax.plot(T_np[::5], Xs_np[::5, :, 0], "k", alpha=0.3)
    ax.grid(True, axis="x")
    ax.set_xlim(0, T_np[-1])
    ax.set_title(f"Speed = {speed} mm/ms")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("X(t)")


def plot_delay(
    ax, D: list[list[float]], W: list[list[float]], T: list[float], speed: float
):
    D_np = np.array(D)
    W_np = np.array(W)
    T_np = np.array(T)

    delays = (D_np[W_np != 0] / speed).flatten()

    ax.hist(delays, bins=100)
    ax.grid(True)
    ax.set_xlim(0, T_np[-1])
    ax.set_xlabel("delay (ms)")
    ax.set_ylabel("# delay")
