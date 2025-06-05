"""Differential equation integration."""

import numpy as np


def em_color(f, g, dt, lam, x):
    """Euler-Maruyama for colored noise."""
    i = 0
    nd = x.shape
    # r = np.random.randn(*nd)
    r = 0
    e = np.sqrt(g(i, x) * lam) * r
    E = np.exp(-lam * dt)
    while True:
        yield x, e
        i += 1
        x += dt * (f(i, x) + e)
        # r = np.random.randn(*nd)
        r = 0
        h = np.sqrt(g(i, x) * lam * (1 - E**2)) * r
        e = e * E + h
