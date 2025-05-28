from lib import data
from lib.random import disable_randomness
from base import simulation as base_simulation
from original import simulation as original_simulation
from jit import simulation as jit_simulation
from tqdm import tqdm
import numpy as np

W, D = data.tvb76_weights_lengths()
W_list = W.tolist()
D_list = D.tolist()

dt = 0.05
tf = 150.0
k = 1e-3
speed = 1.0
freq = 1.0

tolerance = 1e-6

disable_randomness()

for i, speed in tqdm(enumerate([1.0, 2.0, 10.0])):
    T_base, Xs_base = base_simulation.simulate(W_list, D_list, dt, tf, k, speed, freq)
    T_original, Xs_original = original_simulation.simulate(W, D, dt, tf, k, speed, freq)
    T_jit, Xs_jit = jit_simulation.simulate(W_list, D_list, dt, tf, k, speed, freq)

    # Compare results
    assert (
        len(T_base) == len(T_original) == len(T_jit)
    ), f"Length mismatch in T for speed {speed}"
    assert (
        len(Xs_base) == len(Xs_original) == len(Xs_jit)
    ), f"Length mismatch in Xs for speed {speed}"

    for x_base, x_original, x_jit in zip(Xs_base, Xs_original, Xs_jit):
        assert (
            len(x_base) == len(x_original) == len(x_jit)
        ), f"Length mismatch in x for speed {speed}"

        for xi_base, xi_original, xi_jit in zip(x_base, x_original, x_jit):
            assert np.allclose(
                xi_base, xi_original, atol=tolerance
            ), f"Mismatch in Xs for speed {speed} at {xi_base} vs {xi_original} (base vs original)"
            assert np.allclose(
                xi_base, xi_jit, atol=tolerance
            ), f"Mismatch in Xs for speed {speed} at {xi_base} vs {xi_jit} (base vs jit)"


print("All tests passed successfully!")
