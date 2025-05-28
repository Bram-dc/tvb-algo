from lib import data
from lib.random import disable_randomness
from base import simulation as base_simulation
from original import simulation as original_simulation
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

disable_randomness()

for i, speed in tqdm(enumerate([1.0, 2.0, 10.0])):
    T, Xs = base_simulation.simulate(W_list, D_list, dt, tf, k, speed, freq)
    T_original, Xs_original = original_simulation.simulate(W, D, dt, tf, k, speed, freq)

    # Compare results
    assert len(T) == len(T_original), f"Length mismatch for speed {speed}"
    assert len(Xs) == len(Xs_original), f"Length mismatch for speed {speed}"

    for x, x_original in tqdm(zip(Xs, Xs_original)):
        assert len(x) == len(x_original), f"Length mismatch in Xs for speed {speed}"
        for xi, xi_original in zip(x, x_original):
            assert np.allclose(
                xi, xi_original, atol=1e-6
            ), f"Values mismatch for speed {speed}"

print("All tests passed successfully!")
