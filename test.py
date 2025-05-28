from data import data
from lib import simulation
from original import simulation as original_simulation
from tqdm import tqdm

W, D = data.tvb76_weights_lengths()
W_list = W.tolist()
D_list = D.tolist()

dt = 0.05
tf = 150.0
k = 1e-3
speed = 1.0
freq = 1.0

for i, speed in tqdm(enumerate([1.0, 2.0, 10.0])):
    T, Xs = simulation.simulate(W_list, D_list, dt, tf, k, speed, freq)
    T_original, Xs_original = original_simulation.simulate(W, D, dt, tf, k, speed, freq)

    # Compare results
    assert len(T) == len(T_original), f"Length mismatch for speed {speed}"
    assert len(Xs) == len(Xs_original), f"Length mismatch for speed {speed}"
    for t in range(len(T)):
        assert len(Xs[t]) == len(
            Xs_original[t]
        ), f"Node count mismatch at time {t} for speed {speed}"
        for n in range(len(Xs[t])):
            assert (
                abs(Xs[t][n][0] - Xs_original[t][n][0]) < 1e-6
            ), f"X value mismatch at time {t}, node {n} for speed {speed}"
            assert (
                abs(Xs[t][n][1] - Xs_original[t][n][1]) < 1e-6
            ), f"Y value mismatch at time {t}, node {n} for speed {speed}"

print("All tests passed successfully!")
