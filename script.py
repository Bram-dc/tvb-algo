import matplotlib.pyplot as plt
from tvb_algo import simulation, data, plot
import time
from tqdm import tqdm

W, D = data.tvb76_weights_lengths()
n = len(W)

dt = 0.05
elapsed = 0.0

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))  # type: ignore

for i, speed in tqdm(enumerate([1.0, 2.0, 10.0])):
    tic = time.time()
    T, Xs = simulation.simulate(dt, 150.0, k=1e-3, speed=speed)
    elapsed += time.time() - tic

    plot.plot_traj(axes[0, i], T, Xs, speed)
    plot.plot_delay(axes[1, i], D, W, T, speed)

print(f"{elapsed:.3f}s elapsed")

fig.tight_layout()
# plt.show()  # type: ignore
