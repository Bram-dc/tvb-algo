import matplotlib.pyplot as plt
from lib import data, plot
from parallel import simulation
from tqdm import tqdm

W, D = data.tvb76_weights_lengths()
W_list = W.tolist()
D_list = D.tolist()

dt = 0.05
tf = 150.0
k = 1e-3
speed = 1.0
freq = 1.0

elapsed = 0.0

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))  # type: ignore

for i, speed in tqdm(enumerate([1.0, 2.0, 10.0])):

    T, Xs, durations = simulation.simulate(W_list, D_list, dt, tf, k, speed, freq)
    elapsed += durations

    plot.plot_traj(axes[0, i], T, Xs, speed)
    plot.plot_delay(axes[1, i], D, W, T, speed)

print(f"{elapsed:.3f}s elapsed")

fig.tight_layout()
plt.show()  # type: ignore
