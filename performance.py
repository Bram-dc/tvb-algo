import matplotlib.pyplot as plt
from lib import data
from base import simulation as base_simulation
from original import simulation as original_simulation
from base_single_ncv import simulation as base_single_ncv_simulation
from parallel import simulation as parallel_simulation
from jit import simulation as jit_simulation
from jit_parallel import simulation as jit_parallel_simulation
from tqdm import tqdm

# W, D = data.tvb76_weights_lengths()
W, D = data.tvb998_weights_lengths()
W_list = W.tolist()
D_list = D.tolist()

speed = 1.0
tf = 150.0
k = 1e-3
freq = 1.0

dt_values = [0.01, 0.02, 0.05, 0.1, 0.2]
timings_base: list[float] = []
timings_original: list[float] = []
timings_base_single_ncv: list[float] = []
timings_parallel: list[float] = []
timings_jit: list[float] = []
timings_jit_parallel: list[float] = []

for dt in tqdm(dt_values):
    T_base, Xs_base, duration_base = base_simulation.simulate(
        W_list, D_list, dt, tf, k, speed, freq
    )
    timings_base.append(duration_base)

    T_original, Xs_original, duration_original = original_simulation.simulate(
        W, D, dt, tf, k, speed, freq
    )
    timings_original.append(duration_original)

    T_base_single_ncv, Xs_base_single_ncv, duration_base_single_ncv = (
        base_single_ncv_simulation.simulate(W_list, D_list, dt, tf, k, speed, freq)
    )
    timings_base_single_ncv.append(duration_base_single_ncv)

    T_parallel, Xs_parallel, duration_parallel = parallel_simulation.simulate(
        W_list, D_list, dt, tf, k, speed, freq
    )
    timings_parallel.append(duration_parallel)

    T_jit, Xs_jit, duration_jit = jit_simulation.simulate(
        W_list, D_list, dt, tf, k, speed, freq
    )
    timings_jit.append(duration_jit)

    T_jit_parallel, Xs_jit_parallel, duration_jit_parallel = (
        jit_parallel_simulation.simulate(W_list, D_list, dt, tf, k, speed, freq)
    )
    timings_jit_parallel.append(duration_jit_parallel)

plt.figure(figsize=(10, 6))  # type: ignore
plt.plot(dt_values, timings_base, label="Base Simulation", marker="o")  # type: ignore
plt.plot(dt_values, timings_original, label="Original Simulation", marker="o")  # type: ignore
plt.plot(dt_values, timings_base_single_ncv, label="Base NCV=1 Simulation", marker="o")  # type: ignore
plt.plot(dt_values, timings_parallel, label="Parallel Simulation", marker="o")  # type: ignore
plt.plot(dt_values, timings_jit, label="JIT Simulation", marker="o")  # type: ignore
plt.plot(dt_values, timings_jit_parallel, label="JIT Parallel Simulation", marker="o")  # type: ignore
plt.xlabel("Time Step (dt)")  # type: ignore
plt.ylabel("Time (seconds)")  # type: ignore
plt.title("Simulation Timing Comparison")  # type: ignore
plt.legend()  # type: ignore
plt.grid()  # type: ignore
plt.tight_layout()  # type: ignore
plt.show()  # type: ignore
