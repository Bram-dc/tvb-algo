import matplotlib.pyplot as plt

timings_base = [18.2333, 27.9790, 552.1261]
timings_original = [0.6949, 1.0737, 12.5575]
timings_base_single_ncv = [8.1738, 15.7172, 290.7044]
timings_parallel = [30.8425, 70.5493, 576.0387]
timings_jit = [0.3373, 1.5268, 54.5474]
timings_jit_parallel = [8.2826, 8.9790, 16.5065]
roi_values = [76, 192, 998]

plt.figure(figsize=(10, 6))  # type: ignore
plt.plot(roi_values, timings_base, label="Base Simulation", marker="o")  # type: ignore
plt.plot(roi_values, timings_original, label="Original Simulation", marker="o")  # type: ignore
plt.plot(roi_values, timings_base_single_ncv, label="Base NCV=1 Simulation", marker="o")  # type: ignore
plt.plot(roi_values, timings_parallel, label="Parallel Simulation", marker="o")  # type: ignore
plt.plot(roi_values, timings_jit, label="JIT Simulation", marker="o")  # type: ignore
plt.plot(roi_values, timings_jit_parallel, label="JIT Parallel Simulation", marker="o")  # type: ignore
plt.xlabel("# of ROIs")  # type: ignore
plt.ylabel("Time (seconds)")  # type: ignore
plt.yscale("log")  # type: ignore
plt.title("Simulation Timing Comparison per # of ROI (dt = 0.005)")  # type: ignore
plt.legend()  # type: ignore
plt.grid()  # type: ignore
plt.tight_layout()  # type: ignore
plt.show()  # type: ignore
