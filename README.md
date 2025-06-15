# Performance Comparison of `tvb-algo` Implementations

This document compares the performance of different implementations of the same algorithm in pure Python. The algorithm was originally developed in the [tvb-algo](https://github.com/maedoc/tvb-algo) repository, which provides a collection of algorithms for simulating brain activity on the [The Virtual Brain](https://www.thevirtualbrain.org/) platform. For simplicity, random noise is removed from the algorithm.

## Algorithms Tested
- **Original**: The original implementation (in `NumPy`), with noise removed.
- **Base**: A 1-to-1 translation to pure Python (without `NumPy`).
- **Base (Single State Variable)**: Optimized version that uses only a single state variable.
- **Parallel**: Parallelized using `concurrent.futures` with up to 32 threads.
- **JIT**: Compilation with `numba` to machine code.
- **JIT + Parallel**: Combination of `numba` and parallelization with `prange`.

## Implementations

Several implementations of the algorithm are tested in terms of performance. The following implementations were considered:

### 1. **Original**
The original implementation of the algorithm is written in `NumPy`. In this case, the random noise is removed to simplify the comparison. This version serves as the baseline for the performance comparison.

### 2. **Base**
This implementation is a **1-to-1 translation** of the original `NumPy` code to pure Python. In this version:
- Every 2D or 3D array is converted to a list of lists.
- NumPy functions are replaced with loops and list comprehensions.
This approach results in a Python implementation that mimics the original but without using any external libraries like `NumPy`.

### 3. **Base (Single State Variable)**
In this implementation, we leverage the fact that the example script only uses a **single state variable** per iteration. This realization allows us to:
- Remove unnecessary loops and list comprehensions.
- Refactor the code to improve efficiency.
This results in a cleaner and more efficient version of the base implementation.

### 4. **Parallel**
The parallel implementation utilizes the `concurrent.futures` module to parallelize the computation across multiple CPU cores. It uses a `ThreadPoolExecutor` with a maximum of **32 threads** (this can be adjusted to match the available CPU cores on your machine). This allows for parallel execution of the algorithm, improving performance on multi-core systems.

### 5. **JIT**
In this implementation, we use the `numba` library to compile the Python code to **machine code**. This can significantly speed up computations by optimizing the execution at runtime, offering a performance boost without requiring manual optimizations in the code.

### 6. **JIT + Parallel**
This version combines the `numba` library with parallelization. It uses the `prange` function to parallelize the computation across multiple CPU cores. This approach maximizes performance by compiling the code to machine code and leveraging parallel computation.

---

# Results

In this section, we will present the results of the performance comparison between the different implementations. Below are some images and plots that visually illustrate the differences in performance.

### Performance Comparison Plots
[Insert plot for time comparison between different implementations]

### Speedup of Parallelization and JIT
[Insert plot showing the effect of parallelization and JIT compilation on performance]
