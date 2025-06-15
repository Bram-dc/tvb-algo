In this repository, several different implementations of the Python library (tvb-algo)[https://github.com/maedoc/tvb-algo] are provided. tvb-algo is a collection of algorithms for the [The Virtual Brain](https://www.thevirtualbrain.org/) platform, which is used for simulating brain activity.

I used the example script given in the tvb-algo repository to compare the performance of different implementations of the same algorithm in pure `Python`, without the use of `NumPy`. For simplicity, the random noise is removed from the algorithm.

There are a few different implementations of the same algorithm, which are compared in terms of performance. The implementations are:
- `original`: The original implementation of the algorithm (in `NumPy`), where the noise term is removed.
- `base`: This is a 1-to-1 translation of the original implementation to pure Python, without using any libraries. Every 2D or 3D array is converted to a list of lists, and every NumPy function is replaced with loops and list comprehensions.
- `base_single_ncv`: By first realizing that in the example script, we only use a single state variable in the simulation per iteration, we can simplify the implementation by removing the unnecessary loops and list comprehensions. In this implementation, there are also some other refactors to improve efficiency.
- `parallel`: This implementation uses the `concurrent.futures` module to parallelize the computation across multiple CPU cores using `ThreadPoolExecutor`, with a maximum of 32 threads. This number can be adjusted in the script to match the number of available CPU cores on your own machine.
- `jit`: This implementation uses the `numba` library to compile the Python code to machine code, which can significantly speed up the computation.
- `jit_parallel`: This implementation combines the `numba` library with parallelization, using the `prange` function to parallelize the computation across multiple CPU cores.

