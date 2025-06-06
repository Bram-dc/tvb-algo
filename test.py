from lib import data
from base import simulation as base_simulation
from original import simulation as original_simulation
from base_single_ncv import simulation as base_single_ncv_simulation
from parallel import simulation as parallel_simulation
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

for i, speed in tqdm(enumerate([1.0, 2.0, 10.0])):
    T_base, Xs_base, _ = base_simulation.simulate(
        W_list, D_list, dt, tf, k, speed, freq
    )
    T_original, Xs_original, _ = original_simulation.simulate(
        W, D, dt, tf, k, speed, freq
    )
    T_base_single_ncv, Xs_base_single_ncv, _ = base_single_ncv_simulation.simulate(
        W_list, D_list, dt, tf, k, speed, freq
    )
    T_parallel, Xs_parallel, _ = parallel_simulation.simulate(
        W_list, D_list, dt, tf, k, speed, freq
    )
    T_jit, Xs_jit, _ = jit_simulation.simulate(W_list, D_list, dt, tf, k, speed, freq)

    # Compare results
    assert (
        len(T_base)
        == len(T_original)
        == len(T_base_single_ncv)
        == len(T_parallel)
        == len(T_jit)
    ), f"Length mismatch in T for speed {speed}"
    assert (
        len(Xs_base)
        == len(Xs_original)
        == len(Xs_base_single_ncv)
        == len(Xs_parallel)
        == len(Xs_jit)
    ), f"Length mismatch in Xs for speed {speed}"

    for x_base, x_original, x_base_single_ncv, x_parallel, x_jit in zip(
        Xs_base, Xs_original, Xs_base_single_ncv, Xs_parallel, Xs_jit
    ):
        assert (
            len(x_base)
            == len(x_original)
            == len(x_base_single_ncv)
            == len(x_parallel)
            == len(x_jit)
        ), f"Length mismatch in x for speed {speed}"

        for (
            xi_base,
            xi_original,
            xi_base_single_ncv,
            xi_parallel,
            xi_jit,
        ) in zip(x_base, x_original, x_base_single_ncv, x_parallel, x_jit):
            assert np.allclose(
                xi_original, xi_base, atol=tolerance
            ), f"Mismatch in Xs for speed {speed} at {xi_original} vs {xi_base} (original vs base)"
            assert np.allclose(
                xi_original, xi_base_single_ncv, atol=tolerance
            ), f"Mismatch in Xs for speed {speed} at {xi_original} vs {xi_base_single_ncv} (original vs base_single_ncv)"
            assert np.allclose(
                xi_original, xi_parallel, atol=tolerance
            ), f"Mismatch in Xs for speed {speed} at {xi_original} vs {xi_parallel} (original vs parallel)"
            assert np.allclose(
                xi_original, xi_jit, atol=tolerance
            ), f"Mismatch in Xs for speed {speed} at {xi_original} vs {xi_jit} (original vs jit)"


print("All tests passed successfully!")
