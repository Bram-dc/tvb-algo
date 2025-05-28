def zeros_3d(H: int, n: int, ncv: int) -> list[list[list[float]]]:
    return [[[0.0 for _ in range(ncv)] for _ in range(n)] for _ in range(H)]
