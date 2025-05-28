from typing import Callable, TypeVar


def zeros_2d(x: int, y: int) -> list[list[float]]:
    return [[0.0 for _ in range(y)] for _ in range(x)]


def zeros_3d(x: int, y: int, z: int) -> list[list[list[float]]]:
    return [[[0.0 for _ in range(z)] for _ in range(y)] for _ in range(x)]


T = TypeVar("T")
U = TypeVar("U")


def compute_2d(matrix: list[list[T]], fn: Callable[[T], U]) -> list[list[U]]:
    return [[fn(value) for value in row] for row in matrix]


def multiply_2d(M: list[list[float]], factor: float) -> list[list[float]]:
    return [[value * factor for value in row] for row in M]


def divide_2d(M: list[list[float]], factor: float) -> list[list[float]]:
    if factor == 0:
        raise ValueError("Division by zero is not allowed.")
    return [[value / factor for value in row] for row in M]
