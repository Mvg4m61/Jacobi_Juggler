"""
Jacobi Iteration Method - https://en.wikipedia.org/wiki/Jacobi_method
"""
from __future__ import annotations

import numpy as np
from numpy import float64
from numpy.typing import NDArray


# Method to find solution of system of linear equations
def jacobi_iteration_method(
    coefficient_matrix: NDArray[float64],
    constant_matrix: NDArray[float64],
    init_val: list[float],
    iterations: int,
) -> list[float]:
    """
    Jacobi Iteration Method:
    An iterative algorithm to determine the solutions of strictly diagonally dominant
    system of linear equations

    4x1 +  x2 +  x3 =  2
     x1 + 5x2 + 2x3 = -6
     x1 + 2x2 + 4x3 = -4

    x_init = [0.5, -0.5 , -0.5]

    Examples:
