import numpy as np
import pandas as pd
import time
from numpy.typing import NDArray
from scipy.sparse import csr_array
from typing import Any
from lib.misc import check_sym, check_pd


def CG(
    A_m: csr_array,
    b: NDArray[Any],
    x0: NDArray[Any] | None = None,
    max_iter: int = 10000,
    tol: float = 1e-8,
    solution: NDArray[Any] | None = None,
) -> tuple[NDArray[Any], float, pd.DataFrame]:

    if x0 is None:
        x0 = np.zeros_like(b)

    if solution is None:
        solution = np.zeros_like(b) * np.nan

    is_sym = check_sym(A_m)
    is_pd = check_pd(A_m)
    if not is_sym or not is_pd:
        if not is_sym:
            print("Error: Matrix not Symmetric")
        if not is_pd:
            print("Error: Matrix not PD")

        stats = pd.DataFrame(
            [
                {
                    "res": np.nan,
                    "rel_res": np.nan,
                    "error": np.nan,
                    "time": np.nan,
                    "iteration": 0,
                }
            ]
        )
        stats.attrs["method"] = "CG"
        stats.attrs["preconditioner"] = ""
        stats.attrs["converged"] = False
        stats.attrs["rel_tolerance"] = tol
        stats.attrs["tolerance"] = tol * np.linalg.norm(b)
        stats.attrs["residual"] = np.nan
        stats.attrs["rel_residual"] = np.nan
        return x0, np.nan, stats

    res_norm: float = np.nan
    rel_res_norm: float = np.nan
    b_norm = np.linalg.norm(b)
    abs_tol = tol * b_norm

    x_i = x0
    x_ip1 = np.zeros_like(x0)
    r_i = b - A_m @ x0
    r_ip1 = np.zeros_like(x0)

    p_i = r_i

    converged = False
    stats_list = []  # type: ignore

    t1 = time.perf_counter()
    i = 1
    while i < max_iter:
        if i % 20 == 1:
            print(
                f"iteration: {i}, residual (rel): {res_norm} ({rel_res_norm})", end="\r"
            )

        A_p_prod = A_m @ p_i

        a_m = np.dot(r_i, r_i) / np.dot(p_i, A_p_prod)

        x_ip1 = x_i + a_m * p_i
        r_ip1 = r_i - a_m * A_p_prod

        b_i = np.dot(r_ip1, r_ip1) / np.dot(r_i, r_i)
        p_ip1 = r_ip1 + b_i * p_i

        # update variables
        p_i = p_ip1
        r_i = r_ip1
        x_i = x_ip1

        res_norm = np.linalg.norm(r_ip1)  # type: ignore
        rel_res_norm = res_norm / b_norm  # type: ignore
        error = solution - x_ip1
        error_A_norm = np.sqrt(np.dot(A_m @ error, error))

        stats_list.append(
            {
                "res": res_norm,
                "rel_res": res_norm / b_norm,
                "error": error_A_norm,
                "time": time.perf_counter() - t1,
                "iteration": i,
            }
        )
        if res_norm < abs_tol:
            converged = True
            break
        i += 1

    # clear line
    print("\r\033[K", end="")

    stats = pd.DataFrame(stats_list)
    stats.attrs["method"] = "CG"
    stats.attrs["preconditioner"] = ""
    stats.attrs["converged"] = converged
    stats.attrs["rel_tolerance"] = tol
    stats.attrs["tolerance"] = tol * b_norm
    stats.attrs["residual"] = res_norm
    stats.attrs["rel_residual"] = res_norm / b_norm
    stats.attrs["total_time"] = time.perf_counter() - t1
    stats.attrs["iterations"] = i
    return x_ip1, res_norm, stats
