import numpy as np
import pandas as pd
import time
from numpy.typing import NDArray
from typing import Any
from lib.misc import check_sym, check_pd


def CG(
    A_m: NDArray[Any],
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

    x_m = x0
    x_mp1 = np.zeros_like(x0)
    r_m = b - A_m @ x0
    r_mp1 = np.zeros_like(x0)

    p_m = r_m

    converged = False
    stats_list = []  # type: ignore

    t1 = time.perf_counter()
    i = 1
    while i < max_iter:
        if i % 20 == 1:
            print(
                f"iteration: {i}, residual (rel): {res_norm} ({rel_res_norm})", end="\r"
            )

        A_p_prod = A_m @ p_m

        a_m = np.dot(r_m, r_m) / np.dot(p_m, A_p_prod)

        x_mp1 = x_m + a_m * p_m
        r_mp1 = r_m - a_m * A_p_prod

        b_m = np.dot(r_mp1, r_mp1) / np.dot(r_m, r_m)
        p_mp1 = r_mp1 + b_m * p_m

        # update variables
        p_m = p_mp1
        r_m = r_mp1
        x_m = x_mp1

        res_norm = np.linalg.norm(r_mp1)  # type: ignore
        rel_res_norm = res_norm / b_norm  # type: ignore
        error = solution - x_mp1
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

    return x_mp1, res_norm, stats
