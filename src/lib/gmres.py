import numpy as np
import pandas as pd
import scipy as sp
import time
from numpy.typing import NDArray
from scipy.sparse import csr_array
from numpy import floating
from typing import Any, Callable
from lib import check_pd


def get_krylov(
    A_m: csr_array,
    V_m: NDArray[Any],
    j: int,
    l_pre: Callable[[NDArray[Any]], NDArray[Any]] | None = None,
) -> tuple[NDArray[Any] | None, NDArray[Any]]:
    n, _ = V_m.shape
    if l_pre is None:
        w = A_m @ V_m[:, j]
    else:
        w = l_pre(A_m @ V_m[:, j])
    h_j = np.zeros((n + 1))
    for i in range(j + 1):
        v_i = V_m[:, i]
        h_i_j = np.dot(v_i, w)
        h_j[i] = h_i_j
        w = w - h_i_j * v_i

    h_jp1_j = np.linalg.norm(w)
    h_j[j + 1] = h_jp1_j

    if np.isclose(h_jp1_j, 0.0):
        # Krilov Space already spanned by existing vectors
        return None, h_j
    v_jp1 = w / h_jp1_j
    return v_jp1, h_j


def Givens(a: floating, b: floating) -> tuple[floating, floating, floating]:
    r = np.sqrt(a**2 + b**2)
    c = a / r
    s = b / r
    return c, s, r


def _gmres(
    A_m: csr_array,
    b: NDArray[Any],
    x0: NDArray[Any],
    tol: float = 1e-8,
    max_iter: int = -1,
    l_pre: Callable[[NDArray[Any]], NDArray[Any]] | None = None,
) -> tuple[NDArray[Any], float, list[dict[str, Any]]]:

    if l_pre is None:
        r0 = b - A_m @ x0
    else:
        r0 = l_pre(b - A_m @ x0)

    # initialize helper variables
    stats: list[dict[str, Any]] = []
    M = A_m.shape[0]
    e1 = np.eye(M + 1)[:, 0]
    b_norm = np.linalg.norm(b)
    r0_norm = np.linalg.norm(r0)
    arr_size = 0
    use_dyn_arr_size = max_iter < 0

    if use_dyn_arr_size:
        arr_size = 10
        V_m = np.zeros((M, arr_size + 1))
        H_m = np.zeros((M + 1, arr_size))
    else:
        V_m = np.zeros((M, max_iter + 1))
        H_m = np.zeros((M + 1, max_iter))

    v1 = r0 / r0_norm
    V_m[:, 0] = v1
    g = r0_norm * e1
    c = np.zeros(M)
    s = np.zeros(M)

    res_norm: float = np.inf

    t1 = time.perf_counter()

    if max_iter < 0:
        lim = M
    else:
        lim = min(max_iter, M)

    j = 0

    for j in range(lim):

        v_jp1, h_j = get_krylov(A_m, V_m, j, l_pre=l_pre)

        if use_dyn_arr_size and j == arr_size:
            arr_size *= 2
            new_V_m = np.zeros((M, arr_size + 1))
            new_H_m = np.zeros((M + 1, arr_size))
            new_V_m[:, :j] = V_m[:, :j]
            new_H_m[:, :j] = H_m[:, :j]
            V_m = new_V_m
            H_m = new_H_m

        H_m[:, j] = h_j

        for k in range(0, j):
            G_m = np.array([[c[k], s[k]], [-s[k], c[k]]])
            H_m[(k) : (k + 2), j] = G_m @ H_m[(k) : (k + 2), j]

        c[j], s[j], d = Givens(H_m[j, j], H_m[j + 1, j])
        H_m[j, j] = d
        H_m[j + 1, j] = 0
        G_m = np.array([[c[j], s[j]], [-s[j], c[j]]])

        g[j : (j + 2)] = G_m @ g[j : (j + 2)]

        res_norm = np.abs(g[j + 1])  # TODO: Check if necessary

        stats.append(
            {
                "res": res_norm,
                "rel_res": res_norm / b_norm,
                "time": time.perf_counter() - t1,
                "iteration": j + 1,
                "orth": (
                    np.dot(
                        V_m[:, 0],
                        # V_m[:, j],
                        v_jp1,
                    )
                    if v_jp1 is not None
                    else np.nan
                ),
            }
        )

        if v_jp1 is None:  # Krylov space already spanned by Basis
            break

        V_m[:, j + 1] = v_jp1

        if res_norm < tol * b_norm:
            break

    size = j + 1

    R_m = H_m[:size, :size]

    y = np.linalg.solve(R_m, g[:size])

    x = x0 + V_m[:, :size] @ y

    return x, res_norm, stats


def gmres(
    A_m: csr_array,
    b: NDArray[Any],
    tol: float = 1e-8,
    x0: NDArray[Any] | None = None,
    max_inner: int = -1,
    max_restarts: int = 100,
    l_pre: Callable[[NDArray[Any]], NDArray[Any]] | None = None,
) -> tuple[NDArray[Any], float, pd.DataFrame]:

    if x0 is None:
        x0 = np.zeros_like(b)

    if not check_pd(A_m):
        print("Error: Matrix not PD")

        stats = pd.DataFrame(
            [
                {
                    "res": np.nan,
                    "rel_res": np.nan,
                    "error": np.nan,
                    "time": np.nan,
                    "iteration": 0,
                    "orth": np.nan,
                }
            ]
        )
        stats.attrs["method"] = "GMRES"
        stats.attrs["preconditioner"] = ""
        stats.attrs["converged"] = False
        stats.attrs["rel_tolerance"] = tol
        stats.attrs["tolerance"] = tol * np.linalg.norm(b)
        stats.attrs["residual"] = np.nan
        stats.attrs["rel_residual"] = np.nan
        return x0, np.nan, stats

    stats_cum: list[dict[str, Any]] = []

    iter_cum = 0
    time_cum = 0
    i = 0

    x = x0
    b_norm = np.linalg.norm(b)
    abs_tol = tol * b_norm
    res = np.inf

    while res > abs_tol and i < max_restarts:
        print(f"restart {i}, res: {res} ({res/b_norm})", end="\r")
        used_x0 = x
        x, res, stats = _gmres(A_m, b, used_x0, tol, max_iter=max_inner, l_pre=l_pre)
        for entry in stats:
            entry["restart"] = i
            entry["iteration"] += iter_cum
            entry["time"] += time_cum

        iter_cum = stats[-1]["iteration"]
        time_cum = stats[-1]["time"]
        stats_cum += stats
        i += 1

    # clear line
    print("\r\033[K", end="")

    converged = res <= abs_tol

    stats = pd.DataFrame(stats_cum)
    stats.attrs["method"] = "GMRS"
    stats.attrs["preconditioner"] = "" if l_pre is None else "Left Precond."
    stats.attrs["restarts"] = i - 1
    stats.attrs["max_inner"] = max_inner
    stats.attrs["converged"] = converged
    stats.attrs["rel_tolerance"] = tol
    stats.attrs["tolerance"] = abs_tol
    stats.attrs["residual"] = res
    stats.attrs["rel_residual"] = res / b_norm
    stats.attrs["total_time"] = time_cum
    stats.attrs["iterations"] = iter_cum

    return x, res, stats


def jacobi_preconditioner(
    A_m: csr_array,
) -> Callable[[NDArray[Any]], NDArray[Any]]:
    diag = A_m.diagonal()
    return lambda x: x / diag


def gauss_seidel_preconditioner(
    A_m: csr_array,
) -> Callable[[NDArray[Any]], NDArray[Any]]:
    lower_tr = sp.sparse.tril(A_m, format="csr")
    return lambda x: sp.sparse.linalg.spsolve_triangular(lower_tr, x, lower=True)


def ILU(A_m: csr_array, pattern: Callable[[int, int], bool] | None = None) -> csr_array:

    # FIXME: exploit sparse array, this algorithm is creating a dense matrix

    M_m = A_m.toarray()
    n = A_m.shape[0]

    if pattern is None:
        mask: NDArray[np.bool] = np.logical_not(np.isclose(M_m, 0.0))
    else:
        mask: NDArray[np.bool] = np.fromfunction(pattern, (n, n), dtype=int)  # type: ignore

    for i in range(1, n):
        for j in range(i):
            if mask[i, j]:
                M_m[i, j] = M_m[i, j] / M_m[j, j]

                for k in range(j + 1, n):
                    if mask[i, k]:
                        M_m[i, k] = M_m[i, k] - M_m[i, j] * M_m[j, k]
    return csr_array(M_m)


def ILU_preconditioner(
    A_m: csr_array,
) -> Callable[[NDArray[Any]], NDArray[Any]]:
    n = A_m.shape[0]
    M_m = ILU(A_m)
    L_m = sp.sparse.tril(M_m, k=-1, format="csr") + np.eye(n)
    U_m = sp.sparse.triu(M_m, format="csr")

    def solve(x: NDArray[Any]):
        y2 = sp.sparse.linalg.spsolve_triangular(L_m, x, lower=True, unit_diagonal=True)
        y1 = sp.sparse.linalg.spsolve_triangular(U_m, y2, lower=False)
        return y1

    return lambda x: solve(x)
