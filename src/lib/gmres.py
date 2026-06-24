import numpy as np
import pandas as pd
import scipy as sp
import time
from numpy.typing import NDArray
from numpy import floating
from typing import Any, Callable


def get_krylov(
    A_m: NDArray[Any],
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
    A_m: NDArray[Any],
    b: NDArray[Any],
    x0: NDArray[Any],
    tol: float = 1e-8,
    max_iter: int = -1,
    l_pre: Callable[[NDArray[Any]], NDArray[Any]] | None = None,
) -> tuple[NDArray[Any], float, list[dict[str, Any]]]:

    stats: list[dict[str, Any]] = []

    M = A_m.shape[0]
    e1 = np.eye(M + 1)[:, 0]

    b_norm = np.linalg.norm(b)

    if l_pre is None:
        r0 = b - A_m @ x0
    else:
        r0 = l_pre(b - A_m @ x0)
    r0_norm = np.linalg.norm(r0)
    g = r0_norm * e1

    if max_iter < 0:
        V_m = r0[:, None] / r0_norm
        H_m = np.empty((M + 1, 0))
    else:
        v1 = r0 / r0_norm

        V_m = np.zeros((M, max_iter + 1))
        V_m[:, 0] = v1
        H_m = np.zeros((M + 1, max_iter))

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

        if max_iter < 0:
            H_m = np.hstack((H_m, h_j[:, None]))
        else:
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

        if max_iter < 0:
            V_m = np.hstack((V_m, v_jp1[:, None]))
        else:
            V_m[:, j + 1] = v_jp1

        if res_norm < tol * b_norm:
            break

    size = j + 1

    R_m = H_m[:size, :size]

    y = np.linalg.solve(R_m, g[:size])

    x = x0 + V_m[:, :size] @ y

    return x, res_norm, stats


def gmres(
    A_m: NDArray[Any],
    b: NDArray[Any],
    tol: float = 1e-8,
    x0: NDArray[Any] | None = None,
    max_inner: int = -1,
    max_restarts: int = 100,
    l_pre: Callable[[NDArray[Any]], NDArray[Any]] | None = None,
) -> tuple[NDArray[Any], float, pd.DataFrame]:

    if x0 is None:
        x0 = np.zeros_like(b)

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

    return x, res, stats


def jacobi_preconditioner(
    A_m: NDArray[Any],
) -> Callable[[NDArray[Any]], NDArray[Any]]:
    diag = np.diag(A_m)
    return lambda x: x / diag


def gauss_seidel_preconditioner(
    A_m: NDArray[Any],
) -> Callable[[NDArray[Any]], NDArray[Any]]:
    lower_tr = np.tril(A_m)
    return lambda x: sp.linalg.solve_triangular(lower_tr, x, lower=True)


def ILU(
    A_m: NDArray[Any], pattern: Callable[[int, int], bool] | None = None
) -> NDArray[Any]:
    if pattern is None:
        pattern = lambda i, j: not np.isclose(A_m[i, j], 0.0)
    M_m = A_m.copy()
    n = M_m.shape[0]
    for i in range(1, n):
        for j in range(i):
            if pattern(i, j):
                M_m[i, j] = M_m[i, j] / M_m[j, j]

                for k in range(j + 1, n):
                    if pattern(i, k):
                        M_m[i, k] = M_m[i, k] - M_m[i, j] * M_m[j, k]
    return M_m


def ILU_preconditioner(
    A_m: NDArray[Any],
) -> Callable[[NDArray[Any]], NDArray[Any]]:
    n = A_m.shape[0]
    M_m = ILU(A_m)
    L_m = np.tril(M_m, k=-1) + np.eye(n)
    U_m = np.triu(M_m)

    def solve(x: NDArray[Any]):
        y2 = sp.linalg.solve_triangular(L_m, x, lower=True, unit_diagonal=True)
        y1 = sp.linalg.solve_triangular(U_m, y2, lower=False)
        return y1

    return lambda x: solve(x)
