import numpy as np
import pandas as pd
import time
from pathlib import Path
from numpy.typing import NDArray
from numpy import floating
from typing import Any

import matplotlib.pyplot as plt

Mat = NDArray[np.floating]


def load_data(path: Path):
    with open(path) as f:
        stor_type = f.readline().strip()
        if stor_type not in ["s", "n"]:
            raise Exception("bad header")

        data = np.loadtxt(f)

    n_diag, _ = data[0].astype(int)

    diag_vals = data[1 : (n_diag + 1), 1]

    bindx = data[0:, 0].astype(int)
    vals = data[0:, 1]

    # FIXME: This is bad, use sparse instead

    out = np.diag(diag_vals)

    row_ptr = bindx[1]

    for row_ix in range(n_diag):
        next_row_ptr = bindx[2 + row_ix]
        for j in range(next_row_ptr - row_ptr):
            col_ix = bindx[row_ptr + j] - 1
            val = vals[row_ptr + j]
            out[row_ix, col_ix] = val

        row_ptr = next_row_ptr

    if stor_type == "s":
        out += np.tril(out, k=-1).T

    return out


# TODO: Change API to allow V to be any size, get index j
def get_krylov(
    A_m: NDArray[Any],
    V_m: NDArray[Any],
) -> tuple[NDArray[Any] | None, NDArray[Any]]:
    n, k = V_m.shape
    w = A_m @ V_m[:, -1]
    h_j = np.zeros((n + 1))
    for i in range(k):
        v_i = V_m[:, i]
        h_i_j = np.dot(v_i, w)
        h_j[i] = h_i_j
        w = w - h_i_j * v_i

    h_jp1_j = np.linalg.norm(w)
    h_j[k] = h_jp1_j

    if np.isclose(h_jp1_j, 0.0):
        # Krilov Space already spanned by existing vectors
        # TODO: throw exception here?
        return None, h_j
    v_jp1 = w / h_jp1_j
    return v_jp1, h_j


def Givens(a: floating, b: floating) -> tuple[floating, floating, floating]:
    r = np.sqrt(a**2 + b**2)
    c = a / r
    s = b / r
    return c, s, r


def gmrs(
    A_m: NDArray[Any],
    b: NDArray[Any],
    tol: float = 1e-8,
    x0: NDArray[Any] | None = None,
    return_res: bool = False,
) -> tuple[NDArray[Any], pd.DataFrame] | NDArray[Any]:
    if x0 is None:
        x0 = np.zeros_like(b)

    stats: pd.DataFrame | None = None
    t1: float | None = None

    M = A_m.shape[0]
    e1 = np.eye(M + 1)[:, 0]

    b_norm = np.linalg.norm(b)

    r0 = b - A_m @ x0
    r0_norm = np.linalg.norm(r0)
    V_m = r0[:, None] / r0_norm
    g = r0_norm * e1

    H_m = np.empty((M + 1, 0))

    c = np.zeros(M)
    s = np.zeros(M)

    res_norm = np.inf

    if return_res:
        t1 = time.perf_counter()
        stats = pd.DataFrame(columns=["res", "time"])

    for j in range(M):
        v_jp1, h_j = get_krylov(A_m, V_m)

        H_m = np.hstack((H_m, h_j[:, None]))

        for k in range(0, j):
            G_m = np.array([[c[k], s[k]], [-s[k], c[k]]])
            H_m[(k) : (k + 2), j] = G_m @ H_m[(k) : (k + 2), j]

        c[j], s[j], d = Givens(H_m[j, j], H_m[j + 1, j])
        H_m[j, j] = d
        H_m[j + 1, j] = 0
        G_m = np.array([[c[j], s[j]], [-s[j], c[j]]])

        g[j : (j + 2)] = G_m @ g[j : (j + 2)]

        res_norm = abs(g[j + 1])  # TODO: Check if necessary

        if return_res:
            assert stats is not None and t1 is not None
            stats.loc[j, "res"] = res_norm
            stats.loc[j, "rel_res"] = res_norm / b_norm
            stats.loc[j, "time"] = time.perf_counter() - t1
            stats.loc[j, "inner"] = True

        if v_jp1 is None:  # Krylov space already spanned by Basis
            break

        V_m = np.hstack((V_m, v_jp1[:, None]))

        if res_norm < tol * b_norm:
            break

    size = H_m.shape[1]

    R_m = H_m[:size, :size]

    y = np.linalg.solve(R_m, g[:size])

    x = x0 + V_m[:, :size] @ y

    if return_res:
        assert stats is not None and t1 is not None
        stats.attrs["res"] = res_norm
        stats.attrs["time"] = time.perf_counter() - t1
        stats.attrs["method"] = "GMRS"
        stats.attrs["tol"] = tol
        return x, stats
    return x


def plot_stats(stats: list[pd.DataFrame]):
    fig, [ax1, ax2] = plt.subplots(nrows=2)
    for df in stats:
        df.plot(y="time", ax=ax1, use_index=True)
        df.plot(y="res", ax=ax2, use_index=True)

    ax2.set_yscale("log")

    fig.show()
