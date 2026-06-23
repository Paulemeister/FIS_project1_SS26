import numpy as np
import pandas as pd
import time
from pathlib import Path
from numpy.typing import NDArray
from numpy import floating
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

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
    A_m: NDArray[Any], V_m: NDArray[Any], j: int
) -> tuple[NDArray[Any] | None, NDArray[Any]]:
    n, _ = V_m.shape
    w = A_m @ V_m[:, j]
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
        # TODO: throw exception here?
        return None, h_j
    v_jp1 = w / h_jp1_j
    return v_jp1, h_j


def Givens(a: floating, b: floating) -> tuple[floating, floating, floating]:
    r = np.sqrt(a**2 + b**2)
    c = a / r
    s = b / r
    return c, s, r


def _gmrs(
    A_m: NDArray[Any],
    b: NDArray[Any],
    x0: NDArray[Any],
    tol: float = 1e-8,
    max_iter: int = -1,
) -> tuple[NDArray[Any], float, list[dict[str, Any]]]:

    stats: list[dict[str, Any]] = []

    M = A_m.shape[0]
    e1 = np.eye(M + 1)[:, 0]

    b_norm = np.linalg.norm(b)

    r0 = b - A_m @ x0
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

        v_jp1, h_j = get_krylov(A_m, V_m, j)

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


def gmrs(
    A_m: NDArray[Any],
    b: NDArray[Any],
    tol: float = 1e-8,
    x0: NDArray[Any] | None = None,
    max_inner: int = -1,
    max_restarts: int = 100,
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
        print(f"restart {i}")
        used_x0 = x
        x, res, stats = _gmrs(A_m, b, used_x0, tol, max_iter=max_inner)
        for entry in stats:
            entry["restart"] = i
            entry["iteration"] += iter_cum
            entry["time"] += time_cum

        iter_cum = stats[-1]["iteration"]
        time_cum = stats[-1]["time"]
        stats_cum += stats
        i += 1

    converged = res > abs_tol

    stats = pd.DataFrame(stats_cum)
    stats.attrs["method"] = "GMRS"
    stats.attrs["restarts"] = i - 1
    stats.attrs["max_inner"] = max_inner
    stats.attrs["converged"] = converged
    stats.attrs["tolerance"] = tol
    return x, res, stats


def plot_stats(stats: list[pd.DataFrame], out_path: Path):
    _, axs = plt.subplots(nrows=2, sharex=True)
    ax1: Axes = axs[0]
    ax2: Axes = axs[1]

    for df in stats:
        label = f"{df.attrs["method"]} ({df.attrs["max_inner"]})"
        df.plot(y="time", ax=ax1, use_index=True, label=label)
        df.plot(y="res", ax=ax2, use_index=True, label=label)

    ax2.axhline(
        y=stats[0].attrs["tolerance"],
        ls="--",
        color="tab:red",
    )
    ax2.set_yscale("log")
    ax1.set_ylabel("time")
    ax2.set_ylabel("residual")
    ax1.set_xlabel("iterations")

    plt.savefig(out_path / "plot.png")
