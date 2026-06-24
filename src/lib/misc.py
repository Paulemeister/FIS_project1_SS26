import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from numpy.typing import NDArray
from typing import Any

REL_RES_STR = r"$\frac{\|r\|}{\|b\|}$"
RES_STR = r"$\|r\|$"
ERR_A_STR = r"$\|e\|_A$"


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
            if stor_type == "s":
                out[col_ix, row_ix] = val

        row_ptr = next_row_ptr

    return out


def make_gmrs_plot(
    col: str,
    x_label: str,
    stats: list[pd.DataFrame],
    log: bool = True,
    rel_tol: bool = True,
):
    fig, ax = plt.subplots()

    for df in stats:
        label = f"{df.attrs["method"]} ({df.attrs["max_inner"]}) {df.attrs["preconditioner"]}"
        df.plot(y=col, ax=ax, use_index=True, label=label)

    if rel_tol:
        ax.axhline(
            y=stats[0].attrs["rel_tolerance"],
            ls="--",
            color="tab:red",
        )
    if log:
        ax.set_yscale("log")
    ax.set_ylabel(x_label)
    ax.set_xlabel("iterations")
    ax.grid()
    if (legend := ax.get_legend()) is not None:
        legend.remove()
    fig.legend()
    return fig, ax


def plot_gmres_stats(stats: list[pd.DataFrame], out_path: Path):
    full_gmres_stats = [stat for stat in stats if stat.attrs["max_inner"] == -1]

    # relative residual
    fig, _ = make_gmrs_plot("rel_res", REL_RES_STR, full_gmres_stats)
    fig.savefig(out_path / "full_gmres_rel_res.png", dpi=300, bbox_inches="tight")

    full_gmres_no_prec_stats = [
        stat for stat in full_gmres_stats if stat.attrs["preconditioner"] == "-"
    ]

    # orthogonality
    fig, _ = make_gmrs_plot(
        "orth", r"$v_1 \cdot v_m$", full_gmres_no_prec_stats, rel_tol=False
    )
    fig.savefig(out_path / "full_gmres_no_prec_orth.png", dpi=300, bbox_inches="tight")

    # All other
    rest_gmres_stats = [stat for stat in stats if stat.attrs["max_inner"] != -1]

    # relative residual
    fig, _ = make_gmrs_plot("rel_res", REL_RES_STR, rest_gmres_stats)
    fig.savefig(out_path / "rest_gmres_rel_res.png", dpi=300, bbox_inches="tight")
    # runtime
    fig, _ = make_gmrs_plot("time", r"$t$", rest_gmres_stats, log=False)
    fig.savefig(out_path / "rest_gmres_time.png", dpi=300, bbox_inches="tight")

    ms = set([stat.attrs["max_inner"] for stat in stats])

    for m in ms:
        gmres_m_stats = [stat for stat in stats if stat.attrs["max_inner"] == m]
        # relative residual
        fig, _ = make_gmrs_plot("rel_res", REL_RES_STR, gmres_m_stats)
        fig.savefig(out_path / f"gmres_{m}_rel_res.png", dpi=300, bbox_inches="tight")
        # runtime
        fig, _ = make_gmrs_plot("time", r"$t$", gmres_m_stats, log=False)
        fig.savefig(out_path / f"gmres_{m}_time.png", dpi=300, bbox_inches="tight")

    precons = set([stat.attrs["preconditioner"] for stat in stats])

    for precon in precons:
        gmres_m_stats = [
            stat for stat in stats if stat.attrs["preconditioner"] == precon
        ]
        # relative residual
        fig, _ = make_gmrs_plot("rel_res", REL_RES_STR, gmres_m_stats)
        fig.savefig(
            out_path / f"gmres_{precon}_rel_res.png", dpi=300, bbox_inches="tight"
        )
        # runtime
        fig, _ = make_gmrs_plot("time", r"$t$", gmres_m_stats, log=False)
        fig.savefig(out_path / f"gmres_{precon}_time.png", dpi=300, bbox_inches="tight")


def plot_cg_stats(stats: list[pd.DataFrame], out_path: Path):
    err_fig, err_ax = plt.subplots()

    res_fig, res_ax = plt.subplots()

    rel_res_fig, rel_res_ax = plt.subplots()

    for df in stats:
        if np.isnan(df.attrs["residual"]):
            continue
        label = f"{df.attrs["method"]} {df.attrs["data"]}"
        df.plot(y="res", ax=res_ax, use_index=True, label=label)
        df.plot(y="rel_res", ax=rel_res_ax, use_index=True, label=label)
        df.plot(y="error", ax=err_ax, use_index=True, label=label)

    rel_res_ax.axhline(
        y=stats[0].attrs["rel_tolerance"],
        ls="--",
        color="tab:red",
    )
    res_ax.axhline(
        y=stats[0].attrs["tolerance"],
        ls="--",
        color="tab:red",
    )

    # residual
    res_ax.set_yscale("log")
    res_ax.set_ylabel(RES_STR)
    res_ax.set_xlabel("iterations")
    res_ax.grid()
    if (legend := res_ax.get_legend()) is not None:
        legend.remove()
    res_fig.legend()
    res_fig.savefig(out_path / "residual.png", dpi=300, bbox_inches="tight")

    # relative residual
    rel_res_ax.set_yscale("log")
    rel_res_ax.set_ylabel(REL_RES_STR)
    rel_res_ax.set_xlabel("iterations")
    rel_res_ax.grid()
    if (legend := rel_res_ax.get_legend()) is not None:
        legend.remove()
    rel_res_fig.legend()
    rel_res_fig.savefig(out_path / "rel_residual.png", dpi=300, bbox_inches="tight")

    # error in A norm
    err_ax.set_yscale("log")
    err_ax.set_ylabel(ERR_A_STR)
    err_ax.set_xlabel("iterations")
    err_ax.grid()
    if (legend := err_ax.get_legend()) is not None:
        legend.remove()
    err_fig.legend()
    err_fig.savefig(out_path / "error.png", dpi=300, bbox_inches="tight")


def check_sym(A: NDArray[Any]):
    return np.allclose(A, A.T)


def check_pd(A: NDArray[Any]):
    sigma = np.linalg.eigvalsh(A)
    return np.all(sigma > 0.0)


def check_spd(A: NDArray[Any]):
    return check_sym(A) and check_pd(A)
