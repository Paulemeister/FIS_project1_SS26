import numpy as np
import pandas as pd
import scipy as sp
from scipy.sparse import lil_array, csr_array
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

REL_RES_STR = r"$\frac{\|r\|}{\|b\|}$"
RES_STR = r"$\|r\|$"
ERR_A_STR = r"$\|e\|_A$"


def load_data(path: Path) -> csr_array:
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
    out = lil_array((n_diag, n_diag))
    out.setdiag(diag_vals)

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

    return out.tocsr()


def make_plot(
    col: str,
    x_label: str,
    stats: list[pd.DataFrame],
    log: bool = True,
    rel_tol: bool = True,
    abs_tol: bool = False,
    label_type: str = "gmres",
):
    if label_type == "cg":
        label_fn = lambda x: f"{x.attrs["method"]} {x.attrs["data"]}"  # type: ignore
    else:
        label_fn = lambda x: f"{x.attrs["method"]} ({x.attrs["max_inner"]}) {x.attrs["preconditioner"]}"  # type: ignore

    fig, ax = plt.subplots()

    for df in stats:
        label: str = label_fn(df)
        df.plot(y=col, ax=ax, use_index=True, label=label)

    if rel_tol:
        ax.axhline(
            y=stats[0].attrs["rel_tolerance"],
            ls="--",
            color="tab:red",
        )
    if abs_tol:
        ax.axhline(
            y=stats[0].attrs["tolerance"],
            ls="--",
            color="tab:green",
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
    fig, _ = make_plot("rel_res", REL_RES_STR, full_gmres_stats)
    fig.savefig(out_path / "full_gmres_rel_res.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    full_gmres_no_prec_stats = [
        stat for stat in full_gmres_stats if stat.attrs["preconditioner"] == "-"
    ]

    # orthogonality
    fig, ax = make_plot(
        "orth", r"$v_1 \cdot v_m$", full_gmres_no_prec_stats, rel_tol=False
    )
    line = ax.get_lines()[0]
    line.set_linestyle("none")
    line.set_marker("o")
    fig.savefig(out_path / "full_gmres_no_prec_orth.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # All other
    rest_gmres_stats = [stat for stat in stats if stat.attrs["max_inner"] != -1]

    # relative residual
    fig, _ = make_plot("rel_res", REL_RES_STR, rest_gmres_stats)
    fig.savefig(out_path / "rest_gmres_rel_res.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    # runtime
    fig, _ = make_plot("time", r"$t$", rest_gmres_stats, log=False, rel_tol=False)
    fig.savefig(out_path / "rest_gmres_time.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    ms = set([stat.attrs["max_inner"] for stat in stats])

    for m in ms:
        gmres_m_stats = [stat for stat in stats if stat.attrs["max_inner"] == m]
        # relative residual
        fig, _ = make_plot("rel_res", REL_RES_STR, gmres_m_stats)
        fig.savefig(out_path / f"gmres_{m}_rel_res.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        # runtime
        fig, _ = make_plot("time", r"$t$", gmres_m_stats, log=False, rel_tol=False)
        fig.savefig(out_path / f"gmres_{m}_time.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    precons = set([stat.attrs["preconditioner"] for stat in stats])

    for precon in precons:
        gmres_m_stats = [
            stat for stat in stats if stat.attrs["preconditioner"] == precon
        ]
        # relative residual
        fig, _ = make_plot("rel_res", REL_RES_STR, gmres_m_stats)
        fig.savefig(
            out_path / f"gmres_{precon}_rel_res.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)
        # runtime
        fig, _ = make_plot("time", r"$t$", gmres_m_stats, log=False, rel_tol=False)
        fig.savefig(out_path / f"gmres_{precon}_time.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    plot_restarts(stats, out_path)


def plot_cg_stats(stats: list[pd.DataFrame], out_path: Path):

    # residual
    fig, _ = make_plot("res", RES_STR, stats, label_type="cg", rel_tol=False)
    fig.savefig(out_path / "cg_all_res.png", dpi=300, bbox_inches="tight")

    # relative residual
    fig, _ = make_plot("rel_res", REL_RES_STR, stats, label_type="cg")
    fig.savefig(out_path / "cg_all_rel_res.png", dpi=300, bbox_inches="tight")

    # error in A norm
    fig, _ = make_plot("error", ERR_A_STR, stats, label_type="cg", rel_tol=False)
    fig.savefig(out_path / "cg_all_error.png", dpi=300, bbox_inches="tight")

    for df in stats:
        data_label = df.attrs["data"]
        fig, ax = plt.subplots()

        df.plot(y="res", ax=ax, use_index=True, label=RES_STR)
        df.plot(y="rel_res", ax=ax, use_index=True, label=REL_RES_STR)
        df.plot(y="error", ax=ax, use_index=True, label=ERR_A_STR)

        ax.axhline(
            y=df.attrs["rel_tolerance"],
            ls="--",
            color="tab:red",
        )

        ax.set_yscale("log")
        ax.set_xlabel("iterations")
        ax.set_ylabel("")

        ax.grid()
        if (legend := ax.get_legend()) is not None:
            legend.remove()
        fig.legend()

        fig.savefig(
            out_path / f"cg_{data_label}_combined.png", dpi=300, bbox_inches="tight"
        )


def save_stats(stats: list[pd.DataFrame], out_path: Path):
    meta = [stat.attrs for stat in stats]
    pd.DataFrame(meta).to_csv(out_path)


def check_sym(A: csr_array):

    return np.allclose((A - A.T).data, 0)


def check_pd(A: csr_array):

    sigma, _ = sp.sparse.linalg.eigs(A)
    return np.all(sigma > 0.0)


def check_spd(A: csr_array):
    return check_sym(A) and check_pd(A)


def plot_matrix(out_path: Path, A_m: csr_array):
    A_coo = A_m.tocoo()
    a_min = A_m.min(explicit=True)
    a_max = A_m.max(explicit=True)
    a_abs_max = max(np.abs(a_min), np.abs(a_max))

    fig, ax = plt.subplots()
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-a_abs_max, vmax=a_abs_max)

    sc = ax.scatter(
        A_coo.col,
        A_coo.row,
        c=A_coo.data,
        cmap="seismic",
        norm=norm,
        marker="s",
        s=1,
    )
    ax.invert_yaxis()
    ax.set_aspect("equal")

    fig.colorbar(sc, ax=ax)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")


def plot_restarts(stats: list[pd.DataFrame], out_dir: Path):

    meta_df = pd.DataFrame([stat.attrs for stat in stats])

    fig_kr, ax_kr = plt.subplots()
    fig_t, ax_t = plt.subplots()
    for prec, df in meta_df.groupby("preconditioner"):
        m, n_kryl, t = (
            df.sort_values("max_inner")[["max_inner", "restarts", "total_time"]]
            .to_numpy()
            .T
        )
        ax_kr.plot(m, n_kryl, "-o", label=prec)
        ax_t.plot(m, t, "-o", label=prec)

    ax_t.set_xlabel("m")
    ax_kr.set_xlabel("m")
    ax_t.set_ylabel("t")
    ax_kr.set_ylabel("restarts")
    ax_t.grid()
    ax_kr.grid()
    fig_t.legend()
    fig_kr.legend()
    fig_kr.savefig(out_dir / "gmres_krylov_m.png", dpi=300, bbox_inches="tight")
    fig_t.savefig(out_dir / "gmres_t_m.png", dpi=300, bbox_inches="tight")
