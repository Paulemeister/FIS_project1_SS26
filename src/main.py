from pathlib import Path
from lib import load_data, plot_gmres_stats, plot_cg_stats, save_stats, plot_matrix
from lib.gmres import (
    gmres,
    jacobi_preconditioner,
    gauss_seidel_preconditioner,
    ILU_preconditioner,
)
from lib.cg import CG
import numpy as np
import pandas as pd

OUT_DIR = Path("./out")


def gmres_analysis():
    SUB_OUT_DIR = OUT_DIR / "GMRES"
    SUB_OUT_DIR.mkdir(exist_ok=True)

    A_m = load_data(Path("./gmres_matrix_msr.txt"))

    n = A_m.shape[0]

    x_sol = np.ones(n)
    b = A_m @ x_sol
    tol = 1e-8
    x0 = np.zeros(n)

    restarts = [-1, 10, 25, 50, 200, 555]
    # restarts = [200, 300]
    precons = [  # type: ignore
        ("-", None),
        ("J", jacobi_preconditioner(A_m)),
        (
            "GS",
            gauss_seidel_preconditioner(A_m),
        ),
        (
            "ILU",
            ILU_preconditioner(A_m),
        ),
    ]

    stats: list[pd.DataFrame] = []

    max_restarts = 1000

    b_norm = np.linalg.norm(b)

    for label, precon in precons:  # type: ignore
        print("\n" + "#" * 70)
        print(f"Running {label if label is not None else "No Precon"}")
        print("#" * 70)
        for n in restarts:

            print(f"\nRestart after {n} vectors:")

            x_calc, _, gmres_stats = gmres(
                A_m, b, tol, x0, max_inner=n, max_restarts=max_restarts, l_pre=precon
            )

            gmres_stats.attrs["preconditioner"] = label

            res_true = np.linalg.norm(A_m @ x_calc - b)
            res_rel_true = res_true / b_norm

            print("CONVERGED" if gmres_stats.attrs["converged"] else "NOT CONVERGED")
            print(
                f"internal residual (rel): \t{gmres_stats.attrs["residual"]} ({gmres_stats.attrs["rel_residual"]})"
            )
            print(f"actual residual (rel): \t\t{res_true} ({res_rel_true})")

            stats.append(gmres_stats)

    plot_gmres_stats(stats, SUB_OUT_DIR)
    save_stats(
        stats,
        SUB_OUT_DIR / "gmres_stats.csv",
    )
    plot_matrix(SUB_OUT_DIR / "gmres_matrix.png", A_m)


def cg_analysis():
    SUB_OUT_DIR = OUT_DIR / "CG"
    SUB_OUT_DIR.mkdir(exist_ok=True)

    inputs = [
        ("M1", Path("./cg_matrix_msr_1.txt")),
        ("M2", Path("./cg_matrix_msr_2.txt")),
    ]

    tol = 1e-8

    stats: list[pd.DataFrame] = []

    for label, path in inputs:  # type: ignore
        print("\n" + "#" * 70)
        print(f"Running {label}")
        print("#" * 70)

        A_m = load_data(path)
        n = A_m.shape[0]
        x_sol = np.ones(n)
        b = A_m @ x_sol
        b_norm = np.linalg.norm(b)

        x0 = np.zeros(n)

        x_calc, _, cg_stats = CG(
            A_m, b, x0, tol=tol, solution=x_sol, max_iter=10_000_000
        )

        cg_stats.attrs["data"] = label

        res_true = np.linalg.norm(A_m @ x_calc - b)
        res_rel_true = res_true / b_norm

        print("CONVERGED" if cg_stats.attrs["converged"] else "NOT CONVERGED")
        print(
            f"internal residual (rel): \t{cg_stats.attrs["residual"]} ({cg_stats.attrs["rel_residual"]})"
        )
        print(f"actual residual (rel): \t\t{res_true} ({res_rel_true})")

        stats.append(cg_stats)
        plot_matrix(SUB_OUT_DIR / f"cg_matrix_{label}.png", A_m)

    plot_cg_stats(stats, SUB_OUT_DIR)
    save_stats(
        stats,
        SUB_OUT_DIR / "CG_stats.csv",
    )


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)

    gmres_analysis()
    cg_analysis()
