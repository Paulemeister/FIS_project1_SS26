from pathlib import Path
from lib import (
    load_data,
    plot_stats,
    gmrs,
    jacobi_preconditioner,
    gauss_seidel_preconditioner,
)
import numpy as np
import pandas as pd

OUT_DIR = Path("./out")

if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)

    A_m = load_data(Path("./gmres_matrix_msr.txt"))

    n = A_m.shape[0]

    x_sol = np.ones(n)
    b = A_m @ x_sol
    tol = 1e-8
    x0 = np.zeros(n)

    restarts = [-1, 10, 25, 50, 20]
    # restarts = [200, 300]
    precons = [  # type: ignore
        ("-", None),
        ("J", jacobi_preconditioner(A_m)),
        (
            "GS",
            gauss_seidel_preconditioner(A_m),
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

            x_calc, res, gmrs_stats = gmrs(
                A_m, b, tol, x0, max_inner=n, max_restarts=max_restarts, l_pre=precon
            )

            gmrs_stats.attrs["preconditioner"] = label

            res_true = np.linalg.norm(A_m @ x_calc - b)
            res_rel_true = res_true / b_norm

            print("CONVERGED" if gmrs_stats.attrs["converged"] else "NOT CONVERGED")
            print(
                f"internal residual (rel): \t{gmrs_stats.attrs["residual"]} ({gmrs_stats.attrs["rel_residual"]})"
            )
            print(f"actual residual (rel): \t\t{res_true} ({res_rel_true})")

            stats.append(gmrs_stats)

    plot_stats(stats, OUT_DIR)
