from pathlib import Path
from lib import load_data, plot_stats, gmrs
import numpy as np
import pandas as pd

OUT_DIR = Path("./out")

if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)

    A_m = load_data(Path("./cg_matrix_msr_1.txt"))

    n = A_m.shape[0]

    x_sol = np.ones(n)
    b = A_m @ x_sol
    tol = 1e-2
    x0 = np.zeros(n)

    # restarts = [-1, 50, 100, 200, 300]
    restarts = [200, 300]
    stats: list[pd.DataFrame] = []
    for n in restarts:
        x_calc, res, gmrs_stats = gmrs(A_m, b, tol, x0, max_inner=n, max_restarts=5)

        res = np.linalg.norm(A_m @ x_calc - b)
        rel_res = res / np.linalg.norm(b)
        if rel_res > tol:
            print(f"RESULT WRONG!\nn:{n}\nRel Res: {rel_res:.3g}")

        stats.append(gmrs_stats)

    plot_stats(stats, OUT_DIR)
