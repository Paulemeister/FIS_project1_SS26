from pathlib import Path
from lib import load_data, plot_stats, gmrs
import numpy as np

OUT_DIR = Path("./out")

if __name__ == "__main__":
    A_m = load_data(Path("./cg_matrix_msr_1.txt"))

    n = A_m.shape[0]

    x_sol = np.ones(n)
    b = A_m @ x_sol
    tol = 1e-1
    x0 = np.zeros(n)

    x_calc, gmrs_stats = gmrs(A_m, b, tol, x0, return_res=True)

    res = np.linalg.norm(A_m @ x_calc - b)
    rel_res = res / np.linalg.norm(b)
    if rel_res > tol:
        print(f"RESULT WRONG!\nRel Res: {rel_res:.3g}")

    plot_stats([gmrs_stats])
