from lib.gmres import *
from scipy.sparse import csr_array
from pathlib import Path
import pytest
from lib import load_data

# import scipy as sp


def test_krylov():
    A_m = np.array([[12, -5, 0, 3], [0, 4, 0, 0], [0, 0, 2, 0], [5, -2, -3, 14]])
    A_m = csr_array(A_m)
    V_m = np.array([[1, 0, 0, 0]]).T

    for i in range(3):
        v, _ = get_krylov(A_m, V_m, i)
        if v is None:
            break
        for j in range(i + 1):
            assert np.isclose(np.dot(V_m[:, j], v), 0)
        V_m = np.hstack((V_m, v[:, None]))


def test_gmres():
    A_m = np.array([[12, -5, 0, 3], [0, 4, 0, 0], [0, 0, 2, 0], [5, -2, -3, 14]])

    A_m = csr_array(A_m)
    assert check_pd(A_m)

    x_true = np.array([1, 2, 3, 4]).T

    b = A_m @ x_true

    tolerance = 1e-8
    abs_tol = tolerance * np.linalg.norm(b)

    x_sol, res_sol, _ = gmres(A_m, b, tol=tolerance, max_inner=-1)
    res = np.linalg.norm(A_m @ x_sol - b)

    assert np.isclose(res_sol, res)
    assert res < abs_tol

    x_sol, res_sol, _ = gmres(A_m, b, tol=tolerance, max_inner=4)
    res = np.linalg.norm(A_m @ x_sol - b)

    assert np.isclose(res_sol, res)
    assert res < abs_tol


def test_restarted_gmres():
    A_m = np.array([[12, -5, 0, 3], [0, 4, 0, 0], [0, 0, 2, 0], [5, -2, -3, 14]])

    A_m = csr_array(A_m)
    assert check_pd(A_m)

    x_true = np.array([1, 2, 3, 4]).T

    b = A_m @ x_true
    tolerance = 1e-8
    abs_tol = tolerance * np.linalg.norm(b)

    # sanity check
    x_sp, _ = sp.sparse.linalg.gmres(A_m, b, rtol=tolerance, restart=None)
    res = np.linalg.norm(A_m @ x_sp - b)

    assert res < abs_tol

    # actual test
    x_sol, res_sol, _ = gmres(A_m, b, tol=tolerance, max_inner=3)
    res = np.linalg.norm(A_m @ x_sol - b)

    assert res < abs_tol
    assert np.isclose(res, res_sol)


@pytest.mark.timeout(2)
def test_ILU_returns():
    A_m = np.array([[12, -5, 0, 3], [0, 4, 0, 0], [0, 0, 2, 0], [5, -2, -3, 14]])
    A_m = csr_array(A_m)
    _ = ILU(A_m)
    assert True


@pytest.mark.timeout(30)
def test_ILU_big_matrix_returns():
    A_m = load_data(Path("gmres_matrix_msr.txt"))
    _ = ILU(A_m)

    assert True
