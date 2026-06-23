from lib import *
import scipy as sp


def test_krylov():
    A_m = np.array([[12, -5, 0, 3], [0, 4, 0, 0], [0, 0, 2, 0], [5, -2, -3, 14]])
    V_m = np.array([[1, 0, 0, 0]]).T

    for i in range(3):
        v, _ = get_krylov(A_m, V_m, i)
        if v is None:
            break
        for j in range(i + 1):
            assert np.isclose(np.dot(V_m[:, j], v), 0)
        V_m = np.hstack((V_m, v[:, None]))


def test_gmrs():
    # A_m = np.array([[12, -5, 0, 3], [0, 4, 0, 0], [0, 0, 2, 0], [5, -2, -3, 14]])
    A_m = np.array(
        [
            [0.49681415, -0.13826430, 0.64768854, 1.52302986],
            [-0.23415337, -0.23403696, 1.57921282, 0.76743473],
            [-0.46947439, 0.54256004, -0.46331769, -0.46572975],
            [0.24196227, -1.91328024, -1.72491783, -0.56218753],
        ]
    )
    x_true = np.array([1, 2, 3, 4]).T

    b = A_m @ x_true

    tolerance = 1e-8
    abs_tol = tolerance * np.linalg.norm(b)

    x_sol, res_sol, _ = gmrs(A_m, b, tol=tolerance, max_inner=-1)
    res = np.linalg.norm(A_m @ x_sol - b)

    assert np.isclose(res_sol, res)
    assert res < abs_tol

    x_sol, res_sol, _ = gmrs(A_m, b, tol=tolerance, max_inner=4)
    res = np.linalg.norm(A_m @ x_sol - b)

    assert np.isclose(res_sol, res)
    assert res < abs_tol


def test_restarted_gmrs():
    # FIXME: This test does the same as the normal one
    # A_m = np.array([[12, -5, 0, 3], [0, 4, 0, 0], [0, 0, 2, 0], [5, -2, -3, 14]])
    A_m = np.array(
        [
            [0.49681415, -0.13826430, 0.64768854, 1.52302986],
            [-0.23415337, -0.23403696, 1.57921282, 0.76743473],
            [-0.46947439, 0.54256004, -0.46331769, -0.46572975],
            [0.24196227, -1.91328024, -1.72491783, -0.56218753],
        ]
    )
    x_true = np.array([1, 2, 3, 4]).T

    b = A_m @ x_true
    tolerance = 1e-1
    abs_tol = tolerance * np.linalg.norm(b)

    # sanity check
    x_sp, _ = sp.sparse.linalg.gmres(A_m, b, rtol=tolerance, restart=4)
    res = np.linalg.norm(A_m @ x_sp - b)

    assert res < abs_tol

    # actual test
    x_sol, res_sol, _ = gmrs(A_m, b, tol=tolerance, max_inner=4)
    res = np.linalg.norm(A_m @ x_sol - b)

    assert res < abs_tol
    assert np.isclose(res, res_sol)
