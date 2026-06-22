from lib import *


def test_krylov():
    A_m = np.array([[12, -5, 0, 3], [0, 4, 0, 0], [0, 0, 2, 0], [5, -2, -3, 14]])
    V_m = np.array([[1, 0, 0, 0]]).T

    for _ in range(3):
        _, k = V_m.shape
        v, _ = get_krylov(A_m, V_m)
        if v is None:
            continue
        for j in range(k):
            assert np.dot(V_m[:, j], v) == 0
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
    x_sol = gmrs(A_m, b, tol=tolerance)
    res = A_m @ x_sol - b

    assert np.linalg.norm(res) < tolerance
