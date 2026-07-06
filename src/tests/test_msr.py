from lib import load_data
from lib.misc import check_pd, check_spd
from pathlib import Path
import numpy as np


def test_non_sym_msr_load():
    non_sym_path = Path("./msr_test/msr_test_non_symmetric.txt")
    non_sym_loaded = load_data(non_sym_path).toarray()
    non_sym_ref = np.array(
        [[12, -5, 0, 3], [0, 4, 0, 0], [0, 0, 2, 0], [5, -2, -3, 14]]
    )
    assert np.allclose(non_sym_loaded, non_sym_ref)


def test_sym_msr_load():
    sym_path = Path("./msr_test/msr_test_symmetric.txt")
    sym_loaded = load_data(sym_path).toarray()
    sym_ref = np.array([[4, 0, 2, 0], [0, 8, 0, 4], [2, 0, 5, -1], [0, 4, -1, 8]])
    assert np.allclose(sym_loaded, sym_ref)


# this fails: msr storage is not respecting s header.
def test_load_type_makes_a_difference():
    paths = [Path(i) for i in ["./cg_matrix_msr_1.txt", "./cg_matrix_msr_2.txt"]]
    for path in paths:
        sym_load_data = load_data(path).toarray()
        non_sym_load_data = load_data(path, stor_type="not s").toarray()

        assert not np.allclose(sym_load_data, non_sym_load_data)


def test_gmrs_pd():
    A = load_data(Path("gmres_matrix_msr.txt"))

    assert check_pd(A)


def test_cg_spd():
    A_m = load_data(Path("cg_matrix_msr_1.txt"))
    assert check_spd(A_m)

    A_m = load_data(Path("cg_matrix_msr_2.txt"))
    assert check_spd(A_m)
