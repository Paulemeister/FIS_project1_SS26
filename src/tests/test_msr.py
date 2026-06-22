from lib import load_data
from pathlib import Path
import numpy as np


def test_non_sym_msr_load():
    non_sym_path = Path("./msr_test/msr_test_non_symmetric.txt")
    non_sym_loaded = load_data(non_sym_path)
    non_sym_ref = np.array(
        [[12, -5, 0, 3], [0, 4, 0, 0], [0, 0, 2, 0], [5, -2, -3, 14]]
    )
    assert np.allclose(non_sym_loaded, non_sym_ref)


def test_sym_msr_load():
    sym_path = Path("./msr_test/msr_test_symmetric.txt")
    sym_loaded = load_data(sym_path)
    sym_ref = np.array([[4, 0, 2, 0], [0, 8, 0, 4], [2, 0, 5, -1], [0, 4, -1, 8]])
    assert np.allclose(sym_loaded, sym_ref)
