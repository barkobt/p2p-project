import pandas as pd

from src.drift import compute_psi


def test_compute_psi_is_non_negative():
    ref = pd.Series([10, 20, 30, 40, 50, 60, 70, 80])
    cur = pd.Series([11, 19, 31, 39, 52, 58, 73, 81])
    psi = compute_psi(ref, cur)
    assert psi >= 0.0


def test_compute_psi_detects_shift():
    ref = pd.Series([10, 20, 30, 40, 50, 60, 70, 80])
    cur = pd.Series([100, 110, 120, 130, 140, 150, 160, 170])
    psi = compute_psi(ref, cur)
    assert psi > 0.1
