import numpy as np
import pandas as pd
import pytest

from src.models import intervals as iv


def test_conformal_quantile_finite_sample_correction():
    x = np.arange(1, 20, dtype=float)  # n = 19
    # Superior al 90%: ceil(20·0.9)/19 = 18/19 → más conservador que el cuantil simple.
    assert iv._conformal_quantile(x, 0.9) == pytest.approx(np.quantile(x, 18 / 19))
    assert iv._conformal_quantile(x, 0.9) >= np.quantile(x, 0.9)
    assert iv._conformal_quantile(x, 0.1) <= np.quantile(x, 0.1)
    # Con pocos datos el nivel se satura en los extremos de la muestra.
    assert iv._conformal_quantile(np.array([1.0, 2.0, 3.0]), 0.975) == 3.0
    assert iv._conformal_quantile(np.array([1.0, 2.0, 3.0]), 0.025) == 1.0


def _cv(seed=0, n=400):
    rng = np.random.default_rng(seed)
    frames = []
    for uid, scale in [("a", 1.0), ("b", 5.0)]:
        frames.append(pd.DataFrame({"unique_id": uid, "y": 50 + rng.normal(0, scale, n), "m": 50.0}))
    return pd.concat(frames, ignore_index=True)


def test_residual_quantiles_are_per_series_and_ordered():
    q = iv.residual_quantiles(_cv(), "m")
    assert list(q.columns) == ["lo-80", "hi-80", "lo-95", "hi-95"]
    assert (q["lo-95"] <= q["lo-80"]).all() and (q["hi-80"] <= q["hi-95"]).all()
    assert (q["lo-80"] < 0).all() and (q["hi-80"] > 0).all()
    assert q.loc["b", "hi-95"] > 3 * q.loc["a", "hi-95"]  # la serie ruidosa tiene intervalo más ancho


def test_intervals_cover_new_data_at_nominal_level():
    q = iv.residual_quantiles(_cv(seed=0), "m")
    fresh = iv.add_intervals(_cv(seed=1), q, "m")
    cov = iv.coverage(fresh, "m")
    assert cov[80] == pytest.approx(0.80, abs=0.05)
    assert cov[95] == pytest.approx(0.95, abs=0.03)


def test_intervals_are_clipped_at_zero_and_contain_the_forecast():
    q = pd.DataFrame({"lo-80": [-5.0], "hi-80": [4.0], "lo-95": [-9.0], "hi-95": [7.0]}, index=["a"])
    fc = pd.DataFrame({"unique_id": ["a", "a"], "m": [1.0, 20.0]})
    out = iv.add_intervals(fc, q, "m")
    assert out["m-lo-80"].tolist() == [0.0, 15.0]
    assert out["m-hi-95"].tolist() == [8.0, 27.0]
    assert (out["m-lo-95"] <= out["m"]).all() and (out["m"] <= out["m-hi-95"]).all()
