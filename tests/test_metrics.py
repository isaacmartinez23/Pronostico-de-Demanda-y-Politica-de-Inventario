import numpy as np
import pandas as pd
import pytest

from src.utils import metrics as m


def test_point_metrics():
    y, yhat = np.array([0.0, 2.0, 4.0]), np.array([1.0, 2.0, 2.0])
    assert m.mae(y, yhat) == pytest.approx(1.0)
    assert m.rmse(y, yhat) == pytest.approx(np.sqrt(5 / 3))
    assert m.wape(y, yhat) == pytest.approx(3 / 6)


def test_bias_sign():
    y = np.array([10.0, 10.0])
    assert m.bias(y, y + 1) > 0  # sobre-pronostica
    assert m.bias(y, y - 1) < 0
    assert np.isnan(m.bias(np.zeros(2), np.ones(2)))
    assert np.isnan(m.wape(np.zeros(2), np.ones(2)))


def test_naive_scale_ignores_leading_zeros():
    # Tras la primera venta: [2, 4, 1] → diferencias [2, -3] → media de cuadrados 6.5
    assert m.naive_scale(np.array([0, 0, 2, 4, 1])) == pytest.approx(6.5)
    assert m.naive_scale(np.array([2, 4, 1])) == pytest.approx(6.5)


@pytest.mark.parametrize("series", [[0, 0, 0], [0, 0, 5], [3, 3, 3]])
def test_naive_scale_undefined_cases(series):
    assert np.isnan(m.naive_scale(np.array(series)))


def test_rmsse():
    y = np.array([1.0, 3.0])
    assert m.rmsse(y, y, scale=4.0) == 0.0
    assert m.rmsse(y, y + 2, scale=4.0) == pytest.approx(1.0)
    assert np.isnan(m.rmsse(y, y, scale=np.nan))


def _frames(seed=0, n_train=120, h=28):
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2024-01-01", periods=n_train + h)
    rows = []
    for uid, dept, lam, price in [("a", "D1", 5.0, 2.0), ("b", "D1", 1.0, 10.0), ("c", "D2", 0.3, 4.0)]:
        y = rng.poisson(lam, size=len(ds)).astype(float)
        rows.append(pd.DataFrame({"unique_id": uid, "dept_id": dept, "ds": ds, "y": y, "sell_price": price}))
    full = pd.concat(rows, ignore_index=True)
    cutoff = ds[n_train - 1]
    return full[full["ds"] <= cutoff], full[full["ds"] > cutoff].drop(columns="sell_price")


def test_wrmsse_is_zero_for_a_perfect_forecast_and_positive_otherwise():
    train, cv = _frames()
    cv = cv.assign(perfecto=cv["y"], cero=0.0)
    out = m.wrmsse(train, cv, ["perfecto", "cero"])
    assert list(out.index) == ["total", "dept", "sku", "WRMSSE"]
    assert (out["perfecto"] == 0).all()
    assert (out["cero"] > 0).all()
    assert out.loc["WRMSSE", "cero"] == pytest.approx(out.loc[["total", "dept", "sku"], "cero"].mean())


def test_wrmsse_weights_come_from_last_28_days_of_revenue():
    train, cv = _frames()
    tab = m.rmsse_by_level(train, cv.assign(cero=0.0), ["cero"], ["unique_id"])
    last28 = train[train["ds"] > train["ds"].max() - pd.Timedelta(days=28)]
    expected = (last28["y"] * last28["sell_price"]).groupby(last28["unique_id"]).sum()
    pd.testing.assert_series_equal(tab["weight"], expected, check_names=False)


def test_per_series_metrics():
    train, cv = _frames()
    cv = cv.assign(mas_uno=cv["y"] + 1)
    out = m.per_series_metrics(train, cv, ["mas_uno"]).set_index("unique_id")
    assert (out["mae"] == 1).all()
    assert (out["abs_err"] == 28).all() and (out["err"] == 28).all()
    np.testing.assert_allclose(out["demand"], cv.groupby("unique_id")["y"].sum())
    assert (out["rmsse"] > 0).all()
