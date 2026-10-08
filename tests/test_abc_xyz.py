import numpy as np
import pandas as pd

from src.inventory import abc_xyz as ax


def test_abc_cuts_and_boundary_sku_goes_to_higher_class():
    # 'a' aporta 50%, 'b' cruza el 80% (50→85) y queda en A; 'c' cruza el 95% y queda en B.
    revenue = pd.Series({"d": 3.0, "a": 50.0, "c": 12.0, "b": 35.0})
    cls = ax.abc_classify(revenue)
    assert cls.to_dict() == {"d": "C", "a": "A", "c": "B", "b": "A"}
    assert list(cls.index) == list(revenue.index)


def test_abc_without_revenue_does_not_fail():
    cls = ax.abc_classify(pd.Series({"a": 0.0, "b": 0.0}))
    assert set(cls) <= {"A", "B", "C"}


def test_xyz_cuts_and_undefined_cv():
    cv = pd.Series({"x": 0.5, "y": 0.51, "y2": 1.0, "z": 1.01, "sin_ventas": np.nan})
    assert ax.xyz_classify(cv).to_dict() == {
        "x": "X", "y": "Y", "y2": "Y", "z": "Z", "sin_ventas": "Z",
    }


def test_demand_pattern_quadrants():
    adi = pd.Series([1.0, 1.0, 2.0, 2.0])
    cv2 = pd.Series([0.2, 0.8, 0.2, 0.8])
    assert list(ax.demand_pattern(adi, cv2)) == ["suave", "errática", "intermitente", "irregular"]


def _sales(end="2024-12-31", days=500):
    ds = pd.date_range(end=end, periods=days)
    steady = pd.DataFrame({"unique_id": "steady", "ds": ds, "y": 10.0, "sell_price": 2.0})
    sparse_y = np.where(np.arange(days) % 30 == 0, 6.0, 0.0)
    sparse = pd.DataFrame({"unique_id": "sparse", "ds": ds, "y": sparse_y, "sell_price": 1.0})
    dead = pd.DataFrame({"unique_id": "dead", "ds": ds, "y": 0.0, "sell_price": 5.0})
    return pd.concat([steady, sparse, dead], ignore_index=True)


def test_classify_end_to_end():
    tab = ax.classify(_sales()).set_index("unique_id")
    assert tab.loc["steady", ["abc", "xyz", "pattern"]].tolist() == ["A", "X", "suave"]
    assert tab.loc["sparse", "pattern"] == "intermitente"
    assert tab.loc["sparse", "xyz"] == "Z"
    assert tab.loc["dead", "xyz"] == "Z" and tab.loc["dead", "units"] == 0
    assert np.isinf(tab.loc["dead", "adi"])
    assert tab["revenue_share"].sum() == 1.0
    assert tab.loc["steady", "units"] == 3650  # exactamente 365 días


def test_classify_ignores_data_after_end():
    sales = _sales()
    end = sales["ds"].max() - pd.Timedelta(days=84)
    base = ax.classify(sales, end=end)

    future = sales.copy()
    future.loc[future["ds"] > end, "y"] *= 100
    pd.testing.assert_frame_equal(ax.classify(future, end=end), base)
