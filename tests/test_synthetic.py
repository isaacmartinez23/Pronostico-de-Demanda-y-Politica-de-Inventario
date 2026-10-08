import numpy as np

from src import config
from src.data import synthetic


def test_synthetic_m5_has_the_raw_m5_layout():
    raw = synthetic.generate(n_items=8, n_days=420, seed=1)
    assert set(raw) == set(config.RAW_FILES)
    cal, sales, prices = (raw[f] for f in ("calendar.csv", "sales_train_evaluation.csv", "sell_prices.csv"))

    # El calendario cubre la historia y el horizonte de pronóstico.
    assert len(cal) == 420 + config.HORIZON
    assert {"date", "wm_yr_wk", "d", "event_name_1", "event_type_1", "snap_CA"} <= set(cal.columns)
    assert cal["event_name_1"].notna().sum() > 0

    day_cols = [c for c in sales.columns if c.startswith("d_")]
    assert day_cols == list(cal["d"][:420])
    assert list(sales.columns[:6]) == ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    assert (sales[day_cols].to_numpy() >= 0).all()
    assert (sales["store_id"] == config.STORE_ID).all() and (sales["cat_id"] == config.CAT_ID).all()

    assert set(prices["item_id"]) == set(sales["item_id"])
    assert set(prices["wm_yr_wk"]) <= set(cal["wm_yr_wk"])
    assert (prices["sell_price"] > 0).all()


def test_synthetic_m5_is_reproducible_and_mixes_demand_patterns():
    a = synthetic.generate(n_items=12, n_days=420, seed=7)["sales_train_evaluation.csv"]
    b = synthetic.generate(n_items=12, n_days=420, seed=7)["sales_train_evaluation.csv"]
    assert a.equals(b)

    zero_share = (a.filter(like="d_").to_numpy() == 0).mean(axis=1)
    assert zero_share.min() < 0.05 and zero_share.max() > 0.5
    # El último SKU entra tarde al surtido: sin ventas antes de su fecha de alta.
    late = a.filter(like="d_").iloc[-1].to_numpy()
    assert late[: 420 - 210].sum() == 0 and np.count_nonzero(late) > 0
