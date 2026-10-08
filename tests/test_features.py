import numpy as np
import pandas as pd
import pytest

from src.features import build_features as bf
from src.models import ml_model

N_DAYS = 150
EVENT_DAY = 20


def _sales(seed=0, n_days=N_DAYS):
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2024-01-01", periods=n_days)
    frames = []
    for item, dept, price in [("I1", "D1", 2.0), ("I2", "D1", 4.0), ("I3", "D2", 10.0)]:
        frames.append(
            pd.DataFrame(
                {
                    "unique_id": f"{item}_S1", "ds": ds, "y": rng.poisson(3.0, n_days).astype(float),
                    "item_id": item, "dept_id": dept,
                    "event_name_1": None, "event_type_1": None,
                    "snap": (ds.day <= 10).astype(int), "sell_price": price,
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    is_event = df["ds"] == ds[EVENT_DAY]
    df.loc[is_event, ["event_name_1", "event_type_1"]] = ["Feriado", "National"]
    return df


def test_days_to_event_counts_down_and_is_capped():
    out = bf.add_calendar_features(_sales())
    one = out[out["unique_id"] == "I1_S1"].reset_index(drop=True)
    assert one.loc[EVENT_DAY, ["is_event", "days_to_event", "event_name"]].tolist() == [1, 0, "Feriado"]
    assert one.loc[EVENT_DAY - 3, "days_to_event"] == 3
    assert one.loc[EVENT_DAY - 30 if EVENT_DAY >= 30 else 0, "days_to_event"] == bf.MAX_DAYS_TO_EVENT
    # Después del último evento conocido no hay "próximo evento": se usa el tope.
    assert (one.loc[EVENT_DAY + 1 :, "days_to_event"] == bf.MAX_DAYS_TO_EVENT).all()
    assert (one.loc[EVENT_DAY + 1 :, "event_name"] == "none").all()


def test_price_features_for_constant_price():
    out = bf.add_price_features(_sales())
    np.testing.assert_allclose(out["price_rel_item"], 1.0)
    np.testing.assert_allclose(out["price_change_7"], 0.0)
    d1 = out[out["dept_id"] == "D1"].groupby("unique_id")["price_rel_dept"].first()
    assert d1.to_dict() == pytest.approx({"I1_S1": 2 / 3, "I2_S1": 4 / 3})


def test_price_features_react_to_a_markdown():
    sales = _sales()
    promo = (sales["unique_id"] == "I1_S1") & (sales["ds"] >= sales["ds"].min() + pd.Timedelta(days=100))
    sales.loc[promo, "sell_price"] = 1.0
    out = bf.add_price_features(sales)
    one = out[out["unique_id"] == "I1_S1"].reset_index(drop=True)
    assert one.loc[100, "price_change_7"] == pytest.approx(-0.5)
    assert one.loc[107, "price_change_7"] == pytest.approx(0.0)
    assert one.loc[100, "price_rel_item"] < 1.0
    assert (one.loc[:99, "price_rel_item"] == 1.0).all()


def test_price_features_do_not_look_ahead():
    sales = _sales()
    t0 = sales["ds"].min() + pd.Timedelta(days=100)
    changed = sales.copy()
    changed.loc[(changed["unique_id"] == "I1_S1") & (changed["ds"] > t0), "sell_price"] = 0.5

    cols = ["price_rel_item", "price_change_7", "price_rel_dept"]
    before = bf.add_price_features(sales).query("ds <= @t0")[cols]
    after = bf.add_price_features(changed).query("ds <= @t0")[cols]
    pd.testing.assert_frame_equal(before, after)


def test_make_model_frames_splits_history_and_future():
    sales = _sales()
    cutoff = sales["ds"].max() - pd.Timedelta(days=28)
    hist = sales[sales["ds"] <= cutoff]
    future = sales[sales["ds"] > cutoff].drop(columns="y")

    train, fut = bf.make_model_frames(hist, future)
    assert list(train.columns) == ["unique_id", "ds", "y", *bf.STATIC_FEATURES, *bf.DYNAMIC_FEATURES]
    assert list(fut.columns) == ["unique_id", "ds", *bf.DYNAMIC_FEATURES]
    assert train["ds"].max() == cutoff and fut["ds"].min() > cutoff
    assert len(fut) == 3 * 28 and not train["y"].isna().any()
    assert not fut[bf.DYNAMIC_FEATURES].isna().any().any()
    for col in ("event_name", "event_type"):
        assert train[col].cat.categories.equals(fut[col].cat.categories)

    only_train, none = bf.make_model_frames(hist)
    assert none is None and len(only_train) == len(train)


def test_lag_features_only_use_past_demand():
    """Alterar la demanda desde t0 no debe cambiar ninguna variable autorregresiva hasta t0."""
    sales = _sales()
    train, _ = bf.make_model_frames(sales)
    t0 = train["ds"].max() - pd.Timedelta(days=20)
    changed = train.copy()
    changed.loc[changed["ds"] >= t0, "y"] += 1000.0

    fcst = ml_model.make_forecaster()
    before = fcst.preprocess(train, static_features=bf.STATIC_FEATURES)
    after = fcst.preprocess(changed, static_features=bf.STATIC_FEATURES)

    auto = [c for c in before.columns if c.startswith(("lag", "rolling"))]
    assert len(auto) == len(bf.LAGS) + sum(len(v) for v in bf.lag_transforms().values())
    pd.testing.assert_frame_equal(
        before.loc[before["ds"] <= t0, auto], after.loc[after["ds"] <= t0, auto]
    )
    # Control: una semana después (lag mínimo) el cambio sí tiene que verse.
    later = t0 + pd.Timedelta(days=min(bf.LAGS))
    assert not before.loc[before["ds"] >= later, auto].equals(after.loc[after["ds"] >= later, auto])
