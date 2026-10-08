import numpy as np
import pandas as pd
import pytest

from src.models import backtest as bt

H = 28


def _data(seed=0, n_days=400, n_windows=4):
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2023-01-01", periods=n_days)
    frames = []
    for uid, dept, lam, price in [("a", "D1", 6.0, 2.0), ("b", "D1", 1.0, 8.0), ("c", "D2", 0.4, 3.0)]:
        y = rng.poisson(lam, n_days).astype(float)
        frames.append(pd.DataFrame({"unique_id": uid, "ds": ds, "y": y, "sell_price": price, "dept_id": dept}))
    sales = pd.concat(frames, ignore_index=True)

    cv = []
    for w in range(n_windows):
        cutoff = ds[n_days - 1 - (n_windows - w) * H]
        win = sales[(sales["ds"] > cutoff) & (sales["ds"] <= cutoff + pd.Timedelta(days=H))]
        cv.append(win[["unique_id", "ds", "y"]].assign(cutoff=cutoff))
    cv = pd.concat(cv, ignore_index=True)
    cv["perfecto"] = cv["y"]
    cv["mas_uno"] = cv["y"] + 1
    # Un modelo "pesado" que solo corrió las 2 últimas ventanas.
    last2 = sorted(cv["cutoff"].unique())[-2:]
    cv["pesado"] = np.where(cv["cutoff"].isin(last2), cv["y"] + 2, np.nan)
    classes = pd.DataFrame({"unique_id": ["a", "b", "c"], "abc": ["A", "B", "C"], "xyz": ["X", "Y", "Z"],
                            "segment": ["AX", "BY", "CZ"], "pattern": ["suave", "errática", "irregular"],
                            "dept_id": ["D1", "D1", "D2"]})
    return sales, cv, classes


def test_split_of_marks_last_windows_as_eval():
    cutoffs = pd.to_datetime(["2024-03-01", "2024-01-01", "2024-02-01", "2024-04-01"])
    splits = bt.split_of(cutoffs, n_eval=2)
    assert [splits[c] for c in sorted(cutoffs)] == ["calib", "calib", "eval", "eval"]


def test_evaluate_scores_each_window_with_the_models_available():
    sales, cv, _ = _data()
    wr, by_sku = bt.evaluate(sales, cv)

    total = wr[wr["level"] == "WRMSSE"]
    assert (total.loc[total["model"] == "perfecto", "value"] == 0).all()
    assert (total.loc[total["model"] == "mas_uno", "value"] > 0).all()
    # El modelo pesado solo aparece en las ventanas donde tiene pronóstico.
    assert total.loc[total["model"] == "pesado", "cutoff"].nunique() == 2
    assert total.loc[total["model"] == "mas_uno", "cutoff"].nunique() == 4
    assert set(wr["level"]) == {"total", "dept", "sku", "WRMSSE"}

    assert len(by_sku) == 3 * (4 * 2 + 2)  # SKUs × (ventanas × 2 modelos + 2 ventanas del pesado)
    assert set(by_sku["split"]) == {"calib", "eval"}


def test_segment_table_aggregates_by_volume():
    sales, cv, classes = _data()
    _, by_sku = bt.evaluate(sales, cv)
    seg = bt.segment_table(by_sku, classes)

    ev = seg[(seg["split"] == "eval") & (seg["model"] == "mas_uno")]
    total = ev[ev["grouping"] == "total"].iloc[0]
    assert total["n_skus"] == 3 and total["mae"] == pytest.approx(1.0)
    # Error de +1 por día: sesgo = WAPE = días-SKU / demanda.
    eval_cut = sorted(cv["cutoff"].unique())[-bt.config.N_WINDOWS :]
    demand = cv.loc[cv["cutoff"].isin(eval_cut), "y"].sum()
    n_rows = cv["cutoff"].isin(eval_cut).sum()
    assert total["demand"] == demand
    assert total["wape"] == pytest.approx(n_rows / demand)
    assert total["bias"] == pytest.approx(n_rows / demand)

    by_abc = ev[ev["grouping"] == "abc"].set_index("group")
    assert set(by_abc.index) == {"A", "B", "C"}
    assert by_abc["demand"].sum() == demand
    assert by_abc.loc["A", "wape"] < by_abc.loc["C", "wape"]  # mismo error, más volumen


def test_interval_coverage_is_measured_on_eval_windows_only():
    sales, cv, classes = _data(n_windows=6)
    cv = cv.rename(columns={"mas_uno": "LightGBM"})
    cov = bt.interval_coverage(cv, classes)

    total = cov[cov["grouping"] == "total"].set_index("level")
    assert total.loc[80, "n_obs"] == 3 * 3 * H  # 3 SKUs × 3 ventanas de evaluación
    assert total.loc[95, "coverage"] >= total.loc[80, "coverage"]
    assert total.loc[95, "mean_width"] >= total.loc[80, "mean_width"]
    assert ((cov["coverage"] >= 0) & (cov["coverage"] <= 1)).all()
    assert set(cov["grouping"]) == {"total", "abc", "xyz", "pattern"}

    # Cambiar la demanda de las ventanas de evaluación no puede mover los márgenes:
    # con el mismo pronóstico, el ancho del intervalo queda igual.
    eval_cut = sorted(cv["cutoff"].unique())[-3:]
    shifted = cv.copy()
    shifted.loc[shifted["cutoff"].isin(eval_cut), "y"] += 50
    cov2 = bt.interval_coverage(shifted, classes)
    assert cov2["coverage"].max() == 0  # la demanda quedó fuera de todos los intervalos
