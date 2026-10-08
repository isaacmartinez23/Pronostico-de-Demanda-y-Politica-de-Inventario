import numpy as np
import pandas as pd
import pytest

from src.inventory import policy as pol

# ---------------------------------------------------------------------------- fórmulas


def test_z_score_known_values():
    assert pol.z_score(0.95) == pytest.approx(1.6449, abs=1e-4)
    assert pol.z_score(0.50) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_z_score_rejects_out_of_range(bad):
    with pytest.raises(ValueError):
        pol.z_score(bad)


def test_safety_stock_formula():
    ss = pol.safety_stock(sigma_error=2.0, lead_time=9, service_level=0.95)
    assert float(ss) == pytest.approx(1.6449 * 2.0 * 3.0, abs=1e-3)


def test_safety_stock_grows_with_lead_time_and_service_level():
    sigma = np.array([0.5, 2.0, 10.0])
    by_lt = [pol.safety_stock(sigma, lt, 0.95) for lt in (3, 7, 14)]
    by_sl = [pol.safety_stock(sigma, 7, sl) for sl in (0.90, 0.95, 0.99)]
    assert (by_lt[0] < by_lt[1]).all() and (by_lt[1] < by_lt[2]).all()
    assert (by_sl[0] < by_sl[1]).all() and (by_sl[1] < by_sl[2]).all()


def test_safety_stock_never_negative():
    # Con nivel de servicio < 50% z es negativo: el colchón se trunca en cero.
    assert float(pol.safety_stock(3.0, 7, 0.30)) == 0.0
    assert float(pol.safety_stock(0.0, 7, 0.99)) == 0.0
    with pytest.raises(ValueError):
        pol.safety_stock(1.0, -1, 0.95)


def test_reorder_point_is_lead_time_demand_plus_safety_stock():
    np.testing.assert_allclose(pol.reorder_point([10.0, 0.0], [4.0, 1.5]), [14.0, 1.5])


def test_eoq_known_value_and_degenerate_cases():
    assert float(pol.eoq(1000, order_cost=10, holding_cost=2)) == pytest.approx(100.0)
    np.testing.assert_array_equal(pol.eoq([0.0, 100.0], 10, [2.0, 0.0]), [0.0, 0.0])


def test_lot_size_is_bounded_by_days_of_cover():
    costs = pol.CostParams(max_cover_days=14)
    daily = np.array([0.0, 0.02, 1.0, 50.0])
    lot = pol.lot_size(daily, price=np.array([3.0, 3.0, 3.0, 3.0]), costs=costs)
    assert (lot >= 1).all() and (lot == np.round(lot)).all()
    assert (lot <= np.maximum(np.ceil(daily * costs.max_cover_days), 1)).all()
    assert (lot >= np.ceil(daily)).all()


def test_order_quantity_reorders_up_to_target_only_at_or_below_rop():
    q = pol.order_quantity(inventory_position=[5, 10, 11], rop=[10, 10, 10], lot=[20, 20, 20])
    np.testing.assert_array_equal(q, [25, 20, 0])


# -------------------------------------------------------------------------- simulación


def _random_case(seed=0, n_days=120, n=6):
    rng = np.random.default_rng(seed)
    demand = rng.poisson(rng.uniform(0.2, 6.0, size=n), size=(n_days, n)).astype(float)
    rop = np.tile(rng.integers(2, 30, size=n).astype(float), (n_days, 1))
    lot = np.tile(rng.integers(1, 25, size=n).astype(float), (n_days, 1))
    return demand, rop, lot


@pytest.mark.parametrize("lead_time", [1, 3, 7, 14])
def test_simulation_conserves_units(lead_time):
    demand, rop, lot = _random_case()
    sim = pol.simulate_policy(demand, rop, lot, lead_time)

    np.testing.assert_allclose(sim.sales + sim.lost, demand)
    assert (sim.on_hand >= 0).all() and (sim.lost >= 0).all() and (sim.orders >= 0).all()

    # Balance diario: lo pedido en t llega exactamente en t + lead_time.
    arrivals = np.zeros_like(sim.orders)
    arrivals[lead_time:] = sim.orders[:-lead_time]
    start = np.vstack([rop[0] + lot[0], sim.on_hand[:-1]])
    np.testing.assert_allclose(sim.on_hand, start + arrivals - sim.sales)


def test_simulation_never_loses_sales_with_stock_on_hand():
    demand, rop, lot = _random_case(seed=1)
    sim = pol.simulate_policy(demand, rop, lot, lead_time=5)
    assert (sim.on_hand[sim.lost > 0] == 0).all()


def test_simulation_without_demand_places_no_orders():
    zeros = np.zeros((30, 2))
    sim = pol.simulate_policy(zeros, zeros + 5, zeros + 3, lead_time=3)
    assert sim.orders.sum() == 0 and sim.lost.sum() == 0
    np.testing.assert_array_equal(sim.on_hand, zeros + 8)


def test_simulation_higher_reorder_point_improves_fill_rate():
    demand, rop, lot = _random_case(seed=2)
    low = pol.simulate_policy(demand, rop * 0.2, lot, lead_time=7)
    high = pol.simulate_policy(demand, rop * 5.0, lot, lead_time=7)
    assert high.lost.sum() <= low.lost.sum()
    assert high.on_hand.mean() > low.on_hand.mean()


def test_simulation_rejects_zero_lead_time():
    demand, rop, lot = _random_case()
    with pytest.raises(ValueError):
        pol.simulate_policy(demand, rop, lot, lead_time=0)


# ------------------------------------------------------------------ insumos de política


def test_extend_forecast_repeats_last_two_weeks():
    f = np.arange(28, dtype=float).reshape(28, 1)
    out = pol.extend_forecast(f, 16)
    assert out.shape == (44, 1)
    np.testing.assert_array_equal(out[28:42, 0], np.arange(14, 28))
    np.testing.assert_array_equal(out[42:, 0], [14, 15])
    assert pol.extend_forecast(f, 0) is f


def _cubes(seed=0, n_win=6, h=28, n=4):
    rng = np.random.default_rng(seed)
    y = rng.poisson(3.0, size=(n_win, h, n)).astype(float)
    f = np.full((n_win, h, n), 3.0) + rng.normal(0, 0.3, size=(n_win, h, n))
    return y, f


def test_forecast_basis_shapes_and_lead_time_demand():
    y, f = _cubes()
    basis = pol.forecast_basis(y, f, lead_time=7, n_eval=3)
    assert basis.mu_lead.shape == basis.sigma.shape == basis.daily.shape == (3 * 28 + 1, 4)
    # En el primer punto de decisión, la demanda esperada son los primeros 7 días del pronóstico.
    np.testing.assert_allclose(basis.mu_lead[0], f[3, :7].sum(0))
    np.testing.assert_allclose(basis.mu_lead[10], f[3, 10:17].sum(0))


def test_forecast_basis_sigma_does_not_use_the_simulated_period():
    """σ de cada ventana sale solo de ventanas anteriores: cambiar el futuro no la altera."""
    y, f = _cubes()
    base = pol.forecast_basis(y, f, lead_time=7, n_eval=3)

    y_future = y.copy()
    y_future[-1] += 100.0  # la última ventana simulada
    after = pol.forecast_basis(y_future, f, lead_time=7, n_eval=3)
    np.testing.assert_allclose(after.sigma, base.sigma)

    y_past = y.copy()
    y_past[0, ::2] += 100.0  # una ventana de calibración sí debe influir
    assert not np.allclose(pol.forecast_basis(y_past, f, 7, n_eval=3).sigma, base.sigma)


def test_classic_basis_uses_only_history_before_each_cutoff():
    rng = np.random.default_rng(3)
    hist = rng.poisson(4.0, size=(pol.CLASSIC_WINDOW + 3 * 28, 3)).astype(float)
    base = pol.classic_basis(hist, lead_time=7, n_eval=3, h=28)
    np.testing.assert_allclose(base.mu_lead[0], hist[: pol.CLASSIC_WINDOW].mean(0) * 7)

    changed = hist.copy()
    changed[-28:] += 50.0  # la última ventana simulada
    after = pol.classic_basis(changed, lead_time=7, n_eval=3, h=28)
    np.testing.assert_allclose(after.mu_lead, base.mu_lead)
    np.testing.assert_allclose(after.sigma, base.sigma)


def test_cv_cube_orders_by_date_and_sku():
    ds = pd.date_range("2024-01-01", periods=56)
    cv = pd.DataFrame(
        [(u, d, i + 100 * j) for j, u in enumerate(["b", "a"]) for i, d in enumerate(ds)],
        columns=["unique_id", "ds", "y"],
    ).sample(frac=1, random_state=0)
    cube = pol.cv_cube(cv, "y", ids=["a", "b"], h=28)
    assert cube.shape == (2, 28, 2)
    np.testing.assert_array_equal(cube[1, 0], [128, 28])


def test_run_policy_reaches_full_service_with_perfect_forecast():
    y, _ = _cubes(seed=5)
    basis = pol.forecast_basis(y, y.copy(), lead_time=3, n_eval=3)
    demand = y[-3:].reshape(84, 4)
    ids = [f"sku{i}" for i in range(4)]
    out = pol.run_policy(basis, demand, np.full(4, 2.5), ids, lead_time=3, service_level=0.95)
    assert list(out["unique_id"]) == ids
    assert (out["fill_rate"] > 0.99).all()
    np.testing.assert_allclose(out["sales"] + out["lost"], out["demand"])


# ------------------------------------------------------------- variantes de safety stock


def test_rolling_sums_and_lead_time_errors():
    x = np.arange(1, 7, dtype=float).reshape(6, 1)
    np.testing.assert_array_equal(pol.rolling_sums(x, 3)[:, 0], [6, 9, 12, 15])
    np.testing.assert_array_equal(pol.rolling_sums(x, 1), x)

    resid = np.stack([x, 10 * x])  # 2 ventanas: las sumas no cruzan de una ventana a otra
    out = pol.lead_time_errors(resid, 3)
    np.testing.assert_array_equal(out[:, 0], [6, 9, 12, 15, 60, 90, 120, 150])


def test_empirical_quantile_is_conservative_and_per_column():
    rng = np.random.default_rng(0)
    samples = np.column_stack([rng.normal(0, 1, 200), rng.normal(0, 5, 200)])
    q = pol.empirical_quantile(samples, 0.95)
    assert q.shape == (2,) and q[1] > 3 * q[0]
    assert (q >= np.quantile(samples, 0.95, axis=0)).all()
    np.testing.assert_array_equal(pol.empirical_quantile(samples[:5], 0.99), samples[:5].max(0))


def test_sigma_lead_matches_sqrt_rule_only_for_independent_errors():
    """Con errores independientes σ(L) ≈ σ·√L; con errores correlacionados la regla se queda corta."""
    rng = np.random.default_rng(0)
    n_win, h, n, lead = 6, 28, 400, 7
    f = np.full((n_win, h, n), 10.0)

    iid = pol.forecast_basis(f + rng.normal(0, 2, f.shape), f, lead)
    ratio = iid.sigma_lead[0] / (iid.sigma[0] * np.sqrt(lead))
    assert np.median(ratio) == pytest.approx(1.0, abs=0.1)

    # Error persistente: el mismo desvío durante toda la ventana (como un sesgo de nivel).
    level = rng.normal(0, 2, (n_win, 1, n))
    corr = pol.forecast_basis(f + level + rng.normal(0, 0.5, f.shape), f, lead)
    ratio = corr.sigma_lead[0] / (corr.sigma[0] * np.sqrt(lead))
    assert np.median(ratio) > 1.5


def test_safety_stock_methods():
    y, f = _cubes(seed=7)
    basis = pol.forecast_basis(y, f, lead_time=7, n_eval=3)
    assert len(basis.err_lead) == 3 and basis.window.tolist() == [0] * 28 + [1] * 28 + [2] * 29
    # Muestras disponibles: (ventanas previas) × (h - L + 1) → crecen con cada ventana.
    assert [e.shape[0] for e in basis.err_lead] == [3 * 22, 4 * 22, 5 * 22]

    ss = {m: pol.safety_stock_for(basis, 7, 0.95, m) for m in pol.SS_METHODS}
    assert all(v.shape == basis.mu_lead.shape for v in ss.values())
    np.testing.assert_allclose(ss["sqrt"], pol.safety_stock(basis.sigma, 7, 0.95))
    np.testing.assert_allclose(ss["acumulado"], pol.z_score(0.95) * basis.sigma_lead)
    for m in pol.SS_METHODS:
        assert (pol.safety_stock_for(basis, 7, 0.99, m) >= ss[m]).all()
    with pytest.raises(ValueError):
        pol.safety_stock_for(basis, 7, 0.95, "otro")


def test_empirical_safety_stock_corrects_forecast_bias():
    """Si el pronóstico sobreestima siempre, el cuantil empírico baja el punto de reorden."""
    y, _ = _cubes(seed=8)
    basis = pol.forecast_basis(y, y + 5.0, lead_time=7, n_eval=3)
    assert (pol.safety_stock_for(basis, 7, 0.95, "empirico") < 0).all()
    assert (pol.safety_stock_for(basis, 7, 0.95, "acumulado") >= 0).all()

    demand = y[-3:].reshape(84, 4)
    ids = [f"sku{i}" for i in range(4)]
    args = (basis, demand, np.full(4, 2.0), ids, 7, 0.95)
    emp = pol.run_policy(*args, ss_method="empirico")
    sqrt = pol.run_policy(*args, ss_method="sqrt")
    assert (emp["rop"] < sqrt["rop"]).all() and (emp["rop"] >= 0).all()
    assert (emp["avg_on_hand"] < sqrt["avg_on_hand"]).all()


def test_classic_basis_lead_time_errors_are_centered_demand():
    rng = np.random.default_rng(4)
    hist = rng.poisson(4.0, size=(pol.CLASSIC_WINDOW + 3 * 28, 2)).astype(float)
    basis = pol.classic_basis(hist, lead_time=7, n_eval=3, h=28)
    past = hist[: pol.CLASSIC_WINDOW]
    expected = pol.rolling_sums(past, 7) - 7 * past.mean(0)
    np.testing.assert_allclose(basis.err_lead[0], expected)
    assert basis.err_lead[0].shape[0] == pol.CLASSIC_WINDOW - 7 + 1


def test_cycle_service_counts_replenishment_cycles_with_stockouts():
    demand, rop, lot = _random_case(seed=3)
    sim = pol.simulate_policy(demand, rop, lot, lead_time=5)
    assert (sim.stockout_cycles <= sim.cycles).all()
    assert (sim.cycles <= (sim.orders > 0).sum(0)).all()  # solo cuentan los pedidos ya recibidos

    safe = pol.simulate_policy(demand, rop * 20, lot, lead_time=5)
    assert safe.lost.sum() == 0 and safe.stockout_cycles.sum() == 0

    # Un SKU sin inventario ni reposición suficiente quiebra en todos sus ciclos.
    tight = pol.simulate_policy(np.full((60, 1), 10.0), np.zeros((60, 1)), np.ones((60, 1)), 3)
    assert tight.cycles[0] > 0 and tight.stockout_cycles[0] == tight.cycles[0]


# ---------------------------------------------------------------------------- agregados


def test_aggregate_weights_fill_rate_by_value():
    by_sku = pd.DataFrame(
        {
            "demand": [100.0, 100.0], "sales": [100.0, 50.0], "price": [9.0, 1.0],
            "fill_rate": [1.0, 0.5], "stockout_days": [0.0, 0.2], "avg_on_hand": [10.0, 5.0],
            "avg_inv_value": [63.0, 3.5], "daily_demand": [2.0, 2.0], "n_orders": [4, 2],
            "cycles": [4.0, 2.0], "stockout_cycles": [0.0, 1.0],
        }
    )
    agg = pol.aggregate(by_sku, target=0.95)
    assert agg["cycle_service"] == pytest.approx(5 / 6)
    assert agg["fill_rate"] == pytest.approx(950 / 1000)
    assert agg["fill_rate_units"] == pytest.approx(0.75)
    assert agg["skus_meeting_target"] == pytest.approx(0.5)
    assert agg["days_of_supply"] == pytest.approx(15 / 4)


def test_inventory_at_fill_rate_interpolates_and_flags_unreachable_targets():
    frontier = pd.DataFrame(
        {
            "policy": "p", "lead_time": 7, "group": "A",
            "fill_rate": [0.80, 0.90, 0.98], "avg_inv_value": [100.0, 200.0, 400.0],
        }
    )
    assert pol.inventory_at_fill_rate(frontier, 0.94)["avg_inv_value"][0] == pytest.approx(300.0)
    assert np.isnan(pol.inventory_at_fill_rate(frontier, 0.995)["avg_inv_value"][0])


def test_unclipped_safety_stock_extends_the_frontier_below_zero_buffer():
    y, f = _cubes(seed=9)
    basis = pol.forecast_basis(y, f, lead_time=7, n_eval=3)
    for method in ("sqrt", "acumulado"):
        assert (pol.safety_stock_for(basis, 7, 0.10, method) == 0).all()
        assert (pol.safety_stock_for(basis, 7, 0.10, method, clip=False) < 0).all()

    demand = y[-3:].reshape(84, 4)
    args = (basis, demand, np.full(4, 2.0), [f"sku{i}" for i in range(4)], 7)
    low = pol.run_policy(*args, 0.01, clip_ss=False)
    mid = pol.run_policy(*args, 0.50, clip_ss=False)
    assert (low["rop"] <= mid["rop"]).all() and (low["rop"] >= 0).all()
    assert low["avg_on_hand"].sum() < mid["avg_on_hand"].sum()
