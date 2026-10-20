"""Each function must give the same answer as the course page's calculator (page defaults)."""

import math
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import valuation_ledger as L


def test_present_value():  # page: $79.38, $78.66, $1,666.67
    assert L.present_value(100, 3, 0.08) == pytest.approx(79.38, abs=0.005)
    assert L.present_value_continuous(100, 3, 0.08) == pytest.approx(78.66, abs=0.005)
    assert L.growing_perpetuity(100, 0.08, 0.02) == pytest.approx(1666.67, abs=0.005)
    with pytest.raises(ValueError):
        L.growing_perpetuity(100, 0.02, 0.03)


def test_npv_irr():  # page: NPV $1,428.86, IRR 16.27%, PI 1.143
    cfs = [3000, 4000, 4000, 3500]
    assert L.npv(0.10, 10000, cfs) == pytest.approx(1428.86, abs=0.005)
    assert L.irr(10000, cfs) == pytest.approx(0.1627, abs=5e-5)
    assert L.npv(L.irr(10000, cfs), 10000, cfs) == pytest.approx(0, abs=1e-6)
    assert L.profitability_index(0.10, 10000, cfs) == pytest.approx(1.143, abs=5e-4)


def test_capm():  # page: 9.10%
    assert L.capm_cost_of_equity(4.2, 0.98, 5.0) == pytest.approx(9.10)


def test_var_es():  # exact normal quantile: z = 1.6449
    r = L.parametric_var(1_000_000, 0.015, 0.95)
    assert r["z"] == pytest.approx(1.644854, abs=1e-5)
    assert r["var"] == pytest.approx(24673, abs=1)
    assert r["es"] == pytest.approx(30941, abs=1)


def test_bond():  # page: $930.35, Macaulay 6.88, modified 6.71
    r = L.bond_price_duration(0.04, 8, 0.0507)
    assert r["price"] == pytest.approx(930.35, abs=0.005)
    assert r["macaulay"] == pytest.approx(6.88, abs=0.005)
    assert r["modified"] == pytest.approx(6.71, abs=0.005)
    assert L.bond_ytm(r["price"], 0.04, 8) == pytest.approx(0.0507, abs=1e-9)
    assert L.bond_price_duration(0.05, 10, 0.05)["per_100"] == pytest.approx(100)  # par bond


def test_wacc():  # page: 7.86%
    r = L.wacc(28000, 8800, 0.091, 0.0504, 0.22)
    assert r["wacc"] == pytest.approx(0.0786, abs=5e-5)
    assert r["we"] + r["wd"] == pytest.approx(1)


def test_black_scholes():  # page: call $4.77, put $3.13
    r = L.black_scholes(40, 40, 1, 0.042, 0.25)
    assert r["call"] == pytest.approx(4.77, abs=0.005)
    assert r["put"] == pytest.approx(3.13, abs=0.005)
    assert r["call"] - r["put"] == pytest.approx(40 - 40 * math.exp(-0.042))  # put-call parity


def test_terminal_value():  # page: NOPAT $2,343, FCFF $1,875, TV $38,572
    r = L.disciplined_terminal_value(2275, 0.03, 0.15, 0.0786)
    assert r["nopat_next"] == pytest.approx(2343, abs=0.5)
    assert r["fcff_next"] == pytest.approx(1875, abs=0.5)
    assert r["tv"] == pytest.approx(38572, abs=0.5)


def test_ddm():
    assert L.gordon_ddm(2.0, 0.09, 0.03) == pytest.approx(2.06 / 0.06)
    # two-stage with equal growth rates collapses to Gordon
    assert L.two_stage_ddm(2.0, 0.09, 0.03, 5, 0.03) == pytest.approx(L.gordon_ddm(2.0, 0.09, 0.03))


def test_frontier_and_beta():
    mv = L.min_variance_point(1.0)  # perfectly correlated: min risk is the lower-risk asset
    assert mv.weight_1 == 0 and mv.sd == pytest.approx(0.20)
    m = [0.01, -0.02, 0.015, 0.003, -0.007, 0.02]
    s = [1.5 * x + 0.001 for x in m]
    r = L.ols_beta(s, m)
    assert r["beta"] == pytest.approx(1.5) and r["alpha"] == pytest.approx(0.001)
    assert L.blume_adjusted_beta(1.5) == pytest.approx(4 / 3)


def test_monte_carlo_and_gbm_are_reproducible():
    a, b = L.mini_monte_carlo(n=2000, seed=1), L.mini_monte_carlo(n=2000, seed=1)
    assert a == b and a["p5"] < a["median"] < a["p95"]
    p = L.gbm_paths(5, 10, seed=3)
    assert len(p) == 5 and all(len(x) == 11 and x[0] == 40 for x in p)


def test_cli_runs_everything(capsys):
    assert L.main([]) == 0
    out = capsys.readouterr().out
    for title in ("NPV & IRR", "Black-Scholes", "WACC builder", "Mini Monte Carlo"):
        assert title in out
