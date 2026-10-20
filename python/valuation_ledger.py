"""valuation_ledger.py: the Python version of The Valuation Ledger's calculators.

Every live calculator in the course page has a matching function here, using the
same formulas and the same default inputs, so you can check the page's numbers,
reuse them in your own models, or run them from the command line.

    python valuation_ledger.py                 # run every calculator with the page's defaults
    python valuation_ledger.py npv             # one calculator
    python valuation_ledger.py npv --investment 10000 --cash-flows 3000,4000,4000,3500 --rate 10
    python valuation_ledger.py --help          # list calculators and options

Calculators (chapter in brackets):
    pv          present value and growing perpetuity      (Time Value of Money)
    npv         NPV, IRR and profitability index          (NPV & Investment Rules)
    ddm         Gordon and two-stage dividend discount    (Valuing Stocks: DDM)
    frontier    two-asset efficient frontier              (Portfolio Theory)
    beta        beta by OLS regression                    (Beta by Regression)
    capm        CAPM cost of equity                       (CAPM & the Cost of Equity)
    var         parametric VaR and expected shortfall     (GARCH, VaR, ES)
    bond        bond price and duration                   (Duration & Convexity)
    wacc        WACC builder                              (Assembling the WACC)
    payoff      option payoffs and break-evens            (Options)
    bs          Black-Scholes call and put                (Options)
    tv          disciplined terminal value                (Terminal Value, Disciplined)
    mc          mini Monte Carlo valuation                (Monte Carlo Simulation)
    gbm         geometric Brownian motion paths           (Options)

Only the Python standard library is needed. `ddm` and `beta` add worked examples for
chapters that teach those models without a live calculator; the rest use the page's defaults.
"""
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, asdict
from typing import Callable, Sequence

# --------------------------------------------------------------------------- helpers


def norm_cdf(x: float) -> float:
    """Standard normal CDF (exact, via the error function)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return math.exp(-x * x / 2.0) / math.sqrt(2.0 * math.pi)


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation, as on the page)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be between 0 and 1")
    a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.3577518672690, -30.66479806614716, 2.506628277459239]
    b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572]
    c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416]
    lo, hi = 0.02425, 1 - 0.02425
    if p < lo:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > hi:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def _money(x: float, d: int = 2) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"${x:,.{d}f}"


# --------------------------------------------------------------------------- 1. time value of money


def present_value(cash_flow: float, years: float, rate: float) -> float:
    """PV of a single cash flow received in `years`, discrete annual compounding."""
    return cash_flow / (1 + rate) ** years


def present_value_continuous(cash_flow: float, years: float, rate: float) -> float:
    return cash_flow * math.exp(-rate * years)


def growing_perpetuity(next_cash_flow: float, rate: float, growth: float) -> float:
    """Value today of a cash flow starting next year and growing at g forever. Requires g < r."""
    if growth >= rate:
        raise ValueError("growing perpetuity requires g < r")
    return next_cash_flow / (rate - growth)


# --------------------------------------------------------------------------- 2. NPV / IRR


def npv(rate: float, investment: float, cash_flows: Sequence[float]) -> float:
    """NPV of an upfront investment followed by cash flows at the end of years 1..n."""
    return -investment + sum(cf / (1 + rate) ** (t + 1) for t, cf in enumerate(cash_flows))


def irr(investment: float, cash_flows: Sequence[float], lo: float = -0.95, hi: float = 10.0) -> float:
    """IRR by bisection, as on the page. Returns NaN when there is no sign change (no single IRR)."""
    f = lambda r: npv(r, investment, cash_flows)
    if f(lo) * f(hi) >= 0:
        return float("nan")
    for _ in range(200):
        m = (lo + hi) / 2
        if f(lo) * f(m) <= 0:
            hi = m
        else:
            lo = m
    return (lo + hi) / 2


def profitability_index(rate: float, investment: float, cash_flows: Sequence[float]) -> float:
    return (npv(rate, investment, cash_flows) + investment) / investment


# --------------------------------------------------------------------------- 3. dividend discount models


def gordon_ddm(d0: float, ke: float, g: float) -> float:
    """Gordon growth: P0 = D1 / (ke - g), with D1 = D0 (1 + g)."""
    if g >= ke:
        raise ValueError("Gordon model requires g < ke")
    return d0 * (1 + g) / (ke - g)


def two_stage_ddm(d0: float, ke: float, g_high: float, years_high: int, g_stable: float) -> float:
    """High growth for `years_high` years, then Gordon growth at g_stable."""
    if g_stable >= ke:
        raise ValueError("stable growth must be below ke")
    pv, d = 0.0, d0
    for t in range(1, years_high + 1):
        d *= 1 + g_high
        pv += d / (1 + ke) ** t
    terminal = d * (1 + g_stable) / (ke - g_stable)
    return pv + terminal / (1 + ke) ** years_high


# --------------------------------------------------------------------------- 4. portfolio theory


@dataclass
class FrontierPoint:
    weight_1: float
    mean: float
    sd: float


def two_asset_frontier(rho: float, m1: float = 0.10, s1: float = 0.25, m2: float = 0.08, s2: float = 0.20,
                       steps: int = 100) -> list[FrontierPoint]:
    """Mean and risk of every mix of two assets (page defaults: 10%/25% and 8%/20%)."""
    pts = []
    for i in range(steps + 1):
        w = i / steps
        m = w * m1 + (1 - w) * m2
        v = w * w * s1 * s1 + (1 - w) ** 2 * s2 * s2 + 2 * w * (1 - w) * rho * s1 * s2
        pts.append(FrontierPoint(w, m, math.sqrt(max(v, 0.0))))
    return pts


def min_variance_point(rho: float, **kw) -> FrontierPoint:
    return min(two_asset_frontier(rho, **kw), key=lambda p: p.sd)


# --------------------------------------------------------------------------- 5. beta and CAPM


def ols_beta(stock_returns: Sequence[float], market_returns: Sequence[float]) -> dict:
    """Beta, alpha, R² and the standard error of beta from an OLS regression of stock on market."""
    n = len(stock_returns)
    if n != len(market_returns) or n < 3:
        raise ValueError("need two equal-length return series with at least 3 observations")
    mx = sum(market_returns) / n
    my = sum(stock_returns) / n
    sxx = sum((x - mx) ** 2 for x in market_returns)
    sxy = sum((x - mx) * (y - my) for x, y in zip(market_returns, stock_returns))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - a - b * x for x, y in zip(market_returns, stock_returns)]
    sse = sum(e * e for e in resid)
    sst = sum((y - my) ** 2 for y in stock_returns)
    se_b = math.sqrt(sse / (n - 2) / sxx)
    return {"beta": b, "alpha": a, "r2": 1 - sse / sst if sst else float("nan"), "se_beta": se_b, "n": n}


def blume_adjusted_beta(raw_beta: float) -> float:
    """Blume adjustment: two-thirds raw beta, one-third the market beta of 1."""
    return 2 / 3 * raw_beta + 1 / 3


def capm_cost_of_equity(rf: float, beta: float, erp: float) -> float:
    """ke = rf + beta × equity risk premium (all as decimals)."""
    return rf + beta * erp


# --------------------------------------------------------------------------- 6. risk measures


def parametric_var(position: float, daily_vol: float, confidence: float) -> dict:
    """One-day normal VaR and expected shortfall for a position."""
    if not 0.5 < confidence < 1:
        raise ValueError("confidence must be between 0.5 and 1")
    z = norm_ppf(confidence)
    p = 1 - confidence
    return {"z": z, "var": z * daily_vol * position, "es": norm_pdf(z) / p * daily_vol * position}


# --------------------------------------------------------------------------- 7. bonds


def bond_price_duration(coupon_rate: float, years: float, ytm: float, face: float = 1000.0, freq: int = 2) -> dict:
    """Price, Macaulay and modified duration (and convexity) of a fixed-coupon bond."""
    n = round(freq * years)
    cf = coupon_rate * face / freq
    price = wsum = csum = 0.0
    for t in range(1, n + 1):
        pay = cf + (face if t == n else 0.0)
        disc = (1 + ytm / freq) ** t
        price += pay / disc
        wsum += (t / freq) * pay / disc
        csum += t * (t + 1) * pay / disc
    mac = wsum / price
    mod = mac / (1 + ytm / freq)
    convexity = csum / (price * freq ** 2 * (1 + ytm / freq) ** 2)
    return {"price": price, "per_100": price / face * 100, "macaulay": mac, "modified": mod, "convexity": convexity,
            "loss_per_100bp": price * mod / 100}


def bond_ytm(price: float, coupon_rate: float, years: float, face: float = 1000.0, freq: int = 2) -> float:
    """Yield to maturity that reproduces a market price (bisection)."""
    lo, hi = -0.99, 1.0
    f = lambda y: bond_price_duration(coupon_rate, years, y, face, freq)["price"] - price
    for _ in range(200):
        m = (lo + hi) / 2
        if f(lo) * f(m) <= 0:
            hi = m
        else:
            lo = m
    return (lo + hi) / 2


# --------------------------------------------------------------------------- 8. WACC


def wacc(equity: float, debt: float, ke: float, kd: float, tax: float) -> dict:
    """WACC = E/V·ke + D/V·kd·(1−t). Rates as decimals."""
    v = equity + debt
    we, wd = equity / v, debt / v
    return {"we": we, "wd": wd, "wacc": we * ke + wd * kd * (1 - tax)}


# --------------------------------------------------------------------------- 9. options


def option_payoffs(strike: float, premium: float, prices: Sequence[float] | None = None) -> dict:
    """Long call and long put P&L at expiry, with break-evens."""
    prices = list(prices) if prices is not None else list(range(0, 81))
    return {"call_breakeven": strike + premium, "put_breakeven": strike - premium,
            "table": [(s, max(s - strike, 0) - premium, max(strike - s, 0) - premium) for s in prices]}


def black_scholes(spot: float, strike: float, years: float, rate: float, vol: float) -> dict:
    """European call and put (put via put-call parity), plus d1, d2 and intrinsic values."""
    if years <= 0 or vol <= 0:
        raise ValueError("T and sigma must be positive")
    d1 = (math.log(spot / strike) + (rate + vol * vol / 2) * years) / (vol * math.sqrt(years))
    d2 = d1 - vol * math.sqrt(years)
    call = spot * norm_cdf(d1) - strike * math.exp(-rate * years) * norm_cdf(d2)
    put = call + strike * math.exp(-rate * years) - spot
    return {"d1": d1, "d2": d2, "call": call, "put": put,
            "call_intrinsic": max(spot - strike, 0.0), "put_intrinsic": max(strike - spot, 0.0),
            "vega": spot * norm_pdf(d1) * math.sqrt(years)}


def gbm_paths(n_paths: int = 40, steps: int = 120, s0: float = 40.0, mu: float = 0.08, sigma: float = 0.25,
              years: float = 1.0, seed: int | None = 7) -> list[list[float]]:
    """Simulated price paths under geometric Brownian motion."""
    rng = random.Random(seed)
    dt = years / steps
    out = []
    for _ in range(n_paths):
        p = [s0]
        for _ in range(steps):
            z = rng.gauss(0.0, 1.0)
            p.append(p[-1] * math.exp((mu - sigma * sigma / 2) * dt + sigma * math.sqrt(dt) * z))
        out.append(p)
    return out


# --------------------------------------------------------------------------- 10. terminal value and Monte Carlo


def disciplined_terminal_value(final_nopat: float, g: float, roic: float, wacc_: float) -> dict:
    """TV where growth must be paid for: reinvestment rate = g / ROIC (value-driver formula)."""
    if g >= wacc_:
        raise ValueError("g must be below WACC")
    nopat_next = final_nopat * (1 + g)
    reinvest = g / roic
    fcff = nopat_next * (1 - reinvest)
    return {"nopat_next": nopat_next, "reinvestment_rate": reinvest, "fcff_next": fcff, "tv": fcff / (wacc_ - g)}


def mini_monte_carlo(growth_mean: float = 0.068, growth_sd: float = 0.015, wacc_mean: float = 0.0786,
                     wacc_sd: float = 0.006, rho: float = -0.2, n: int = 5000, market_price: float = 40.0,
                     seed: int | None = 7) -> dict:
    """The course's worked-example company ("Meridian"): 5-year DCF with correlated growth and WACC draws.

    Revenue 10,000 grows at g for five years, EBIT margin 21%, tax 22%, FCFF = 88% of NOPAT,
    terminal growth min(3%, WACC − 1.2%) with 15% ROIC; bridge +600 cash −8,800 debt −150 other; 705 shares.
    """
    rho = max(-0.95, min(0.95, rho))
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        z1 = rng.gauss(0, 1)
        z2 = rho * z1 + math.sqrt(1 - rho * rho) * rng.gauss(0, 1)
        g = growth_mean + growth_sd * z1
        w = max(wacc_mean + wacc_sd * z2, 0.045)
        g_t = min(0.03, w - 0.012)
        rev, pv, nop = 10000.0, 0.0, 0.0
        for t in range(1, 6):
            rev *= 1 + g
            nop = rev * 0.21 * 0.78
            pv += nop * 0.88 / (1 + w) ** t
        nop6 = nop * (1 + g_t)
        tv = nop6 * (1 - g_t / 0.15) / (w - g_t)
        ev = pv + tv / (1 + w) ** 5
        vals.append((ev + 600 - 8800 - 150) / 705)
    vals.sort()
    q = lambda p: vals[int(p * n)]
    return {"p5": q(0.05), "p25": q(0.25), "median": q(0.5), "p75": q(0.75), "p95": q(0.95),
            "prob_above_price": sum(v > market_price for v in vals) / n, "n": n}


# --------------------------------------------------------------------------- command line


def _print_kv(title: str, rows: list[tuple[str, str]]) -> None:
    print(f"\n{title}\n" + "─" * len(title))
    w = max(len(k) for k, _ in rows)
    for k, v in rows:
        print(f"  {k:<{w}}  {v}")


def cmd_pv(a):
    rows = [(f"PV of {_money(a.cash_flow)} in {a.years:g}y at {a.rate:g}%", _money(present_value(a.cash_flow, a.years, a.rate / 100))),
            ("Same, continuous compounding", _money(present_value_continuous(a.cash_flow, a.years, a.rate / 100)))]
    try:
        rows.append((f"Growing perpetuity (g {a.growth:g}%)", _money(growing_perpetuity(a.cash_flow, a.rate / 100, a.growth / 100))))
    except ValueError as e:
        rows.append(("Growing perpetuity", str(e)))
    _print_kv("Present value & growing perpetuity", rows)


def cmd_npv(a):
    cfs = [float(x) for x in a.cash_flows.split(",") if x.strip()]
    v = npv(a.rate / 100, a.investment, cfs)
    r = irr(a.investment, cfs)
    _print_kv("NPV & IRR", [(f"NPV at {a.rate:g}%", f"{_money(v)}  →  {'ACCEPT' if v > 0 else 'REJECT'}"),
                            ("IRR", "no single IRR" if math.isnan(r) else f"{r:.2%}"),
                            ("Profitability index", f"{profitability_index(a.rate / 100, a.investment, cfs):.3f}")])


def cmd_ddm(a):
    _print_kv("Dividend discount models", [
        (f"Gordon (D0 {_money(a.d0)}, ke {a.ke:g}%, g {a.g:g}%)", _money(gordon_ddm(a.d0, a.ke / 100, a.g / 100))),
        (f"Two-stage ({a.g_high:g}% for {a.years_high}y, then {a.g:g}%)",
         _money(two_stage_ddm(a.d0, a.ke / 100, a.g_high / 100, a.years_high, a.g / 100)))])


def cmd_frontier(a):
    mv = min_variance_point(a.rho)
    rows = [(f"w1={p.weight_1:.0%}", f"mean {p.mean:.2%}  sd {p.sd:.2%}") for p in two_asset_frontier(a.rho, steps=10)]
    rows.append(("Minimum variance", f"w1 {mv.weight_1:.0%}  mean {mv.mean:.2%}  sd {mv.sd:.2%}"))
    _print_kv(f"Two-asset frontier (ρ = {a.rho:.2f})", rows)


def cmd_beta(a):
    rng = random.Random(a.seed)
    m = [rng.gauss(0.0004, 0.01) for _ in range(a.n)]
    s = [a.true_beta * x + rng.gauss(0, 0.012) for x in m]
    r = ols_beta(s, m)
    _print_kv("Beta by OLS (simulated daily returns)", [
        ("True beta", f"{a.true_beta:.2f}"), ("Estimated beta", f"{r['beta']:.3f}  (s.e. {r['se_beta']:.3f})"),
        ("95% interval", f"{r['beta'] - 1.96 * r['se_beta']:.3f} to {r['beta'] + 1.96 * r['se_beta']:.3f}"),
        ("Blume-adjusted", f"{blume_adjusted_beta(r['beta']):.3f}"), ("R²", f"{r['r2']:.3f}"), ("Observations", str(r["n"]))])


def cmd_capm(a):
    ke = capm_cost_of_equity(a.rf, a.beta, a.erp)
    _print_kv("CAPM cost of equity", [(f"ke = {a.rf:g}% + {a.beta:g} × {a.erp:g}%", f"{ke:.2f}%"),
                                      ("At β ± 0.20", f"{capm_cost_of_equity(a.rf, a.beta - .2, a.erp):.2f}% to {capm_cost_of_equity(a.rf, a.beta + .2, a.erp):.2f}%")])


def cmd_var(a):
    r = parametric_var(a.position, a.vol / 100, a.confidence / 100)
    _print_kv("Parametric VaR & expected shortfall (one day)", [
        (f"{a.confidence:g}% VaR", f"{_money(r['var'], 0)}  (z = {r['z']:.3f})"), ("Expected shortfall", _money(r["es"], 0))])


def cmd_bond(a):
    r = bond_price_duration(a.coupon / 100, a.years, a.ytm / 100, a.face)
    _print_kv("Bond price & duration (semiannual)", [
        ("Price", f"{_money(r['price'])}  ({r['per_100']:.2f} per 100)"), ("Macaulay duration", f"{r['macaulay']:.2f} yrs"),
        ("Modified duration", f"{r['modified']:.2f}"), ("Convexity", f"{r['convexity']:.2f}"),
        ("If yields +100bp", f"ΔP ≈ −{r['modified']:.2f}% ≈ {_money(r['loss_per_100bp'])} lost")])


def cmd_wacc(a):
    r = wacc(a.equity, a.debt, a.ke / 100, a.kd / 100, a.tax / 100)
    _print_kv("WACC builder", [("Weights", f"E/V {r['we']:.1%}   D/V {r['wd']:.1%}"), ("WACC", f"{r['wacc']:.2%}")])


def cmd_payoff(a):
    r = option_payoffs(a.strike, a.premium, range(0, 81, 10))
    rows = [(f"S = {s}", f"call {c:+.1f}   put {p:+.1f}") for s, c, p in r["table"]]
    rows += [("Call break-even", f"{r['call_breakeven']:g}"), ("Put break-even", f"{r['put_breakeven']:g}")]
    _print_kv(f"Option P&L at expiry (K {a.strike:g}, premium {a.premium:g})", rows)


def cmd_bs(a):
    r = black_scholes(a.spot, a.strike, a.years, a.rate / 100, a.vol / 100)
    _print_kv("Black-Scholes", [("d1 / d2", f"{r['d1']:.3f} / {r['d2']:.3f}"), ("Call", _money(r["call"])),
                                ("Put (via parity)", _money(r["put"])), ("Vega (per 1.00 σ)", f"{r['vega']:.2f}"),
                                ("Intrinsic", f"call {_money(r['call_intrinsic'])} · put {_money(r['put_intrinsic'])}")])


def cmd_tv(a):
    r = disciplined_terminal_value(a.nopat, a.g / 100, a.roic / 100, a.wacc / 100)
    # as on the page: hold next-year NOPAT fixed and move only g, to isolate the denominator effect
    g_lo, g_hi, w = a.g / 100 - .005, min(a.g / 100 + .005, a.wacc / 100 - .005), a.wacc / 100
    lo = r["nopat_next"] * (1 - g_lo / (a.roic / 100)) / (w - g_lo)
    hi = r["nopat_next"] * (1 - g_hi / (a.roic / 100)) / (w - g_hi)
    _print_kv("Disciplined terminal value", [
        ("NOPAT (T+1)", _money(r["nopat_next"], 0)), ("Reinvestment = g/ROIC", f"{r['reinvestment_rate']:.1%}"),
        ("FCFF (T+1)", _money(r["fcff_next"], 0)), ("Terminal value", _money(r["tv"], 0)),
        ("g ± 0.5%", f"{_money(lo, 0)} … {_money(hi, 0)}")])


def cmd_mc(a):
    r = mini_monte_carlo(a.g / 100, a.g_sd / 100, a.wacc / 100, a.wacc_sd / 100, a.rho, a.n, a.price, a.seed)
    _print_kv(f"Mini Monte Carlo ({r['n']:,} paths)", [
        ("P5 / P25", f"{_money(r['p5'])} / {_money(r['p25'])}"), ("Median", _money(r["median"])),
        ("P75 / P95", f"{_money(r['p75'])} / {_money(r['p95'])}"), (f"P(value > {_money(a.price)})", f"{r['prob_above_price']:.1%}")])


def cmd_gbm(a):
    paths = gbm_paths(a.paths, a.steps, a.s0, a.mu / 100, a.sigma / 100, seed=a.seed)
    ends = sorted(p[-1] for p in paths)
    _print_kv(f"GBM: {a.paths} paths over one year", [
        ("Start", _money(a.s0)), ("Lowest / highest end", f"{_money(ends[0])} / {_money(ends[-1])}"),
        ("Median end", _money(ends[len(ends) // 2])), ("Mean end (theory: S0·e^μ)", f"{_money(sum(ends) / len(ends))} ({_money(a.s0 * math.exp(a.mu / 100))})")])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="calc")

    def add(name: str, fn: Callable, help_: str, args: list[tuple]):
        s = sub.add_parser(name, help=help_)
        for flag, typ, default, h in args:
            s.add_argument(flag, type=typ, default=default, help=f"{h} (default {default})")
        s.set_defaults(fn=fn)

    add("pv", cmd_pv, "present value & growing perpetuity", [("--cash-flow", float, 100, "cash flow $"), ("--years", float, 3, "years"), ("--rate", float, 8, "discount rate %"), ("--growth", float, 2, "perpetuity growth %")])
    add("npv", cmd_npv, "NPV, IRR, PI", [("--investment", float, 10000, "initial investment $"), ("--cash-flows", str, "3000,4000,4000,3500", "comma-separated cash flows"), ("--rate", float, 10, "discount rate %")])
    add("ddm", cmd_ddm, "dividend discount models", [("--d0", float, 2.00, "last dividend $"), ("--ke", float, 9.0, "cost of equity %"), ("--g", float, 3.0, "stable growth %"), ("--g-high", float, 8.0, "high growth %"), ("--years-high", int, 5, "high-growth years")])
    add("frontier", cmd_frontier, "two-asset efficient frontier", [("--rho", float, 0.30, "correlation")])
    add("beta", cmd_beta, "beta by OLS on simulated returns", [("--true-beta", float, 1.2, "beta used to simulate"), ("--n", int, 504, "daily observations"), ("--seed", int, 7, "random seed")])
    add("capm", cmd_capm, "CAPM cost of equity", [("--rf", float, 4.2, "risk-free %"), ("--beta", float, 0.98, "beta"), ("--erp", float, 5.0, "equity risk premium %")])
    add("var", cmd_var, "parametric VaR & ES", [("--position", float, 1_000_000, "position $"), ("--vol", float, 1.5, "daily volatility %"), ("--confidence", float, 95, "confidence %")])
    add("bond", cmd_bond, "bond price & duration", [("--coupon", float, 4.0, "coupon rate %/yr"), ("--years", float, 8, "years to maturity"), ("--ytm", float, 5.07, "yield %"), ("--face", float, 1000, "face value $")])
    add("wacc", cmd_wacc, "WACC builder", [("--equity", float, 28000, "equity value $M"), ("--debt", float, 8800, "debt $M"), ("--ke", float, 9.10, "cost of equity %"), ("--kd", float, 5.04, "pre-tax cost of debt %"), ("--tax", float, 22, "tax rate %")])
    add("payoff", cmd_payoff, "option payoffs", [("--strike", float, 40, "strike"), ("--premium", float, 5, "premium")])
    add("bs", cmd_bs, "Black-Scholes", [("--spot", float, 40, "spot"), ("--strike", float, 40, "strike"), ("--years", float, 1, "years to expiry"), ("--rate", float, 4.2, "risk-free %"), ("--vol", float, 25, "volatility %")])
    add("tv", cmd_tv, "disciplined terminal value", [("--nopat", float, 2275, "final-year NOPAT $"), ("--g", float, 3.0, "terminal growth %"), ("--roic", float, 15, "ROIC on new capital %"), ("--wacc", float, 7.86, "WACC %")])
    add("mc", cmd_mc, "mini Monte Carlo valuation", [("--g", float, 6.8, "growth mean %"), ("--g-sd", float, 1.5, "growth sd %"), ("--wacc", float, 7.86, "WACC mean %"), ("--wacc-sd", float, 0.6, "WACC sd %"), ("--rho", float, -0.2, "growth/WACC correlation"), ("--n", int, 5000, "paths"), ("--price", float, 40, "market price $"), ("--seed", int, 7, "random seed")])
    add("gbm", cmd_gbm, "GBM price paths", [("--paths", int, 40, "paths"), ("--steps", int, 120, "steps"), ("--s0", float, 40, "start price"), ("--mu", float, 8, "drift %"), ("--sigma", float, 25, "volatility %"), ("--seed", int, 7, "random seed")])
    return p


def main(argv=None) -> int:
    p = build_parser()
    a = p.parse_args(argv)
    if a.calc is None:  # run everything with the page's defaults
        for name, sp in p._subparsers._group_actions[0].choices.items():
            sp.parse_args([], namespace=(ns := argparse.Namespace()))
            ns.fn(ns)
        return 0
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
