# The Valuation Ledger

[![CI](https://github.com/Speedkelvyn/Valuation-gem/actions/workflows/ci.yml/badge.svg)](https://github.com/Speedkelvyn/Valuation-gem/actions/workflows/ci.yml)

A complete, interactive course in valuing companies, in one page, with worked examples, calculators and self-tests throughout.

**Read the course:** https://speedkelvyn.github.io/Valuation-gem/

## What it covers
Eight parts and more than 40 chapters, from first principles to a full valuation in code:

| Part | Topics |
|---|---|
| **Foundations** | Time value of money, financial statements line by line, ratios, NPV and capital budgeting, dividend discount models |
| **Risk & Return** | Return statistics, portfolio theory, beta by OLS regression, CAPM, APT, Fama-French factors, GARCH, VaR and expected shortfall |
| **Fixed Income** | Bond pricing, yield-curve bootstrapping, duration and convexity, immunization, swaps, credit risk and the cost of debt |
| **Corporate Finance Theory** | Capital structure, WACC, payout policy and APV, options and real options, IPOs, M&A and LBOs, hedging, distress |
| **The Valuation Engine** | Forecasting the statements, free cash flow and DCF, terminal value, enterprise value to share price |
| **Uncertainty & Cross-Checks** | Scenarios and reverse DCF, Monte Carlo, multiples, the football field and the audit |
| **The Python Toolkit** | pandas for finance, regression, bond curves and a full DCF in code |
| **The Capstone** | Final exam, drill deck, formula sheet, glossary and a worked end-to-end answer key |

## Inside the page
- Interactive calculators and charts
- Quizzes, a drill deck and a scored final exam
- Formulas typeset with KaTeX
- A companion Excel workbook with DCF and DDM model sheets: [`workbooks/valuation-models-dcf-ddm.xlsx`](workbooks/valuation-models-dcf-ddm.xlsx)

## Python version
[`python/valuation_ledger.py`](python/valuation_ledger.py) has every calculator in the course as a Python function, with the same formulas and default inputs, so results match the page:

```bash
python python/valuation_ledger.py                 # run all calculators with the course defaults
python python/valuation_ledger.py npv --cash-flows 3000,4000,4000,3500 --rate 10
python python/valuation_ledger.py bs --spot 40 --strike 45 --vol 30
```

Covers present value, NPV/IRR, dividend discount models, the efficient frontier, beta by OLS, CAPM, VaR and expected shortfall, bond price and duration, WACC, option payoffs, Black-Scholes, disciplined terminal value, Monte Carlo and GBM paths. Standard library only; tests in `python/tests` check each function against the page's outputs.

## Run it
Open `index.html` in a browser, or use the live link above.

## Related
Applied valuations built with the same methods (Dell Technologies and Sandisk) are on the [portfolio site](https://speedkelvyn.github.io/Speedy.github.io/#valuations).

## Development

```bash
pip install ruff pytest
ruff check . && ruff format --check .   # lint and formatting
pytest -q                               # tests
npx html-validate@8.24.1 *.html         # HTML validation
```

Style is set by `ruff.toml`, `.prettierrc.json`, `.htmlvalidate.json` and `.editorconfig`; GitHub Actions runs the same checks on every push.
