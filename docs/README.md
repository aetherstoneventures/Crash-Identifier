# `docs/` — index

Start here, in this order.

| Document | What it answers |
|---|---|
| [V6_HONEST_SCORECARD.md](V6_HONEST_SCORECARD.md) | **How well does it work, and where does it still fail?** Results, all five kill criteria, and the disclosure that the 2021+ window is no longer a clean holdout. |
| [V6_POSTMORTEM.md](V6_POSTMORTEM.md) | **Is this idea even possible?** The nine defects that made v6.0.0-alpha untestable, each verified numerically, and what the repaired system does and does not establish. |
| [CRASH_KPI_ENGINE_DESIGN.md](CRASH_KPI_ENGINE_DESIGN.md) | The approved architecture: five engines, the Bayesian aggregator, the L1/L2/L3 gate, and the pre-declared validation protocol. |
| [DATA_SOURCES.md](DATA_SOURCES.md) | Every column, its FRED series, its point-in-time rule, and why some columns are quarantined. |
| [HISTORICAL_CRASHES_REFERENCE.md](HISTORICAL_CRASHES_REFERENCE.md) | Reference table of historical US equity crashes. |
| [INVESTOR_LAWS.md](INVESTOR_LAWS.md) | Background research notes on crash dynamics. |
| [CHANGELOG.md](CHANGELOG.md) | Version history. |

## The short version

The v6 engine estimates `P(maxDD ≥ x% within h trading days)` for any `x` and
`h` you ask for, from a single fit. On the 2021–2026 window it passes all five
pre-declared kill criteria — precision 0.776 at 2.07× the base rate, a
38.5-day median lead, drawdown cut from −36.4% to −26.5% at equal Sharpe.

It does **not** beat buy-and-hold on return, its probabilities are not yet
well calibrated, three of four walk-forward folds still fail, and the BLIND
window was inspected during development. The scorecard says all of this in
more detail; read it before quoting any number above.
