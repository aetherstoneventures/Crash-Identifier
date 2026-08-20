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

Freezing and holdout machinery lives in `src/v6/freeze.py` and
`scripts/v6/holdout_eval.py` — see the scorecard's *"The BLIND set is not
blind"* section for why it exists.

## The short version

The v6 engine estimates `P(maxDD ≥ x% within h trading days)` for any `x` and
`h` you ask for, from a single fit. Its gate *ranks* days usefully — pooled
precision 0.859 at 2.39× the base rate with a ~40-day median lead, and
drawdown cut from −36.4% to −25.7% on 2021–2026.

It **fails its own kill criteria on every window**. The probabilities are not
trustworthy in absolute terms (the base rate of the target event varies 3.4×
across decades), the two earliest folds never fire, the `credit_led` archetype
has never fired at all, it does not beat buy-and-hold on return, and the
2021+ window was inspected during development so it is not a clean holdout.

Read the scorecard before quoting any number above.
