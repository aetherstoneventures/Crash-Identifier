# Legacy artefacts (pre-v6)

Frozen outputs from earlier model generations, kept for provenance. **Nothing
in the v6 pipeline reads these files.**

| File | What it is |
|---|---|
| `alarm_config_v5.json` | Frozen v5 alarm hysteresis configuration |
| `experiment_A_vol_gate.json` | v5-era research scorecard — volatility gate |
| `experiment_C_reentry.json` | v5-era research scorecard — re-entry rule |
| `experiment_D_round2.json` | v5-era research scorecard — round 2 |
| `v6_kill_verdict.json` | The v6.0.0-alpha kill verdict |

`v6_kill_verdict.json` records the alpha's failure. The reasons behind that
failure — and why most of them were mechanical rather than empirical — are
documented in [`../../docs/V6_POSTMORTEM.md`](../../docs/V6_POSTMORTEM.md).

Current validation artefacts live in [`../v6_artifacts/`](../v6_artifacts).
The full v5 model and code are at git tag `v5-BENCHMARK` and branch
`v5-benchmark-protected`.
