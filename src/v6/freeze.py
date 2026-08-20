"""Configuration freezing — making "no retuning" checkable instead of promised.

WHY THIS EXISTS
===============
The v6 protocol requires a single-shot BLIND evaluation with no retuning. In
practice that promise was broken during v6.1 development: the 2022 gate
diagnostics were inspected, the Layer 1 archetype split was designed in
response, and the 2021+ window stopped being a clean holdout. Nothing in the
codebase noticed or could have.

A promise that cannot be checked is not a control. This module turns the
freeze into an artefact: every tunable value is serialised and hashed, so a
later evaluation can *prove* it ran the same configuration rather than
asserting it.

The hash covers only decision-affecting settings — engine hyperparameters,
gate thresholds, aggregator options, fold boundaries, kill criteria. It
deliberately excludes paths and anything environmental, so moving the repo
does not invalidate a freeze.

USAGE
-----
Freeze the current configuration::

    python -c "from src.v6.freeze import write_freeze; write_freeze('6.1.0')"

Later, evaluate it on data that did not exist at freeze time::

    python scripts/v6/holdout_eval.py --freeze data/v6_artifacts/frozen_config_v6.1.0.json

The evaluator refuses to run if the live configuration no longer matches the
frozen hash, so an accidental retune surfaces as an error rather than as a
quietly better number.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

from src.v6.config import (
    ARTIFACTS_DIR, BLIND_START, CONFIG, DEFAULT_HORIZON_DAYS, DEFAULT_X_PCT,
    SUPPORTED_HORIZON_DAYS, SUPPORTED_X_PCT, WALK_FORWARD_FOLDS,
)


def _normalise(value: Any) -> Any:
    """Canonicalise a value so the hash survives a JSON round-trip.

    Tuples become lists, because a config read back from a freeze file
    returns lists and would otherwise register as drift when nothing
    actually changed. A freeze mechanism that cries wolf gets ignored.
    """
    if isinstance(value, tuple):
        return [_normalise(v) for v in value]
    if isinstance(value, list):
        return [_normalise(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalise(v) for k, v in value.items()}
    return value


def config_payload() -> Dict[str, Any]:
    """Every decision-affecting setting, as plain JSON-able data.

    Paths and environment are excluded on purpose: they change with the
    checkout location and have no bearing on what the model decides.
    """
    return _normalise({
        "walk_forward_folds": list(WALK_FORWARD_FOLDS),
        "blind_start": BLIND_START,
        "default_x_pct": DEFAULT_X_PCT,
        "default_horizon_td": DEFAULT_HORIZON_DAYS,
        "supported_x_pct": list(SUPPORTED_X_PCT),
        "supported_horizon_td": list(SUPPORTED_HORIZON_DAYS),
        "anomaly": dataclasses.asdict(CONFIG.anomaly),
        "regime": dataclasses.asdict(CONFIG.regime),
        "analog": dataclasses.asdict(CONFIG.analog),
        "causal": dataclasses.asdict(CONFIG.causal),
        "aggregator": dataclasses.asdict(CONFIG.aggregator),
        "gate": dataclasses.asdict(CONFIG.gate),
        "kill": dataclasses.asdict(CONFIG.kill),
    })


def config_hash(payload: Optional[Dict[str, Any]] = None) -> str:
    """Stable SHA-256 over the configuration payload."""
    data = payload if payload is not None else config_payload()
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def freeze_record(version: str, lock_date: str, note: str = "") -> Dict[str, Any]:
    """Assemble a freeze record for `version`.

    Parameters
    ----------
    version : str
        Version label, e.g. "6.1.0". Should match the git tag.
    lock_date : str
        The last date whose data informed this configuration. A later
        holdout evaluation scores only dates strictly after it — that is
        what makes the evaluation genuinely out-of-sample.
    note : str
        Free text recorded alongside, e.g. known contamination.
    """
    payload = config_payload()
    return {
        "version": version,
        "lock_date": lock_date,
        "config_hash": config_hash(payload),
        "note": note,
        "config": payload,
    }


def write_freeze(version: str, lock_date: str = "2026-08-19",
                 note: str = "", path: Optional[str] = None) -> Path:
    """Write a freeze record and return its path."""
    record = freeze_record(version, lock_date, note)
    out = Path(path) if path else ARTIFACTS_DIR / f"frozen_config_v{version}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(record, f, indent=2, sort_keys=False)
    return out


def load_freeze(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def verify_freeze(path: str) -> Dict[str, Any]:
    """Check the live configuration against a freeze record.

    Returns a dict with `matches`, the two hashes, and a `differences` map of
    the settings that drifted, so a mismatch says *what* changed rather than
    only that something did.
    """
    record = load_freeze(path)
    live = config_payload()
    frozen = record.get("config", {})
    live_hash = config_hash(live)

    differences: Dict[str, Any] = {}
    for section in sorted(set(frozen) | set(live)):
        before, after = frozen.get(section), live.get(section)
        if before == after:
            continue
        if isinstance(before, dict) and isinstance(after, dict):
            for key in sorted(set(before) | set(after)):
                if before.get(key) != after.get(key):
                    differences[f"{section}.{key}"] = {
                        "frozen": before.get(key), "live": after.get(key),
                    }
        else:
            differences[section] = {"frozen": before, "live": after}

    return {
        "matches": live_hash == record.get("config_hash"),
        "frozen_hash": record.get("config_hash"),
        "live_hash": live_hash,
        "version": record.get("version"),
        "lock_date": record.get("lock_date"),
        "note": record.get("note", ""),
        "differences": differences,
    }
