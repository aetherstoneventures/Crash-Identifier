#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# run.sh — single-command launcher for the Crash-Identifier pipeline + dashboard
#
# Behaviour:
#   • If a venv already exists, asks whether to do a FRESH install or REUSE it.
#   • If no venv exists, always does a FRESH install.
#   • Then runs the v6 Crash KPI Engine pipeline:
#       FRED backfill → crash events → walk-forward + BLIND validation
#       → Streamlit dashboard (v6 Crash KPI Engine tab).
#
# Usage:
#   ./run.sh                    # interactive prompt
#   ./run.sh --fresh            # force fresh install (non-interactive)
#   ./run.sh --reuse            # force reuse existing venv (non-interactive)
#   ./run.sh --dashboard-only   # skip pipeline, just launch dashboard
#   ./run.sh --skip-backfill    # skip the FRED refresh (offline / rate-limited)
#   ./run.sh --x 10 --h 63      # crash threshold % and horizon in trading days
# ----------------------------------------------------------------------------
set -euo pipefail

# ---- locate project root --------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"
ROOT="$SCRIPT_DIR"
VENV="$ROOT/venv"

# ---- colours --------------------------------------------------------------
R='\033[0;31m'; G='\033[0;32m'; Y='\033[1;33m'; B='\033[0;34m'; C='\033[0;36m'; N='\033[0m'

banner() {
    echo -e "${B}================================================================================${N}"
    echo -e "${B}  Crash-Identifier · v6 Crash KPI Engine${N}"
    echo -e "${B}================================================================================${N}"
}

# ---- parse args -----------------------------------------------------------
MODE=""
DASHBOARD_ONLY=0
SKIP_BACKFILL=0
X_PCT=10
HORIZON=63
while [ $# -gt 0 ]; do
    case "$1" in
        --fresh)          MODE="fresh" ;;
        --reuse)          MODE="reuse" ;;
        --dashboard-only) DASHBOARD_ONLY=1 ;;
        --skip-backfill)  SKIP_BACKFILL=1 ;;
        --x)              X_PCT="${2:?--x needs a value}"; shift ;;
        --h)              HORIZON="${2:?--h needs a value}"; shift ;;
        -h|--help)
            sed -n '2,22p' "$0"; exit 0 ;;
        *) echo -e "${R}Unknown arg: $1${N}"; exit 2 ;;
    esac
    shift
done

banner

# ---- decide fresh vs reuse ------------------------------------------------
if [ -z "$MODE" ]; then
    if [ -d "$VENV" ] && [ -x "$VENV/bin/python3" ]; then
        echo -e "${Y}Existing virtual environment detected at:${N} $VENV"
        echo -e "  Python: $("$VENV/bin/python3" --version 2>&1)"
        echo
        echo -e "${C}Choose mode:${N}"
        echo "  [1] FRESH  — delete venv, recreate, reinstall requirements (slow, clean)"
        echo "  [2] REUSE  — keep existing venv, skip pip install (fast)"
        echo
        read -r -p "Enter 1 or 2 [default: 2]: " choice
        case "${choice:-2}" in
            1|f|F|fresh|FRESH)  MODE="fresh" ;;
            2|r|R|reuse|REUSE|"") MODE="reuse" ;;
            *) echo -e "${R}Invalid choice.${N}"; exit 2 ;;
        esac
    else
        echo -e "${Y}No existing venv found — running FRESH install.${N}"
        MODE="fresh"
    fi
fi
echo -e "${G}Mode: $MODE${N}\n"

# ---- find a compatible Python (3.9–3.12) for fresh installs ---------------
find_python() {
    local candidates=(
        /opt/homebrew/bin/python3.11 /opt/homebrew/bin/python3.12
        /opt/homebrew/bin/python3.10 /opt/homebrew/bin/python3.9
        /usr/local/bin/python3.11    /usr/local/bin/python3.12
        /usr/local/bin/python3.10    /usr/local/bin/python3.9
        python3.11 python3.12 python3.10 python3.9 python3
    )
    for py in "${candidates[@]}"; do
        if command -v "$py" >/dev/null 2>&1; then
            local v
            v=$("$py" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
            local maj=${v%%.*} min=${v##*.}
            if [ "$maj" = "3" ] && [ "$min" -ge 9 ] && [ "$min" -le 12 ]; then
                echo "$py"; return 0
            fi
        fi
    done
    return 1
}

# ---- FRESH path -----------------------------------------------------------
if [ "$MODE" = "fresh" ]; then
    echo -e "${Y}[1/3] Recreating virtual environment...${N}"
    [ -d "$VENV" ] && rm -rf "$VENV"
    PY=$(find_python) || {
        echo -e "${R}ERROR: Python 3.9–3.12 not found. Install via: brew install python@3.11${N}"; exit 1; }
    echo "  Using: $PY ($("$PY" --version 2>&1))"
    "$PY" -m venv "$VENV"
    # shellcheck source=/dev/null
    source "$VENV/bin/activate"
    echo -e "${Y}[2/3] Installing requirements (this can take a few minutes)...${N}"
    "$VENV/bin/pip" install --upgrade pip setuptools wheel >/dev/null
    "$VENV/bin/pip" install -r requirements.txt
    echo -e "${G}✓ Fresh environment ready${N}\n"
else
    echo -e "${Y}[1/3] Reusing existing venv at $VENV${N}"
    # shellcheck source=/dev/null
    source "$VENV/bin/activate"
    echo -e "${G}✓ Activated · $(python --version 2>&1)${N}\n"
fi

PYTHON="$VENV/bin/python3"
STREAMLIT="$VENV/bin/streamlit"

# ---- dashboard-only shortcut ---------------------------------------------
if [ "$DASHBOARD_ONLY" -eq 1 ]; then
    echo -e "${B}--- Dashboard-only mode ---${N}"
    echo -e "${G}→ http://localhost:8501${N}\n"
    exec "$STREAMLIT" run "$ROOT/src/dashboard/pages/v6_kpi_engine.py"
fi

# ---- pipeline -------------------------------------------------------------
echo -e "${Y}[2/3] Running pipeline...${N}\n"
run_step() {
    local title=$1; shift
    echo -e "${B}--- $title ---${N}"
    "$PYTHON" -W ignore "$@"
    echo
}

# 1. Repair/refresh the data layer from FRED. This is the step that keeps
#    the price series full-history and macro series stamped on their real
#    release dates; see docs/V6_POSTMORTEM.md for why it exists.
if [ "$SKIP_BACKFILL" -eq 1 ]; then
    echo -e "${Y}--- 1. FRED backfill SKIPPED (--skip-backfill) ---${N}\n"
else
    if [ -f "$ROOT/.env" ] && grep -q '^FRED_API_KEY=.\+' "$ROOT/.env"; then
        run_step "1. Repairing + refreshing indicators from FRED" \
            "$ROOT/scripts/data/backfill_fred.py"
    else
        echo -e "${Y}--- 1. FRED backfill skipped: no FRED_API_KEY in .env ---${N}"
        echo -e "    Add one (free: https://fred.stlouisfed.org/docs/api/api_key.html)"
        echo -e "    or pass --skip-backfill to silence this.\n"
    fi
fi

run_step "2. Labelling historical crash episodes" \
    "$ROOT/scripts/data/populate_crash_events.py"

run_step "3. Walk-forward + BLIND validation (x=${X_PCT}%, h=${HORIZON}d)" \
    "$ROOT/scripts/v6/validate.py" both --x "$X_PCT" --h "$HORIZON"

echo -e "${G}✓ Pipeline complete${N}"
echo -e "${C}  Scorecard artefact: data/v6_artifacts/v6_validation_x${X_PCT}_h${HORIZON}.json${N}"
echo -e "${C}  Honest verdict:     docs/V6_HONEST_SCORECARD.md${N}\n"

# ---- dashboard ------------------------------------------------------------
echo -e "${Y}[3/3] Launching dashboard...${N}"
echo -e "${G}→ http://localhost:8501${N}"
echo -e "${C}(Ctrl+C to stop)${N}\n"
exec "$STREAMLIT" run "$ROOT/src/dashboard/pages/v6_kpi_engine.py"
