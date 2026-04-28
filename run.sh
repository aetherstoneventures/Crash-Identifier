#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# run.sh — single-command launcher for the Crash-Identifier pipeline + dashboard
#
# Behaviour:
#   • If a venv already exists, asks whether to do a FRESH install or REUSE it.
#   • If no venv exists, always does a FRESH install.
#   • Then runs: data collection → crash events → training → predictions →
#     evaluation → Streamlit dashboard (v5 Production tab).
#
# Usage:
#   ./run.sh           # interactive prompt
#   ./run.sh --fresh   # force fresh install (non-interactive)
#   ./run.sh --reuse   # force reuse existing venv (non-interactive)
#   ./run.sh --dashboard-only   # skip pipeline, just launch dashboard
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
    echo -e "${B}  Crash-Identifier · Pipeline Launcher${N}"
    echo -e "${B}================================================================================${N}"
}

# ---- parse args -----------------------------------------------------------
MODE=""
DASHBOARD_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --fresh)          MODE="fresh" ;;
        --reuse)          MODE="reuse" ;;
        --dashboard-only) DASHBOARD_ONLY=1 ;;
        -h|--help)
            sed -n '2,18p' "$0"; exit 0 ;;
        *) echo -e "${R}Unknown arg: $arg${N}"; exit 2 ;;
    esac
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
    exec "$STREAMLIT" run "$ROOT/src/dashboard/app.py"
fi

# ---- pipeline -------------------------------------------------------------
echo -e "${Y}[2/3] Running pipeline...${N}\n"
run_step() {
    local title=$1; shift
    echo -e "${B}--- $title ---${N}"
    "$PYTHON" -W ignore "$@"
    echo
}

run_step "1. Collecting market & economic data"      "$ROOT/scripts/data/collect_data.py"
run_step "2. Populating historical crash events"     "$ROOT/scripts/data/populate_crash_events.py"
run_step "3a. Training StatV3 (statistical model)"   "$ROOT/scripts/training/train_statistical_model_v3.py"
run_step "3b. Generating StatV3 predictions"         "$ROOT/scripts/utils/generate_predictions_v5.py"
run_step "3c. Training v5 walk-forward (canonical)"  "$ROOT/scripts/training/train_v5_walkforward.py"
run_step "3d. Training bottom-predictor"             "$ROOT/scripts/training/train_bottom_predictor.py"
run_step "4. Generating bottom predictions"          "$ROOT/scripts/utils/generate_bottom_predictions.py"
run_step "5. Evaluating crash detection"             "$ROOT/scripts/evaluation/evaluate_crash_detection.py"

echo -e "${G}✓ Pipeline complete${N}\n"

# ---- dashboard ------------------------------------------------------------
echo -e "${Y}[3/3] Launching dashboard...${N}"
echo -e "${G}→ http://localhost:8501${N}"
echo -e "${C}(Ctrl+C to stop)${N}\n"
exec "$STREAMLIT" run "$ROOT/src/dashboard/app.py"
