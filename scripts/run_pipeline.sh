#!/bin/bash

################################################################################
# Market Crash & Bottom Prediction System - Complete Pipeline Setup & Run
# 
# This script:
# 1. Deactivates and deletes any existing virtual environments
# 2. Cleans up files that should not exist on a fresh run
# 3. Creates and activates a new virtual environment
# 4. Installs all requirements
# 5. Runs the full pipeline with the dashboard
#
# Usage: bash run_pipeline.sh
################################################################################

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Go to project root (parent of scripts directory)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}Market Crash & Bottom Prediction System - Pipeline Setup & Run${NC}"
echo -e "${BLUE}================================================================================${NC}\n"

# ============================================================================
# STEP 1: Deactivate any active virtual environment
# ============================================================================
echo -e "${YELLOW}[STEP 1/5] Deactivating any active virtual environments...${NC}"

if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "  Deactivating: $VIRTUAL_ENV"
    deactivate 2>/dev/null || true
fi

echo -e "${GREEN}✓ Virtual environment deactivated${NC}\n"

# ============================================================================
# STEP 2: Delete old virtual environments and clean up fresh-run files
# ============================================================================
echo -e "${YELLOW}[STEP 2/5] Cleaning up old venvs and fresh-run files...${NC}"

# Delete venv directories
for venv_dir in venv .venv env .env_venv; do
    if [ -d "$venv_dir" ]; then
        echo "  Deleting: $venv_dir"
        rm -rf "$venv_dir"
    fi
done

# Delete database and data files (fresh run)
echo "  Cleaning data directory..."
if [ -f "data/market_crash.db" ]; then
    rm -f "data/market_crash.db"
    echo "    Deleted: data/market_crash.db"
fi

# Clean data subdirectories
for subdir in logs models processed raw; do
    if [ -d "data/$subdir" ]; then
        find "data/$subdir" -type f -delete 2>/dev/null || true
        echo "    Cleaned: data/$subdir"
    fi
done

# Delete any cached Python files
echo "  Cleaning Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

echo -e "${GREEN}✓ Cleanup complete${NC}\n"

# ============================================================================
# STEP 3: Create and activate virtual environment
# ============================================================================
echo -e "${YELLOW}[STEP 3/5] Creating and activating virtual environment...${NC}"

# Find Python 3.9, 3.10, 3.11, or 3.12 (TensorFlow compatible)
# Priority: Homebrew versions first (isolated), then system versions
PYTHON_CMD=""

# List of Python commands to try (Homebrew paths first for isolation)
PYTHON_CANDIDATES=(
    "/opt/homebrew/bin/python3.11"
    "/opt/homebrew/bin/python3.12"
    "/opt/homebrew/bin/python3.10"
    "/opt/homebrew/bin/python3.9"
    "/usr/local/bin/python3.11"
    "/usr/local/bin/python3.12"
    "/usr/local/bin/python3.10"
    "/usr/local/bin/python3.9"
    "python3.11"
    "python3.12"
    "python3.10"
    "python3.9"
    "python3"
)

echo "  Searching for compatible Python (3.9-3.12)..."

for py_version in "${PYTHON_CANDIDATES[@]}"; do
    if command -v $py_version &> /dev/null; then
        # Get version and extract major.minor
        PY_VER=$($py_version --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
        if [ -z "$PY_VER" ]; then
            PY_VER=$($py_version --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
        fi

        if [ -n "$PY_VER" ]; then
            PY_MAJOR=$(echo $PY_VER | cut -d. -f1)
            PY_MINOR=$(echo $PY_VER | cut -d. -f2)

            # Check if version is 3.9-3.12
            if [ "$PY_MAJOR" = "3" ] && [ "$PY_MINOR" -ge 9 ] && [ "$PY_MINOR" -le 12 ]; then
                PYTHON_CMD=$py_version
                echo "  ✓ Found compatible Python: $py_version (version $PY_VER)"
                break
            else
                echo "  ✗ Skipping $py_version (version $PY_VER - need 3.9-3.12)"
            fi
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo -e "${RED}ERROR: Python 3.9-3.12 required for TensorFlow compatibility${NC}"
    echo -e "${RED}No compatible Python version found.${NC}"
    echo ""
    echo -e "${YELLOW}To install Python 3.11 (recommended):${NC}"
    echo -e "${YELLOW}  brew install python@3.11${NC}"
    echo ""
    echo -e "${YELLOW}After installation, run this script again.${NC}"
    exit 1
fi

if [ ! -d "venv" ]; then
    echo "  Creating new virtual environment with $PYTHON_CMD..."
    $PYTHON_CMD -m venv venv
    echo "  Virtual environment created at: venv"
else
    echo "  Virtual environment already exists"
fi

# Activate virtual environment
echo "  Activating virtual environment..."
source venv/bin/activate

# Verify Python version in venv
VENV_PY_VER=$(python --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
echo "  Virtual environment Python version: $VENV_PY_VER"

echo -e "${GREEN}✓ Virtual environment activated${NC}\n"

# ============================================================================
# STEP 4: Install requirements
# ============================================================================
echo -e "${YELLOW}[STEP 4/5] Installing requirements...${NC}"

echo "  Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1

echo "  Installing requirements from requirements.txt..."
pip install -r requirements.txt

echo -e "${GREEN}✓ Requirements installed${NC}\n"

# ============================================================================
# STEP 5: Run the full pipeline
# ============================================================================
echo -e "${YELLOW}[STEP 5/5] Running the full pipeline...${NC}\n"

# Use venv python (properly quoted for paths with spaces)
PYTHON="${PROJECT_ROOT}/venv/bin/python3"

echo -e "${BLUE}--- Step 1: Collecting Market and Economic Data ---${NC}"
"${PYTHON}" "${PROJECT_ROOT}/scripts/data/collect_data.py"

echo -e "\n${BLUE}--- Step 2: Populating Historical Crashes ---${NC}"
"${PYTHON}" "${PROJECT_ROOT}/scripts/data/populate_crash_events.py"

echo -e "\n${BLUE}--- Step 3: Training Advanced Models (LSTM + XGBoost + Statistical V3) ---${NC}"
"${PYTHON}" "${PROJECT_ROOT}/scripts/training/train_advanced_models.py"

echo -e "\n${BLUE}--- Step 3b: Training Statistical Model V3 (Standalone) ---${NC}"
"${PYTHON}" "${PROJECT_ROOT}/scripts/training/train_statistical_model_v3.py"

echo -e "\n${BLUE}--- Step 3c: Training Bottom Prediction Model ---${NC}"
"${PYTHON}" "${PROJECT_ROOT}/scripts/training/train_bottom_predictor.py"

echo -e "\n${BLUE}--- Step 4: Generating Crash Probability Predictions (V5 ML + V2 Statistical) ---${NC}"
"${PYTHON}" "${PROJECT_ROOT}/scripts/utils/generate_predictions_v5.py"

echo -e "\n${BLUE}--- Step 4b: Generating Bottom Predictions (Optimal Re-Entry Timing) ---${NC}"
"${PYTHON}" "${PROJECT_ROOT}/scripts/utils/generate_bottom_predictions.py"

echo -e "\n${BLUE}--- Step 5: Evaluating Crash Detection ---${NC}"
"${PYTHON}" "${PROJECT_ROOT}/scripts/evaluation/evaluate_crash_detection.py"

echo -e "\n${BLUE}--- Step 6: Starting Dashboard ---${NC}"
echo -e "${GREEN}Dashboard will start on http://localhost:8501${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the dashboard${NC}\n"

"${PROJECT_ROOT}/venv/bin/streamlit" run "${PROJECT_ROOT}/src/dashboard/app.py"

# ============================================================================
# Cleanup on exit
# ============================================================================
echo -e "\n${BLUE}================================================================================${NC}"
echo -e "${GREEN}Pipeline execution complete!${NC}"
echo -e "${BLUE}================================================================================${NC}\n"

