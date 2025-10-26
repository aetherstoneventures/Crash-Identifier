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

if [ ! -d "venv" ]; then
    echo "  Creating new virtual environment..."
    python3 -m venv venv
    echo "  Virtual environment created at: venv"
else
    echo "  Virtual environment already exists"
fi

# Activate virtual environment
echo "  Activating virtual environment..."
source venv/bin/activate

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
echo -e "${YELLOW}[STEP 5/7] Running the full pipeline...${NC}\n"

echo -e "${BLUE}--- Step 1: Running Data Backfill ---${NC}"
python3 "$PROJECT_ROOT/scripts/backfill_data.py"

echo -e "\n${BLUE}--- Step 2: Training Models & Feature Engineering ---${NC}"
python3 "$PROJECT_ROOT/scripts/train_models.py"

echo -e "\n${BLUE}--- Step 3: Running Tests ---${NC}"
python3 -m pytest "$PROJECT_ROOT/tests/" -v --tb=short

echo -e "\n${BLUE}--- Step 4: Starting Dashboard ---${NC}"
echo -e "${GREEN}Dashboard will start on http://localhost:8501${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the dashboard${NC}\n"

streamlit run "$PROJECT_ROOT/src/dashboard/app.py"

# ============================================================================
# Cleanup on exit
# ============================================================================
echo -e "\n${BLUE}================================================================================${NC}"
echo -e "${GREEN}Pipeline execution complete!${NC}"
echo -e "${BLUE}================================================================================${NC}\n"

