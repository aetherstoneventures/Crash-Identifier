#!/bin/bash

################################################################################
# Market Crash & Bottom Prediction System - Dashboard Only
# 
# This script:
# 1. Activates the existing virtual environment
# 2. Runs the Streamlit dashboard
#
# Usage: bash run_dashboard.sh
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
echo -e "${BLUE}Market Crash & Bottom Prediction System - Dashboard${NC}"
echo -e "${BLUE}================================================================================${NC}\n"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo -e "${RED}✗ Virtual environment not found!${NC}"
    echo -e "${YELLOW}Please run: bash run_pipeline.sh${NC}\n"
    exit 1
fi

# Deactivate any active venv
if [[ -n "$VIRTUAL_ENV" ]]; then
    deactivate 2>/dev/null || true
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

echo -e "${GREEN}✓ Virtual environment activated${NC}\n"

# Check if database exists
if [ ! -f "data/market_crash.db" ]; then
    echo -e "${RED}✗ Database not found!${NC}"
    echo -e "${YELLOW}Please run: bash run_pipeline.sh${NC}\n"
    exit 1
fi

echo -e "${BLUE}--- Starting Dashboard ---${NC}"
echo -e "${GREEN}Dashboard will start on http://localhost:8501${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the dashboard${NC}\n"

streamlit run src/dashboard/app.py

