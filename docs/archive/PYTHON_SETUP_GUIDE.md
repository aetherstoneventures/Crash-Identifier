# Python Setup Guide - Isolated Virtual Environment

## Problem Solved ✅

**Issue**: System Python 3.13 is incompatible with TensorFlow 2.15.0

**Solution**: Installed Python 3.11 via Homebrew for project isolation

---

## What Was Done

### 1. Installed Python 3.11 via Homebrew
```bash
brew install python@3.11
```

**Result**: Python 3.11.14 installed at `/opt/homebrew/bin/python3.11`

### 2. Updated Pipeline Script
**File**: `scripts/run_pipeline.sh`

**Changes**:
- Prioritizes Homebrew Python installations (isolated from system)
- Searches in this order:
  1. `/opt/homebrew/bin/python3.11` ← **USED** (isolated)
  2. `/opt/homebrew/bin/python3.12`
  3. Other Homebrew versions
  4. System Python (only if compatible)

**Benefits**:
- ✅ **Isolated**: venv uses Homebrew Python, not system Python
- ✅ **Safe**: System Python 3.13 remains untouched
- ✅ **Compatible**: Python 3.11.14 works with TensorFlow 2.15.0
- ✅ **Reproducible**: Same Python version across all runs

---

## How Virtual Environment Isolation Works

### Your System Now Has:

1. **System Python 3.13.7** (at `/usr/bin/python3`)
   - Used by macOS system tools
   - **NOT touched by this project**
   - Remains at version 3.13.7

2. **Homebrew Python 3.13** (at `/opt/homebrew/bin/python3`)
   - Installed via `brew install python@3.13`
   - **NOT used by this project**

3. **Homebrew Python 3.11** (at `/opt/homebrew/bin/python3.11`)
   - Installed via `brew install python@3.11`
   - **USED by this project's venv** ✅
   - Isolated from system

4. **Project Virtual Environment** (at `./venv/`)
   - Created from Homebrew Python 3.11
   - Contains all project dependencies
   - **Completely isolated** from system Python
   - Only active when you run the pipeline

---

## Verification

### Check What Python Is Used:

```bash
# System Python (untouched)
/usr/bin/python3 --version
# Output: Python 3.13.7

# Homebrew Python 3.13 (not used by project)
/opt/homebrew/bin/python3 --version
# Output: Python 3.13.x

# Homebrew Python 3.11 (used by project)
/opt/homebrew/bin/python3.11 --version
# Output: Python 3.11.14

# Project venv Python (when activated)
source venv/bin/activate
python --version
# Output: Python 3.11.14
```

### Pipeline Output:
```
[STEP 3/5] Creating and activating virtual environment...
  Searching for compatible Python (3.9-3.12)...
  ✓ Found compatible Python: /opt/homebrew/bin/python3.11 (version 3.11.14)
  Creating new virtual environment with /opt/homebrew/bin/python3.11...
  Virtual environment Python version: 3.11.14
✓ Virtual environment activated
```

---

## Why This Approach Is Best

### 1. **System Python Isolation** ✅
- System Python 3.13 remains untouched
- macOS system tools continue to work
- No risk of breaking system dependencies

### 2. **Project Isolation** ✅
- Virtual environment uses dedicated Python 3.11
- All dependencies installed in `./venv/`
- No pollution of system Python packages
- Easy to delete and recreate

### 3. **Reproducibility** ✅
- Same Python version (3.11.14) every time
- Same dependencies (pinned in requirements.txt)
- Works on any machine with Homebrew Python 3.11

### 4. **Multiple Projects** ✅
- Each project can have its own Python version
- No conflicts between projects
- Easy to manage different TensorFlow versions

---

## How to Run the Pipeline

### Simple Method (Recommended):
```bash
# Just run the pipeline script
bash scripts/run_pipeline.sh
```

**What it does**:
1. Finds Homebrew Python 3.11
2. Creates isolated venv with Python 3.11
3. Installs all dependencies in venv
4. Runs the complete pipeline
5. Launches dashboard

### Manual Method (for development):
```bash
# Create venv manually
/opt/homebrew/bin/python3.11 -m venv venv

# Activate venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run individual scripts
python scripts/data/collect_data.py
python scripts/training/train_advanced_models.py
streamlit run src/dashboard/app.py
```

---

## Troubleshooting

### Q: What if I upgrade system Python to 3.14?
**A**: No problem! The venv uses Homebrew Python 3.11, which is isolated.

### Q: Can I use a different Python version?
**A**: Yes, install it via Homebrew:
```bash
brew install python@3.12
```
The script will automatically find and use it.

### Q: How do I delete the venv?
**A**: Just delete the folder:
```bash
rm -rf venv
```
Run the pipeline script again to recreate it.

### Q: How do I update dependencies?
**A**: Update `requirements.txt`, then:
```bash
rm -rf venv
bash scripts/run_pipeline.sh
```

### Q: Can I use this venv in my IDE?
**A**: Yes! Point your IDE to:
```
/path/to/project/venv/bin/python
```

---

## Summary

✅ **System Python 3.13**: Untouched, used by macOS
✅ **Homebrew Python 3.11**: Installed, used by project venv
✅ **Project venv**: Isolated, contains all dependencies
✅ **Pipeline**: Works perfectly with Python 3.11.14
✅ **TensorFlow 2.15.0**: Compatible with Python 3.11

**Your system is safe, and the project is isolated!** 🎉

---

## Quick Reference

| Python | Location | Used By | Version |
|--------|----------|---------|---------|
| System Python | `/usr/bin/python3` | macOS | 3.13.7 |
| Homebrew Python | `/opt/homebrew/bin/python3` | Not used | 3.13.x |
| Homebrew Python 3.11 | `/opt/homebrew/bin/python3.11` | **Project venv** | 3.11.14 |
| Project venv | `./venv/bin/python` | **Pipeline** | 3.11.14 |

**To run the pipeline**:
```bash
bash scripts/run_pipeline.sh
```

**That's it!** The script handles everything automatically.

