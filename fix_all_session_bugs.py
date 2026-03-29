#!/usr/bin/env python3
"""
Fix all database session context manager bugs across the codebase.
This script finds and fixes all instances of:
    session = db.get_session()
    
And replaces them with:
    with db.get_session() as session:
"""

import re
import os
from pathlib import Path

# Files to fix (from grep output)
FILES_TO_FIX = [
    "scripts/training/train_bottom_predictor.py",
    "scripts/training/train_statistical_model_v2.py",
    "scripts/utils/generate_bottom_predictions.py",
    "scripts/utils/generate_predictions_v5.py",
    "scripts/evaluation/evaluate_crash_detection.py",
    "scripts/evaluation/evaluate_bottom_predictions.py",
    "scripts/data/populate_crash_events.py",
    "src/dashboard/pages/bottom_predictions.py",
    "src/dashboard/pages/crash_predictions.py",
    "src/dashboard/pages/indicators.py",
    "src/dashboard/pages/overview.py",
]

def fix_file(filepath):
    """Fix database session bugs in a single file."""
    print(f"\nProcessing: {filepath}")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Pattern 1: session = db.get_session() followed by session.close()
    # This is the most common pattern
    pattern1 = r'(\s+)session = db\.get_session\(\)\s*\n(.*?)(\s+)session\.close\(\)'
    
    def replace_pattern1(match):
        indent = match.group(1)
        middle_code = match.group(2)
        
        # Re-indent the middle code
        lines = middle_code.split('\n')
        reindented_lines = []
        for line in lines:
            if line.strip():  # Only re-indent non-empty lines
                reindented_lines.append('    ' + line)  # Add 4 spaces
            else:
                reindented_lines.append(line)
        reindented_middle = '\n'.join(reindented_lines)
        
        return f'{indent}with db.get_session() as session:\n{reindented_middle}'
    
    content = re.sub(pattern1, replace_pattern1, content, flags=re.DOTALL)
    
    # Pattern 2: Simple session = db.get_session() without close
    # Replace with with statement (user needs to manually indent the rest)
    if 'session = db.get_session()' in content:
        print(f"  ⚠️  WARNING: File still has 'session = db.get_session()' - manual review needed")
    
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  ✅ Fixed")
        return True
    else:
        print(f"  ℹ️  No changes needed")
        return False

def main():
    print("=" * 80)
    print("FIXING ALL DATABASE SESSION CONTEXT MANAGER BUGS")
    print("=" * 80)
    
    fixed_count = 0
    for filepath in FILES_TO_FIX:
        if os.path.exists(filepath):
            if fix_file(filepath):
                fixed_count += 1
        else:
            print(f"\n⚠️  File not found: {filepath}")
    
    print("\n" + "=" * 80)
    print(f"✅ Fixed {fixed_count} files")
    print("=" * 80)
    print("\n⚠️  IMPORTANT: Some files may need manual review!")
    print("   Run the pipeline and check for any remaining errors.")

if __name__ == '__main__':
    main()

