"""Pytest configuration and fixtures."""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pytest


@pytest.fixture(scope="session")
def project_root_fixture():
    """Provide project root path."""
    return project_root


@pytest.fixture(scope="session")
def data_dir():
    """Provide data directory path."""
    return project_root / "data"


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide test data directory path."""
    return project_root / "tests" / "data"

