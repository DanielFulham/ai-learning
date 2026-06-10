"""Pytest configuration.

Ensures the project root is on sys.path so test modules can import application,
domain, infra, and interfaces packages without installing the project.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))