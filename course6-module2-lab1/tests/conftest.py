"""pytest configuration: ensures project root is on sys.path for imports.

Without this, `from infra.local_csv_loader import LocalCsvLoader` fails
because pytest doesn't add the project root automatically.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))