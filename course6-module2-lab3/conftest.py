import sys
import threading
from pathlib import Path

import pytest

# Add project root to sys.path so tests can import from
# application/, infra/, interfaces/ without installing the package.
sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture
def blocking_barrier():
    """Yields a threading.Event that blocks a worker thread until released.

    Used by timeout tests that need a real concurrent.futures path to fire.
    Automatically released on test teardown so the ThreadPoolExecutor can
    shut down cleanly even if the test failed before manually setting the
    barrier.
    """
    barrier = threading.Event()
    yield barrier
    barrier.set()