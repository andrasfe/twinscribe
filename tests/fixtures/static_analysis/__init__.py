"""
Static Analysis Fixtures

This package contains mock static analysis outputs for testing.
The fixtures simulate output from PyCG and other static analyzers.

Files:
- pycg_output.json: Sample PyCG call graph output
- pyan_output.json: Sample pyan3 output
- call_graph_ground_truth.json: Complete ground truth call graph
"""

import json
from pathlib import Path
from typing import Any


FIXTURES_DIR = Path(__file__).parent


def load_pycg_output() -> dict[str, Any]:
    """Load the PyCG output fixture."""
    with open(FIXTURES_DIR / "pycg_output.json") as f:
        return json.load(f)


def load_call_graph_ground_truth() -> dict[str, Any]:
    """Load the ground truth call graph fixture."""
    with open(FIXTURES_DIR / "call_graph_ground_truth.json") as f:
        return json.load(f)
