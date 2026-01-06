"""
Discrepancy Scenario Fixtures

This package contains sample discrepancy scenarios for testing
the comparison and resolution logic.

Scenarios:
- call_graph_discrepancies.json: Call graph edge differences
- documentation_discrepancies.json: Documentation content differences
- mixed_discrepancies.json: Complex scenarios with multiple discrepancy types
"""

import json
from pathlib import Path
from typing import Any


FIXTURES_DIR = Path(__file__).parent


def load_scenario(name: str) -> dict[str, Any]:
    """
    Load a discrepancy scenario by name.

    Args:
        name: Scenario name (without .json extension)

    Returns:
        Scenario data dictionary
    """
    with open(FIXTURES_DIR / f"{name}.json") as f:
        return json.load(f)


def get_call_graph_scenarios() -> list[dict[str, Any]]:
    """Get all call graph discrepancy scenarios."""
    data = load_scenario("call_graph_discrepancies")
    return data.get("scenarios", [])


def get_documentation_scenarios() -> list[dict[str, Any]]:
    """Get all documentation discrepancy scenarios."""
    data = load_scenario("documentation_discrepancies")
    return data.get("scenarios", [])


def get_mixed_scenarios() -> list[dict[str, Any]]:
    """Get all mixed discrepancy scenarios."""
    data = load_scenario("mixed_discrepancies")
    return data.get("scenarios", [])
