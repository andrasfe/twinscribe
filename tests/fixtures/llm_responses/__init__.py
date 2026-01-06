"""
LLM Response Fixtures

This package contains mock LLM responses for testing the documentation system
without making actual API calls. Each JSON file represents a specific scenario.

Files:
- documenter_responses.json: Sample documentation generation responses
- validator_responses.json: Sample validation responses
- comparator_responses.json: Sample comparison responses
- error_responses.json: Simulated error scenarios
"""

import json
from pathlib import Path
from typing import Any


FIXTURES_DIR = Path(__file__).parent


def load_fixture(name: str) -> dict[str, Any]:
    """
    Load a fixture JSON file by name.

    Args:
        name: Name of the fixture file (without .json extension)

    Returns:
        Parsed JSON content

    Raises:
        FileNotFoundError: If fixture file doesn't exist
    """
    path = FIXTURES_DIR / f"{name}.json"
    with open(path) as f:
        return json.load(f)


def get_documenter_response(component_id: str) -> dict[str, Any] | None:
    """
    Get a mock documenter response for a component.

    Args:
        component_id: Component ID to look up

    Returns:
        Mock response or None if not found
    """
    try:
        responses = load_fixture("documenter_responses")
        return responses.get(component_id)
    except FileNotFoundError:
        return None


def get_validator_response(component_id: str) -> dict[str, Any] | None:
    """
    Get a mock validator response for a component.

    Args:
        component_id: Component ID to look up

    Returns:
        Mock response or None if not found
    """
    try:
        responses = load_fixture("validator_responses")
        return responses.get(component_id)
    except FileNotFoundError:
        return None


def get_comparator_response(comparison_id: str) -> dict[str, Any] | None:
    """
    Get a mock comparator response.

    Args:
        comparison_id: Comparison ID to look up

    Returns:
        Mock response or None if not found
    """
    try:
        responses = load_fixture("comparator_responses")
        return responses.get(comparison_id)
    except FileNotFoundError:
        return None
