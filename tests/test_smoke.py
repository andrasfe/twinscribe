"""
Smoke tests to verify the test infrastructure is working correctly.

These tests verify:
- pytest is properly configured
- asyncio support works
- fixtures are accessible
- mocking infrastructure is functional
"""

import pytest
from pathlib import Path


class TestInfrastructure:
    """Tests to verify the testing infrastructure itself."""

    def test_pytest_works(self):
        """Verify basic pytest functionality."""
        assert True

    def test_fixtures_directory_exists(self, fixtures_dir: Path):
        """Verify the fixtures directory is accessible."""
        assert fixtures_dir.exists()
        assert fixtures_dir.is_dir()

    def test_sample_codebase_exists(self, sample_codebase_dir: Path):
        """Verify the sample codebase fixture is accessible."""
        assert sample_codebase_dir.exists()
        assert sample_codebase_dir.is_dir()

    def test_project_root_fixture(self, project_root: Path):
        """Verify project root fixture returns correct path."""
        assert project_root.exists()
        assert (project_root / "pyproject.toml").exists()


class TestAsyncSupport:
    """Tests to verify async test support."""

    @pytest.mark.asyncio
    async def test_async_function(self):
        """Verify async test functions work."""
        import asyncio
        await asyncio.sleep(0.001)
        assert True

    @pytest.mark.asyncio
    async def test_async_fixture(self, async_mock):
        """Verify async mock fixture works."""
        async_mock.return_value = "test_value"
        result = await async_mock()
        assert result == "test_value"


class TestMockFixtures:
    """Tests to verify mock fixtures work correctly."""

    def test_mock_openai_client(self, mock_openai_client):
        """Verify OpenAI mock client fixture."""
        assert mock_openai_client is not None
        assert hasattr(mock_openai_client.chat.completions, 'create')

    def test_mock_anthropic_client(self, mock_anthropic_client):
        """Verify Anthropic mock client fixture."""
        assert mock_anthropic_client is not None
        assert hasattr(mock_anthropic_client.messages, 'create')

    def test_mock_documenter_response(self, mock_documenter_response: dict):
        """Verify documenter response fixture has expected structure."""
        assert "component_id" in mock_documenter_response
        assert "documentation" in mock_documenter_response
        assert "call_graph" in mock_documenter_response
        assert "metadata" in mock_documenter_response

    def test_mock_validation_response(self, mock_validation_response: dict):
        """Verify validation response fixture has expected structure."""
        assert "component_id" in mock_validation_response
        assert "validation_result" in mock_validation_response
        assert "completeness" in mock_validation_response
        assert "call_graph_accuracy" in mock_validation_response

    def test_mock_comparison_response(self, mock_comparison_response: dict):
        """Verify comparison response fixture has expected structure."""
        assert "comparison_id" in mock_comparison_response
        assert "summary" in mock_comparison_response
        assert "discrepancies" in mock_comparison_response
        assert "convergence_status" in mock_comparison_response


class TestSampleCodeFixtures:
    """Tests to verify sample code fixtures."""

    def test_sample_python_function(self, sample_python_function: str):
        """Verify sample function fixture contains valid Python code."""
        assert "def calculate_total" in sample_python_function
        assert "items: list[dict]" in sample_python_function
        assert "return subtotal" in sample_python_function

    def test_sample_python_class(self, sample_python_class: str):
        """Verify sample class fixture contains valid Python code."""
        assert "class DataProcessor:" in sample_python_class
        assert "def __init__" in sample_python_class
        assert "def process" in sample_python_class

    def test_sample_python_module(self, sample_python_module: str):
        """Verify sample module fixture contains multiple components."""
        assert "def helper_function" in sample_python_module
        assert "class Calculator:" in sample_python_module
        assert "class AdvancedCalculator" in sample_python_module


class TestCallGraphFixtures:
    """Tests to verify call graph fixtures."""

    def test_sample_call_graph_structure(self, sample_call_graph: dict):
        """Verify call graph fixture has expected structure."""
        assert "edges" in sample_call_graph
        assert "nodes" in sample_call_graph
        assert len(sample_call_graph["edges"]) > 0
        assert len(sample_call_graph["nodes"]) > 0

    def test_sample_call_graph_edges(self, sample_call_graph: dict):
        """Verify call graph edges have required fields."""
        for edge in sample_call_graph["edges"]:
            assert "caller" in edge
            assert "callee" in edge
            assert "call_site_line" in edge
            assert "call_type" in edge

    def test_empty_call_graph(self, empty_call_graph: dict):
        """Verify empty call graph fixture."""
        assert empty_call_graph["edges"] == []
        assert empty_call_graph["nodes"] == []


class TestConfigFixtures:
    """Tests to verify configuration fixtures."""

    def test_mock_config_structure(self, mock_config: dict):
        """Verify mock config has all required sections."""
        assert "codebase" in mock_config
        assert "models" in mock_config
        assert "convergence" in mock_config
        assert "beads" in mock_config
        assert "static_analysis" in mock_config
        assert "output" in mock_config

    def test_mock_config_models(self, mock_config: dict):
        """Verify mock config has model configuration."""
        models = mock_config["models"]
        assert "stream_a" in models
        assert "stream_b" in models
        assert "comparator" in models

    def test_mock_env_vars(self, mock_env_vars: dict):
        """Verify environment variables are set."""
        import os
        for key, value in mock_env_vars.items():
            assert os.environ.get(key) == value


class TestTempFixtures:
    """Tests to verify temporary directory fixtures."""

    def test_temp_codebase(self, temp_codebase: Path):
        """Verify temp codebase fixture creates directory with files."""
        assert temp_codebase.exists()
        assert temp_codebase.is_dir()
        assert (temp_codebase / "sample_module.py").exists()
        assert (temp_codebase / "__init__.py").exists()

    def test_temp_output_dir(self, temp_output_dir: Path):
        """Verify temp output directory fixture."""
        assert temp_output_dir.exists()
        assert temp_output_dir.is_dir()


@pytest.mark.unit
class TestMarkers:
    """Tests to verify pytest markers work."""

    def test_unit_marker(self):
        """Test with unit marker."""
        assert True

    @pytest.mark.slow
    def test_slow_marker(self):
        """Test with slow marker."""
        assert True

    @pytest.mark.llm
    def test_llm_marker(self):
        """Test with llm marker."""
        assert True

    @pytest.mark.beads
    def test_beads_marker(self):
        """Test with beads marker."""
        assert True

    @pytest.mark.static_analysis
    def test_static_analysis_marker(self):
        """Test with static_analysis marker."""
        assert True
