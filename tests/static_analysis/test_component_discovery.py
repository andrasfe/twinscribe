"""
Tests for the ComponentDiscovery module.

Tests component discovery functionality including:
- File filtering based on include/exclude patterns
- Component extraction (functions, methods, classes)
- Source code extraction
- Topological ordering based on call graph
"""

from pathlib import Path

import pytest

from twinscribe.analysis.component_discovery import (
    ComponentDiscovery,
    ComponentVisitor,
    DiscoveryResult,
    create_component_discovery,
    discover_components,
)
from twinscribe.models.base import ComponentType
from twinscribe.models.call_graph import CallEdge, CallGraph

# Test fixture path
FIXTURES_PATH = Path(__file__).parent.parent / "fixtures" / "sample_codebase"


class TestComponentVisitor:
    """Tests for ComponentVisitor AST visitor."""

    def test_extracts_function(self, tmp_path: Path) -> None:
        """Test extraction of a standalone function."""
        source = '''
def greet(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}!"
'''
        file_path = tmp_path / "test.py"
        file_path.write_text(source)
        source_lines = source.splitlines()

        visitor = ComponentVisitor(
            module_name="test",
            file_path="test.py",
            source_lines=source_lines,
            codebase_root=tmp_path,
        )

        import ast

        tree = ast.parse(source)
        visitor.visit(tree)

        assert len(visitor.components) == 1
        comp = visitor.components[0]
        assert comp.component_id == "test.greet"
        assert comp.name == "greet"
        assert comp.type == ComponentType.FUNCTION
        assert comp.existing_docstring == "Say hello."
        assert "def greet(name: str) -> str" in comp.signature

    def test_extracts_class_with_methods(self, tmp_path: Path) -> None:
        """Test extraction of a class with methods."""
        source = '''
class Calculator:
    """A simple calculator."""

    def __init__(self, precision: int = 2) -> None:
        """Initialize."""
        self.precision = precision

    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        return a + b

    @staticmethod
    def identity(x: float) -> float:
        """Return the input."""
        return x

    @classmethod
    def from_default(cls) -> "Calculator":
        """Create default calculator."""
        return cls()

    @property
    def description(self) -> str:
        """Get description."""
        return "calculator"
'''
        file_path = tmp_path / "calc.py"
        file_path.write_text(source)
        source_lines = source.splitlines()

        visitor = ComponentVisitor(
            module_name="calc",
            file_path="calc.py",
            source_lines=source_lines,
            codebase_root=tmp_path,
        )

        import ast

        tree = ast.parse(source)
        visitor.visit(tree)

        # Should have: class + __init__ + add + identity + from_default + description
        assert len(visitor.components) == 6

        comp_ids = {c.component_id for c in visitor.components}
        assert "calc.Calculator" in comp_ids
        assert "calc.Calculator.__init__" in comp_ids
        assert "calc.Calculator.add" in comp_ids
        assert "calc.Calculator.identity" in comp_ids
        assert "calc.Calculator.from_default" in comp_ids
        assert "calc.Calculator.description" in comp_ids

        # Check component types
        type_map = {c.component_id: c.type for c in visitor.components}
        assert type_map["calc.Calculator"] == ComponentType.CLASS
        assert type_map["calc.Calculator.__init__"] == ComponentType.METHOD
        assert type_map["calc.Calculator.add"] == ComponentType.METHOD
        assert type_map["calc.Calculator.identity"] == ComponentType.STATICMETHOD
        assert type_map["calc.Calculator.from_default"] == ComponentType.CLASSMETHOD
        assert type_map["calc.Calculator.description"] == ComponentType.PROPERTY

    def test_extracts_async_function(self, tmp_path: Path) -> None:
        """Test extraction of an async function."""
        source = '''
async def fetch_data(url: str) -> dict:
    """Fetch data from URL."""
    return {}
'''
        file_path = tmp_path / "async_test.py"
        file_path.write_text(source)
        source_lines = source.splitlines()

        visitor = ComponentVisitor(
            module_name="async_test",
            file_path="async_test.py",
            source_lines=source_lines,
            codebase_root=tmp_path,
        )

        import ast

        tree = ast.parse(source)
        visitor.visit(tree)

        assert len(visitor.components) == 1
        comp = visitor.components[0]
        assert "async def" in comp.signature

    def test_extracts_source_code(self, tmp_path: Path) -> None:
        """Test that source code is correctly extracted."""
        source = '''def foo():
    """Foo function."""
    x = 1
    return x
'''
        file_path = tmp_path / "source_test.py"
        file_path.write_text(source)
        source_lines = source.splitlines()

        visitor = ComponentVisitor(
            module_name="source_test",
            file_path="source_test.py",
            source_lines=source_lines,
            codebase_root=tmp_path,
        )

        import ast

        tree = ast.parse(source)
        visitor.visit(tree)

        assert "source_test.foo" in visitor.source_code_map
        source_code = visitor.source_code_map["source_test.foo"]
        assert "def foo():" in source_code
        assert "return x" in source_code


class TestComponentDiscovery:
    """Tests for ComponentDiscovery class."""

    @pytest.mark.asyncio
    async def test_discover_from_sample_codebase(self) -> None:
        """Test discovery from the sample codebase fixture."""
        if not FIXTURES_PATH.exists():
            pytest.skip("Sample codebase fixtures not found")

        discovery = ComponentDiscovery(
            codebase_path=FIXTURES_PATH,
            include_private=False,
        )

        result = await discovery.discover()

        # Should find components
        assert len(result.components) > 0

        # Check for expected components from calculator.py
        comp_ids = {c.component_id for c in result.components}

        # Note: component IDs will include module path from fixtures dir
        # Look for Calculator-related components
        calculator_comps = [cid for cid in comp_ids if "Calculator" in cid]
        assert len(calculator_comps) > 0, "Should find Calculator class"

        # Check source code was extracted
        assert len(result.source_code_map) > 0

        # Check processing order was generated
        assert len(result.processing_order) == len(result.components)

    @pytest.mark.asyncio
    async def test_discover_with_call_graph(self, tmp_path: Path) -> None:
        """Test discovery with call graph for topological ordering."""
        # Create simple test files
        (tmp_path / "module_a.py").write_text('''
def func_a():
    """A function that calls func_b."""
    func_b()

def func_b():
    """B function called by A."""
    pass
''')

        discovery = ComponentDiscovery(codebase_path=tmp_path)

        # Create a call graph showing func_a calls func_b
        call_graph = CallGraph(
            edges=[
                CallEdge(caller="module_a.func_a", callee="module_a.func_b"),
            ],
            source="test",
        )

        result = await discovery.discover(call_graph=call_graph)

        # Both functions should be found
        comp_ids = {c.component_id for c in result.components}
        assert "module_a.func_a" in comp_ids
        assert "module_a.func_b" in comp_ids

        # func_b should come before func_a in processing order
        # (dependencies before dependents)
        order = result.processing_order
        idx_a = order.index("module_a.func_a")
        idx_b = order.index("module_a.func_b")
        assert idx_b < idx_a, "func_b (callee) should be processed before func_a (caller)"

    @pytest.mark.asyncio
    async def test_exclude_patterns(self, tmp_path: Path) -> None:
        """Test that exclude patterns filter out files."""
        # Create main code and test code
        (tmp_path / "main.py").write_text("""
def main_func():
    pass
""")
        (tmp_path / "test_main.py").write_text("""
def test_main_func():
    pass
""")
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_unit.py").write_text("""
def test_unit():
    pass
""")

        discovery = ComponentDiscovery(
            codebase_path=tmp_path,
            # Pattern test_* matches files starting with test_ at any level
            # Pattern tests/** matches anything in tests directory
            exclude_patterns=["test_*", "**/test_*", "**/tests/**", "tests/**"],
        )

        result = await discovery.discover()

        comp_ids = {c.component_id for c in result.components}
        assert "main.main_func" in comp_ids
        assert "test_main.test_main_func" not in comp_ids
        assert "tests.test_unit.test_unit" not in comp_ids

    @pytest.mark.asyncio
    async def test_include_private(self, tmp_path: Path) -> None:
        """Test including/excluding private components."""
        (tmp_path / "private.py").write_text("""
def public_func():
    pass

def _private_func():
    pass

class PublicClass:
    pass

class _PrivateClass:
    pass
""")

        # Without private
        discovery = ComponentDiscovery(codebase_path=tmp_path, include_private=False)
        result = await discovery.discover()
        comp_ids = {c.component_id for c in result.components}

        assert "private.public_func" in comp_ids
        assert "private.PublicClass" in comp_ids
        assert "private._private_func" not in comp_ids
        assert "private._PrivateClass" not in comp_ids

        # With private
        discovery = ComponentDiscovery(codebase_path=tmp_path, include_private=True)
        result = await discovery.discover()
        comp_ids = {c.component_id for c in result.components}

        assert "private._private_func" in comp_ids
        assert "private._PrivateClass" in comp_ids

    @pytest.mark.asyncio
    async def test_handles_syntax_errors(self, tmp_path: Path) -> None:
        """Test that syntax errors are handled gracefully."""
        (tmp_path / "good.py").write_text("""
def good_func():
    pass
""")
        (tmp_path / "bad.py").write_text("""
def bad_func(
    # Missing closing paren - syntax error
""")

        discovery = ComponentDiscovery(codebase_path=tmp_path)
        result = await discovery.discover()

        # Should find the good function
        comp_ids = {c.component_id for c in result.components}
        assert "good.good_func" in comp_ids

        # Should have recorded error for bad file
        assert len(result.errors) > 0
        assert any("bad.py" in err for err in result.errors)

    @pytest.mark.asyncio
    async def test_nonexistent_path(self) -> None:
        """Test that nonexistent path raises error."""
        discovery = ComponentDiscovery(codebase_path="/nonexistent/path")

        with pytest.raises(FileNotFoundError):
            await discovery.discover()


class TestTopologicalOrdering:
    """Tests for topological ordering functionality."""

    @pytest.mark.asyncio
    async def test_handles_cycles(self, tmp_path: Path) -> None:
        """Test that cycles in call graph are handled."""
        (tmp_path / "cycle.py").write_text("""
def func_a():
    func_b()

def func_b():
    func_a()  # Creates cycle
""")

        discovery = ComponentDiscovery(codebase_path=tmp_path)

        # Create call graph with cycle
        call_graph = CallGraph(
            edges=[
                CallEdge(caller="cycle.func_a", callee="cycle.func_b"),
                CallEdge(caller="cycle.func_b", callee="cycle.func_a"),
            ],
            source="test",
        )

        result = await discovery.discover(call_graph=call_graph)

        # Should still return all components despite cycle
        assert len(result.components) == 2
        assert len(result.processing_order) == 2

    @pytest.mark.asyncio
    async def test_complex_dependency_order(self, tmp_path: Path) -> None:
        """Test ordering with complex dependencies."""
        (tmp_path / "deps.py").write_text('''
def leaf():
    """No dependencies."""
    pass

def mid_a():
    """Calls leaf."""
    leaf()

def mid_b():
    """Calls leaf."""
    leaf()

def top():
    """Calls mid_a and mid_b."""
    mid_a()
    mid_b()
''')

        # Graph: top -> mid_a -> leaf
        #        top -> mid_b -> leaf
        call_graph = CallGraph(
            edges=[
                CallEdge(caller="deps.top", callee="deps.mid_a"),
                CallEdge(caller="deps.top", callee="deps.mid_b"),
                CallEdge(caller="deps.mid_a", callee="deps.leaf"),
                CallEdge(caller="deps.mid_b", callee="deps.leaf"),
            ],
            source="test",
        )

        discovery = ComponentDiscovery(codebase_path=tmp_path)
        result = await discovery.discover(call_graph=call_graph)

        order = result.processing_order

        # leaf should come before mid_a and mid_b
        idx_leaf = order.index("deps.leaf")
        idx_mid_a = order.index("deps.mid_a")
        idx_mid_b = order.index("deps.mid_b")
        idx_top = order.index("deps.top")

        assert idx_leaf < idx_mid_a
        assert idx_leaf < idx_mid_b
        assert idx_mid_a < idx_top
        assert idx_mid_b < idx_top


class TestConvenienceFunctions:
    """Tests for convenience factory functions."""

    def test_create_component_discovery(self, tmp_path: Path) -> None:
        """Test factory function."""
        discovery = create_component_discovery(
            codebase_path=tmp_path,
            include_private=True,
        )

        assert isinstance(discovery, ComponentDiscovery)
        assert discovery._include_private is True

    @pytest.mark.asyncio
    async def test_discover_components_function(self, tmp_path: Path) -> None:
        """Test convenience discover function."""
        (tmp_path / "test.py").write_text("""
def test_func():
    pass
""")

        result = await discover_components(tmp_path)

        assert isinstance(result, DiscoveryResult)
        assert len(result.components) == 1
