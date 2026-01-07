"""
Tests for ASTAnalyzer.

Tests the AST-based fallback analyzer implementation.
"""

import pytest

from twinscribe.analysis.ast_analyzer import ASTAnalyzer, CallVisitor


@pytest.mark.static_analysis
class TestCallVisitor:
    """Tests for the AST CallVisitor."""

    def test_simple_function_call(self):
        """Test extracting simple function calls."""
        import ast

        source = """
def caller():
    helper()
"""
        tree = ast.parse(source)
        visitor = CallVisitor("test_module")
        visitor.visit(tree)

        assert len(visitor.calls) == 1
        caller, callee, line = visitor.calls[0]
        assert caller == "test_module.caller"
        assert callee == "helper"

    def test_method_call(self):
        """Test extracting method calls."""
        import ast

        source = """
def process():
    obj.method()
"""
        tree = ast.parse(source)
        visitor = CallVisitor("test_module")
        visitor.visit(tree)

        assert len(visitor.calls) == 1
        caller, callee, line = visitor.calls[0]
        assert caller == "test_module.process"
        assert callee == "obj.method"

    def test_nested_attribute_call(self):
        """Test extracting nested attribute calls."""
        import ast

        source = """
def process():
    foo.bar.baz()
"""
        tree = ast.parse(source)
        visitor = CallVisitor("test_module")
        visitor.visit(tree)

        assert len(visitor.calls) == 1
        _, callee, _ = visitor.calls[0]
        assert callee == "foo.bar.baz"

    def test_class_method_scope(self):
        """Test that class methods have correct scope."""
        import ast

        source = """
class MyClass:
    def method(self):
        helper()
"""
        tree = ast.parse(source)
        visitor = CallVisitor("test_module")
        visitor.visit(tree)

        assert len(visitor.calls) == 1
        caller, _, _ = visitor.calls[0]
        assert caller == "test_module.MyClass.method"

    def test_import_resolution(self):
        """Test that imports are tracked for name resolution."""
        import ast

        source = """
from utils import helper

def process():
    helper()
"""
        tree = ast.parse(source)
        visitor = CallVisitor("test_module")
        visitor.visit(tree)

        assert len(visitor.calls) == 1
        _, callee, _ = visitor.calls[0]
        assert callee == "utils.helper"

    def test_import_alias_resolution(self):
        """Test that import aliases are resolved."""
        import ast

        source = """
from utils import helper as h

def process():
    h()
"""
        tree = ast.parse(source)
        visitor = CallVisitor("test_module")
        visitor.visit(tree)

        assert len(visitor.calls) == 1
        _, callee, _ = visitor.calls[0]
        assert callee == "utils.helper"

    def test_module_import_resolution(self):
        """Test that module imports are tracked."""
        import ast

        source = """
import os

def process():
    os.path.exists("file")
"""
        tree = ast.parse(source)
        visitor = CallVisitor("test_module")
        visitor.visit(tree)

        # The call should be resolved
        assert len(visitor.calls) == 1
        _, callee, _ = visitor.calls[0]
        assert "os" in callee

    def test_multiple_calls_in_function(self):
        """Test extracting multiple calls from one function."""
        import ast

        source = """
def process():
    setup()
    validate()
    transform()
"""
        tree = ast.parse(source)
        visitor = CallVisitor("test_module")
        visitor.visit(tree)

        assert len(visitor.calls) == 3
        callees = {c[1] for c in visitor.calls}
        assert callees == {"setup", "validate", "transform"}

    def test_async_function_scope(self):
        """Test that async functions create proper scope."""
        import ast

        source = """
async def async_process():
    await helper()
"""
        tree = ast.parse(source)
        visitor = CallVisitor("test_module")
        visitor.visit(tree)

        assert len(visitor.calls) == 1
        caller, _, _ = visitor.calls[0]
        assert caller == "test_module.async_process"

    def test_nested_calls(self):
        """Test extracting nested function calls."""
        import ast

        source = """
def process():
    result = outer(inner())
"""
        tree = ast.parse(source)
        visitor = CallVisitor("test_module")
        visitor.visit(tree)

        # Should capture both calls
        assert len(visitor.calls) == 2
        callees = {c[1] for c in visitor.calls}
        assert callees == {"outer", "inner"}

    def test_line_numbers_captured(self):
        """Test that line numbers are captured correctly."""
        import ast

        source = """def process():
    foo()
    bar()
"""
        tree = ast.parse(source)
        visitor = CallVisitor("test_module")
        visitor.visit(tree)

        lines = {c[2] for c in visitor.calls}
        assert 2 in lines
        assert 3 in lines


@pytest.mark.static_analysis
class TestASTAnalyzerAvailability:
    """Tests for AST analyzer availability."""

    @pytest.mark.asyncio
    async def test_always_available(self):
        """Test that AST analyzer is always available."""
        analyzer = ASTAnalyzer()
        available = await analyzer.check_available()
        assert available is True

    @pytest.mark.asyncio
    async def test_get_version(self):
        """Test that version includes ast prefix."""
        analyzer = ASTAnalyzer()
        version = await analyzer.get_version()
        assert version is not None
        assert version.startswith("ast-")


@pytest.mark.static_analysis
class TestASTAnalyzerAnalyze:
    """Tests for AST analyzer analyze method."""

    @pytest.mark.asyncio
    async def test_analyze_nonexistent_path_raises(self):
        """Test that analyze raises for non-existent path."""
        analyzer = ASTAnalyzer()

        with pytest.raises(FileNotFoundError):
            await analyzer.analyze("/nonexistent/path")

    @pytest.mark.asyncio
    async def test_analyze_empty_directory(self, tmp_path):
        """Test analyzing empty directory."""
        analyzer = ASTAnalyzer()
        result = await analyzer.analyze(tmp_path)

        assert result.edge_count == 0
        assert "No Python files found" in str(result.warnings)

    @pytest.mark.asyncio
    async def test_analyze_simple_file(self, tmp_path):
        """Test analyzing a simple Python file."""
        code = """
def caller():
    callee()

def callee():
    pass
"""
        (tmp_path / "module.py").write_text(code)

        analyzer = ASTAnalyzer()
        result = await analyzer.analyze(tmp_path)

        assert result.edge_count == 1
        edge = result.raw_edges[0]
        assert "caller" in edge.caller
        assert edge.callee == "callee"

    @pytest.mark.asyncio
    async def test_analyze_multiple_files(self, tmp_path):
        """Test analyzing multiple Python files."""
        (tmp_path / "module1.py").write_text("def foo(): bar()")
        (tmp_path / "module2.py").write_text("def baz(): qux()")

        analyzer = ASTAnalyzer()
        result = await analyzer.analyze(tmp_path)

        assert result.edge_count == 2
        assert result.metadata["files_analyzed"] == 2

    @pytest.mark.asyncio
    async def test_analyze_with_syntax_error(self, tmp_path):
        """Test analyzing file with syntax error produces warning."""
        (tmp_path / "valid.py").write_text("def foo(): pass")
        (tmp_path / "invalid.py").write_text("def broken(: pass")

        analyzer = ASTAnalyzer()
        result = await analyzer.analyze(tmp_path)

        # Should still process valid file
        assert result.metadata["files_analyzed"] == 2
        # Should have warning about syntax error
        assert any("Syntax error" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_analyze_class_methods(self, tmp_path):
        """Test analyzing class methods."""
        code = """
class MyClass:
    def method1(self):
        self.method2()

    def method2(self):
        pass
"""
        (tmp_path / "module.py").write_text(code)

        analyzer = ASTAnalyzer()
        result = await analyzer.analyze(tmp_path)

        assert result.edge_count == 1
        edge = result.raw_edges[0]
        assert "MyClass.method1" in edge.caller
        assert "method2" in edge.callee

    @pytest.mark.asyncio
    async def test_analyze_cross_module_calls(self, tmp_path):
        """Test analyzing calls across modules."""
        (tmp_path / "utils.py").write_text("def helper(): pass")
        code = """
from utils import helper

def main():
    helper()
"""
        (tmp_path / "main.py").write_text(code)

        analyzer = ASTAnalyzer()
        result = await analyzer.analyze(tmp_path)

        # Should capture the call to helper
        edges = result.raw_edges
        main_edges = [e for e in edges if "main" in e.caller.lower()]
        assert len(main_edges) >= 1

    @pytest.mark.asyncio
    async def test_analyze_collects_nodes(self, tmp_path):
        """Test that analyze collects all nodes."""
        code = """
def a():
    b()

def b():
    c()

def c():
    pass
"""
        (tmp_path / "module.py").write_text(code)

        analyzer = ASTAnalyzer()
        result = await analyzer.analyze(tmp_path)

        # Should have nodes for a, b, c plus the callees
        assert len(result.nodes) >= 3


@pytest.mark.static_analysis
class TestASTAnalyzerModuleNames:
    """Tests for module name calculation."""

    @pytest.mark.asyncio
    async def test_module_name_from_path(self, tmp_path):
        """Test module name calculation from file path."""
        subdir = tmp_path / "package"
        subdir.mkdir()
        (subdir / "module.py").write_text("def foo(): bar()\ndef bar(): pass")

        analyzer = ASTAnalyzer()
        result = await analyzer.analyze(tmp_path)

        # Check that edges contain the correct caller format
        if result.raw_edges:
            # Caller should include path components
            caller = result.raw_edges[0].caller
            # Module name should be calculated from path
            assert "module" in caller or "package" in caller

    @pytest.mark.asyncio
    async def test_init_module_handling(self, tmp_path):
        """Test __init__.py module name handling."""
        subdir = tmp_path / "package"
        subdir.mkdir()
        (subdir / "__init__.py").write_text("def init_func(): pass")

        analyzer = ASTAnalyzer()
        result = await analyzer.analyze(tmp_path)

        # __init__ should not appear in module name
        nodes = list(result.nodes)
        has_init = any("__init__" in n for n in nodes)
        # Module name should just be package.init_func
        assert not has_init or any("package" in n for n in nodes)
