"""
AST-based Fallback Analyzer.

Provides a simpler AST-based call graph analyzer as fallback when PyCG
or other primary analyzers are not available. This analyzer uses Python's
built-in ast module and doesn't require external dependencies.

Limitations compared to PyCG:
- No inter-procedural analysis
- Limited dynamic dispatch resolution
- No handling of higher-order functions
- May miss some indirect calls

However, it provides a reasonable approximation for most direct calls
and is always available since it only uses the standard library.
"""

import ast
import logging
import time
from datetime import datetime
from pathlib import Path

from twinscribe.analysis.analyzer import (
    Analyzer,
    AnalyzerConfig,
    AnalyzerResult,
    AnalyzerType,
    Language,
    RawCallEdge,
)

logger = logging.getLogger(__name__)


class CallVisitor(ast.NodeVisitor):
    """AST visitor that extracts function/method calls.

    Visits an AST and collects all call expressions, tracking the
    current scope (module, class, function) to determine callers.
    """

    def __init__(self, module_name: str) -> None:
        """Initialize the call visitor.

        Args:
            module_name: Name of the module being analyzed
        """
        self.module_name = module_name
        self.calls: list[tuple[str, str, int]] = []  # (caller, callee, line)
        self._scope_stack: list[str] = [module_name]  # Stack of current scopes
        self._imports: dict[str, str] = {}  # Alias -> full name mapping

    @property
    def current_scope(self) -> str:
        """Get the current scope as a dotted name."""
        return ".".join(self._scope_stack)

    def visit_Import(self, node: ast.Import) -> None:
        """Track import statements for name resolution."""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self._imports[name] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Track from...import statements for name resolution."""
        module = node.module or ""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            full_name = f"{module}.{alias.name}" if module else alias.name
            self._imports[name] = full_name
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Enter a class scope."""
        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Enter a function scope and visit body."""
        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Enter an async function scope and visit body."""
        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        """Extract call information."""
        caller = self.current_scope
        callee = self._resolve_callee(node.func)

        if callee:
            self.calls.append((caller, callee, node.lineno))

        # Continue visiting nested calls
        self.generic_visit(node)

    def _resolve_callee(self, node: ast.expr) -> str | None:
        """Resolve the callee name from a call expression.

        Args:
            node: The function/method being called

        Returns:
            Resolved callee name or None if cannot resolve
        """
        if isinstance(node, ast.Name):
            # Simple name: foo()
            name = node.id
            # Check if it's an imported name
            if name in self._imports:
                return self._imports[name]
            return name

        elif isinstance(node, ast.Attribute):
            # Attribute access: obj.method()
            parts = self._resolve_attribute_chain(node)
            if parts:
                full_name = ".".join(parts)
                # Check if first part is an import
                if parts[0] in self._imports:
                    full_name = self._imports[parts[0]] + "." + ".".join(parts[1:])
                return full_name

        elif isinstance(node, ast.Subscript):
            # Subscript call: obj[key]()
            return self._resolve_callee(node.value)

        return None

    def _resolve_attribute_chain(self, node: ast.expr) -> list[str]:
        """Resolve a chain of attribute accesses.

        Args:
            node: AST node representing attribute access

        Returns:
            List of names in the chain, or empty if cannot resolve
        """
        if isinstance(node, ast.Attribute):
            base = self._resolve_attribute_chain(node.value)
            if base:
                base.append(node.attr)
                return base
            return [node.attr]

        elif isinstance(node, ast.Name):
            return [node.id]

        elif isinstance(node, ast.Call):
            # Method chaining: foo().bar()
            # We track the outer call but lose inner chain
            return []

        return []


class ASTAnalyzer(Analyzer):
    """AST-based analyzer for Python call graphs.

    Uses Python's built-in ast module to extract call relationships.
    This is a simpler but always-available alternative to PyCG.

    This analyzer:
    - Tracks direct function and method calls
    - Resolves imports to full module names
    - Handles class and function scopes
    - Works with any Python version

    Limitations:
    - No dynamic dispatch resolution
    - Limited handling of indirect calls
    - No inter-procedural analysis
    - May produce false positives for dynamic calls

    Usage:
        analyzer = ASTAnalyzer()
        if await analyzer.check_available():
            result = await analyzer.analyze(Path("/path/to/codebase"))
    """

    # Use a custom analyzer type indicator
    AST_ANALYZER_TYPE = AnalyzerType.PYCG  # Reuse PYCG type for compatibility

    def __init__(self, config: AnalyzerConfig | None = None) -> None:
        """Initialize the AST analyzer.

        Args:
            config: Analyzer configuration
        """
        if config is None:
            config = AnalyzerConfig(
                analyzer_type=self.AST_ANALYZER_TYPE,
                language=Language.PYTHON,
                include_patterns=["**/*.py"],
                exclude_patterns=[
                    "**/test_*",
                    "**/tests/**",
                    "**/__pycache__/**",
                    "**/venv/**",
                    "**/.venv/**",
                ],
            )
        super().__init__(config)

    async def check_available(self) -> bool:
        """Check if AST analyzer is available.

        Always returns True since ast is a standard library module.

        Returns:
            True (always available)
        """
        return True

    async def get_version(self) -> str | None:
        """Get analyzer version.

        Returns:
            Python version since this uses stdlib ast
        """
        import sys

        return f"ast-{sys.version_info.major}.{sys.version_info.minor}"

    async def analyze(self, codebase_path: Path) -> AnalyzerResult:
        """Run AST-based analysis on the codebase.

        Parses each Python file and extracts call relationships.

        Args:
            codebase_path: Path to the codebase root

        Returns:
            AnalyzerResult with raw call edges

        Raises:
            FileNotFoundError: If codebase_path doesn't exist
        """
        # Ensure codebase_path is a Path object
        if not isinstance(codebase_path, Path):
            codebase_path = Path(codebase_path)

        if not codebase_path.exists():
            raise FileNotFoundError(f"Codebase path not found: {codebase_path}")

        start_time = time.time()
        logger.info(f"Starting AST analysis on {codebase_path}")

        # Get files to analyze
        files = self._filter_files(codebase_path)
        if not files:
            logger.warning(f"No Python files found in {codebase_path}")
            return AnalyzerResult(
                analyzer_type=self.AST_ANALYZER_TYPE,
                codebase_path=str(codebase_path),
                warnings=["No Python files found matching patterns"],
            )

        logger.debug(f"Found {len(files)} Python files to analyze")

        # Analyze each file
        all_edges: list[RawCallEdge] = []
        all_nodes: set[str] = set()
        warnings: list[str] = []

        for file_path in files:
            try:
                edges, nodes = self._analyze_file(file_path, codebase_path)
                all_edges.extend(edges)
                all_nodes.update(nodes)
            except SyntaxError as e:
                warnings.append(f"Syntax error in {file_path}: {e}")
                logger.warning(f"Syntax error in {file_path}: {e}")
            except Exception as e:
                warnings.append(f"Failed to analyze {file_path}: {e}")
                logger.warning(f"Failed to analyze {file_path}: {e}")

        execution_time = time.time() - start_time
        logger.info(
            f"AST analysis complete: {len(all_edges)} edges, "
            f"{len(all_nodes)} nodes in {execution_time:.2f}s"
        )

        return AnalyzerResult(
            analyzer_type=self.AST_ANALYZER_TYPE,
            codebase_path=str(codebase_path),
            raw_edges=all_edges,
            nodes=all_nodes,
            execution_time_seconds=execution_time,
            timestamp=datetime.utcnow(),
            warnings=warnings,
            metadata={
                "files_analyzed": len(files),
                "analyzer": "ast_fallback",
            },
        )

    def _analyze_file(
        self,
        file_path: Path,
        codebase_root: Path,
    ) -> tuple[list[RawCallEdge], set[str]]:
        """Analyze a single Python file.

        Args:
            file_path: Path to the file
            codebase_root: Root of the codebase for module name calculation

        Returns:
            Tuple of (edges, nodes)
        """
        # Read and parse the file
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))

        # Calculate module name from path
        module_name = self._path_to_module(file_path, codebase_root)

        # Visit the AST
        visitor = CallVisitor(module_name)
        visitor.visit(tree)

        # Convert to RawCallEdge
        edges = []
        nodes = set()

        for caller, callee, line in visitor.calls:
            edges.append(
                RawCallEdge(
                    caller=caller,
                    callee=callee,
                    line_number=line,
                )
            )
            nodes.add(caller)
            nodes.add(callee)

        return edges, nodes

    def _path_to_module(self, file_path: Path, codebase_root: Path) -> str:
        """Convert a file path to a module name.

        Args:
            file_path: Path to the Python file
            codebase_root: Root of the codebase

        Returns:
            Module name in dotted notation
        """
        try:
            relative = file_path.relative_to(codebase_root)
        except ValueError:
            relative = file_path

        # Convert path to module name
        parts = list(relative.parts)

        # Remove .py extension from last part
        if parts and parts[-1].endswith(".py"):
            parts[-1] = parts[-1][:-3]

        # Handle __init__.py
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]

        return ".".join(parts)

    def parse_output(self, raw_output: str) -> list[RawCallEdge]:
        """Parse output (not used for AST analyzer).

        The AST analyzer generates edges directly, so this method
        is not used but must be implemented.

        Args:
            raw_output: Raw output string

        Returns:
            Empty list (not applicable)
        """
        # AST analyzer doesn't use external output
        return []


def create_ast_analyzer(config: AnalyzerConfig | None = None) -> ASTAnalyzer:
    """Factory function for creating AST analyzer.

    Args:
        config: Optional configuration

    Returns:
        Configured ASTAnalyzer instance
    """
    return ASTAnalyzer(config)
