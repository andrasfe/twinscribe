"""
Component Discovery Module.

Discovers all documentable Python components (functions, methods, classes)
in a codebase using AST parsing and provides topological ordering based
on the call graph for processing by the documentation orchestrator.

This module bridges the gap between static analysis (call graph extraction)
and the documentation pipeline (component processing order).
"""

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path

from twinscribe.models.base import ComponentType
from twinscribe.models.call_graph import CallGraph
from twinscribe.models.components import Component, ComponentLocation

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryResult:
    """Result from component discovery.

    Attributes:
        components: All discovered components
        component_map: Map of component_id to Component
        processing_order: Component IDs in topological order
        source_code_map: Map of component_id to source code
        files_analyzed: Number of files analyzed
        errors: List of errors encountered during discovery
    """

    components: list[Component] = field(default_factory=list)
    component_map: dict[str, Component] = field(default_factory=dict)
    processing_order: list[str] = field(default_factory=list)
    source_code_map: dict[str, str] = field(default_factory=dict)
    files_analyzed: int = 0
    errors: list[str] = field(default_factory=list)


class ComponentVisitor(ast.NodeVisitor):
    """AST visitor that discovers documentable components.

    Visits an AST and collects all functions, methods, and classes
    along with their source code and location information.
    """

    def __init__(
        self,
        module_name: str,
        file_path: str,
        source_lines: list[str],
        codebase_root: Path,
    ) -> None:
        """Initialize the component visitor.

        Args:
            module_name: Fully qualified module name
            file_path: Relative path to the source file
            source_lines: Source code split into lines
            codebase_root: Root path of the codebase
        """
        self.module_name = module_name
        self.file_path = file_path
        self.source_lines = source_lines
        self.codebase_root = codebase_root
        self.components: list[Component] = []
        self.source_code_map: dict[str, str] = {}
        self._scope_stack: list[str] = []  # Stack of class/function names

    @property
    def current_scope(self) -> str:
        """Get the current scope as a dotted name."""
        if self._scope_stack:
            return f"{self.module_name}.{'.'.join(self._scope_stack)}"
        return self.module_name

    def _extract_source(self, node: ast.AST) -> str:
        """Extract source code for an AST node.

        Args:
            node: AST node to extract source from

        Returns:
            Source code string for the node
        """
        start_line = node.lineno - 1  # Convert to 0-indexed
        end_line = node.end_lineno if node.end_lineno else start_line + 1

        # Ensure bounds are valid
        start_line = max(0, start_line)
        end_line = min(len(self.source_lines), end_line)

        return "\n".join(self.source_lines[start_line:end_line])

    def _extract_docstring(self, node: ast.AST) -> str | None:
        """Extract docstring from a function or class node.

        Args:
            node: AST node (FunctionDef, AsyncFunctionDef, or ClassDef)

        Returns:
            Docstring string or None
        """
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            return node.body[0].value.value
        return None

    def _extract_signature(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        """Extract function signature string.

        Args:
            node: Function definition node

        Returns:
            Signature string like "def func(a: int, b: str = 'default') -> bool"
        """
        args = []

        # Regular arguments
        defaults_start = len(node.args.args) - len(node.args.defaults)
        for i, arg in enumerate(node.args.args):
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            if i >= defaults_start:
                default = node.args.defaults[i - defaults_start]
                arg_str += f" = {ast.unparse(default)}"
            args.append(arg_str)

        # *args
        if node.args.vararg:
            arg_str = f"*{node.args.vararg.arg}"
            if node.args.vararg.annotation:
                arg_str += f": {ast.unparse(node.args.vararg.annotation)}"
            args.append(arg_str)

        # Keyword-only arguments
        kw_defaults_dict = {i: d for i, d in enumerate(node.args.kw_defaults) if d is not None}
        for i, arg in enumerate(node.args.kwonlyargs):
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            if i in kw_defaults_dict:
                arg_str += f" = {ast.unparse(kw_defaults_dict[i])}"
            args.append(arg_str)

        # **kwargs
        if node.args.kwarg:
            arg_str = f"**{node.args.kwarg.arg}"
            if node.args.kwarg.annotation:
                arg_str += f": {ast.unparse(node.args.kwarg.annotation)}"
            args.append(arg_str)

        # Build signature
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        sig = f"{prefix} {node.name}({', '.join(args)})"

        if node.returns:
            sig += f" -> {ast.unparse(node.returns)}"

        return sig

    def _determine_component_type(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> ComponentType:
        """Determine the type of a function/method.

        Args:
            node: Function definition node

        Returns:
            ComponentType enum value
        """
        # Check if it's a method (inside a class)
        if self._scope_stack:
            # Check for decorators
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name):
                    if decorator.id == "staticmethod":
                        return ComponentType.STATICMETHOD
                    elif decorator.id == "classmethod":
                        return ComponentType.CLASSMETHOD
                    elif decorator.id == "property":
                        return ComponentType.PROPERTY
            return ComponentType.METHOD

        return ComponentType.FUNCTION

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit a class definition.

        Args:
            node: Class definition AST node
        """
        component_id = f"{self.current_scope}.{node.name}"

        # Create component
        component = Component(
            component_id=component_id,
            name=node.name,
            type=ComponentType.CLASS,
            location=ComponentLocation(
                file_path=self.file_path,
                line_start=node.lineno,
                line_end=node.end_lineno or node.lineno,
            ),
            signature=f"class {node.name}",
            parent_id=self.current_scope if self._scope_stack else None,
            existing_docstring=self._extract_docstring(node),
            is_public=not node.name.startswith("_"),
        )

        self.components.append(component)
        self.source_code_map[component_id] = self._extract_source(node)

        # Visit children
        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit a function/method definition.

        Args:
            node: Function definition AST node
        """
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit an async function/method definition.

        Args:
            node: Async function definition AST node
        """
        self._visit_function(node)

    def _visit_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        """Common handler for function definitions.

        Args:
            node: Function definition AST node
        """
        component_id = f"{self.current_scope}.{node.name}"
        component_type = self._determine_component_type(node)

        # Create component
        component = Component(
            component_id=component_id,
            name=node.name,
            type=component_type,
            location=ComponentLocation(
                file_path=self.file_path,
                line_start=node.lineno,
                line_end=node.end_lineno or node.lineno,
            ),
            signature=self._extract_signature(node),
            parent_id=self.current_scope if self._scope_stack else None,
            existing_docstring=self._extract_docstring(node),
            is_public=not node.name.startswith("_"),
        )

        self.components.append(component)
        self.source_code_map[component_id] = self._extract_source(node)

        # Visit nested functions/classes
        self._scope_stack.append(node.name)
        self.generic_visit(node)
        self._scope_stack.pop()


class ComponentDiscovery:
    """Discovers documentable components in a Python codebase.

    Parses Python files to find all functions, methods, and classes,
    extracts their source code and metadata, and computes a topological
    ordering based on the call graph for efficient documentation.

    Usage:
        discovery = ComponentDiscovery(
            codebase_path="/path/to/project",
            include_patterns=["**/*.py"],
            exclude_patterns=["**/tests/**"],
        )

        # Discover components
        result = await discovery.discover()

        # Or with an existing call graph
        result = await discovery.discover(call_graph=existing_call_graph)
    """

    def __init__(
        self,
        codebase_path: str | Path,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        include_private: bool = False,
    ) -> None:
        """Initialize the component discovery.

        Args:
            codebase_path: Path to the codebase root
            include_patterns: Glob patterns for files to include
            exclude_patterns: Glob patterns for files to exclude
            include_private: Whether to include private components
        """
        self._codebase_path = Path(codebase_path)
        self._include_patterns = include_patterns or ["**/*.py"]
        self._exclude_patterns = exclude_patterns or [
            "**/test_*",
            "**/tests/**",
            "**/__pycache__/**",
            "**/venv/**",
            "**/.venv/**",
            "**/.*/**",
        ]
        self._include_private = include_private

    async def discover(
        self,
        call_graph: CallGraph | None = None,
    ) -> DiscoveryResult:
        """Discover all components in the codebase.

        Args:
            call_graph: Optional call graph for computing processing order.
                        If not provided, order is based on file structure.

        Returns:
            DiscoveryResult with components and processing order

        Raises:
            FileNotFoundError: If codebase_path doesn't exist
        """
        if not self._codebase_path.exists():
            raise FileNotFoundError(f"Codebase path not found: {self._codebase_path}")

        result = DiscoveryResult()

        # Get files to analyze
        files = self._get_files()
        result.files_analyzed = len(files)

        logger.info(f"Discovering components in {len(files)} files")

        # Analyze each file
        for file_path in files:
            try:
                components, source_map = self._analyze_file(file_path)

                # Filter private components if needed
                if not self._include_private:
                    components = [c for c in components if c.is_public]

                for component in components:
                    result.components.append(component)
                    result.component_map[component.component_id] = component

                result.source_code_map.update(source_map)

            except SyntaxError as e:
                error_msg = f"Syntax error in {file_path}: {e}"
                result.errors.append(error_msg)
                logger.warning(error_msg)
            except Exception as e:
                error_msg = f"Error analyzing {file_path}: {e}"
                result.errors.append(error_msg)
                logger.warning(error_msg)

        # Compute processing order
        if call_graph is not None:
            result.processing_order = self._compute_topological_order(result.components, call_graph)
        else:
            # Default order: by file path and line number
            result.processing_order = [
                c.component_id
                for c in sorted(
                    result.components,
                    key=lambda c: (c.location.file_path, c.location.line_start),
                )
            ]

        logger.info(f"Discovered {len(result.components)} components, {len(result.errors)} errors")

        return result

    def _get_files(self) -> list[Path]:
        """Get list of files to analyze based on include/exclude patterns.

        Returns:
            List of file paths to analyze
        """
        import fnmatch

        all_files: set[Path] = set()

        # Collect files matching include patterns
        for pattern in self._include_patterns:
            all_files.update(self._codebase_path.glob(pattern))

        # Filter out excluded patterns
        filtered = []
        for file_path in all_files:
            try:
                rel_path = str(file_path.relative_to(self._codebase_path))
            except ValueError:
                rel_path = str(file_path)

            excluded = any(fnmatch.fnmatch(rel_path, exc) for exc in self._exclude_patterns)
            if not excluded and file_path.is_file():
                filtered.append(file_path)

        return sorted(filtered)

    def _analyze_file(
        self,
        file_path: Path,
    ) -> tuple[list[Component], dict[str, str]]:
        """Analyze a single Python file.

        Args:
            file_path: Path to the file

        Returns:
            Tuple of (components, source_code_map)
        """
        # Read source
        source = file_path.read_text(encoding="utf-8")
        source_lines = source.splitlines()

        # Parse AST
        tree = ast.parse(source, filename=str(file_path))

        # Calculate module name
        module_name = self._path_to_module(file_path)

        # Calculate relative file path
        try:
            rel_path = str(file_path.relative_to(self._codebase_path))
        except ValueError:
            rel_path = str(file_path)

        # Visit AST
        visitor = ComponentVisitor(
            module_name=module_name,
            file_path=rel_path,
            source_lines=source_lines,
            codebase_root=self._codebase_path,
        )
        visitor.visit(tree)

        return visitor.components, visitor.source_code_map

    def _path_to_module(self, file_path: Path) -> str:
        """Convert a file path to a module name.

        Args:
            file_path: Path to the Python file

        Returns:
            Module name in dotted notation
        """
        try:
            relative = file_path.relative_to(self._codebase_path)
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

        return ".".join(parts) if parts else ""

    def _compute_topological_order(
        self,
        components: list[Component],
        call_graph: CallGraph,
    ) -> list[str]:
        """Compute topological ordering of components.

        Components are ordered so that dependencies (callees) are
        processed before dependents (callers). This enables
        the documentation pipeline to use completed documentation
        of dependencies when documenting dependents.

        Args:
            components: List of discovered components
            call_graph: Call graph with dependency information

        Returns:
            List of component IDs in topological order
        """
        component_ids = {c.component_id for c in components}

        # Build dependency graph
        # A depends on B if A calls B, so B should be processed first
        in_degree: dict[str, int] = dict.fromkeys(component_ids, 0)
        adjacency: dict[str, list[str]] = {cid: [] for cid in component_ids}

        # Build graph edges from call graph
        for edge in call_graph.edges:
            caller = edge.caller
            callee = edge.callee

            # Only consider edges where both ends are in our component set
            if caller in component_ids and callee in component_ids:
                # Callee should come before caller in processing order
                adjacency[callee].append(caller)
                in_degree[caller] += 1

        # Kahn's algorithm for topological sort
        # Start with components that have no dependencies
        queue = [cid for cid, deg in in_degree.items() if deg == 0]
        result = []

        while queue:
            # Sort for deterministic order
            queue.sort()
            current = queue.pop(0)
            result.append(current)

            for neighbor in adjacency[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Handle cycles: append remaining components
        remaining = [cid for cid in component_ids if cid not in result]
        if remaining:
            logger.warning(f"Dependency cycle detected involving {len(remaining)} components")
            result.extend(sorted(remaining))

        return result


def create_component_discovery(
    codebase_path: str | Path,
    include_private: bool = False,
) -> ComponentDiscovery:
    """Factory function to create a ComponentDiscovery instance.

    Args:
        codebase_path: Path to the codebase
        include_private: Whether to include private components

    Returns:
        Configured ComponentDiscovery instance
    """
    return ComponentDiscovery(
        codebase_path=codebase_path,
        include_private=include_private,
    )


async def discover_components(
    codebase_path: str | Path,
    call_graph: CallGraph | None = None,
    include_private: bool = False,
) -> DiscoveryResult:
    """Convenience function to discover components in a codebase.

    Args:
        codebase_path: Path to the codebase
        call_graph: Optional call graph for topological ordering
        include_private: Whether to include private components

    Returns:
        DiscoveryResult with discovered components

    Usage:
        result = await discover_components("/path/to/project", call_graph)
        for component in result.components:
            print(f"{component.component_id}: {component.type}")
    """
    discovery = create_component_discovery(codebase_path, include_private)
    return await discovery.discover(call_graph)
