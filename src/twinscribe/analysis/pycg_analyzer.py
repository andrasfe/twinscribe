"""
PyCG Analyzer Implementation.

Wraps the PyCG static analysis tool for extracting call graphs from
Python source code. PyCG performs whole-program analysis and produces
accurate call graphs including dynamic dispatch resolution.
"""

import asyncio
import json
import logging
import tempfile
import time
from datetime import datetime
from pathlib import Path

from twinscribe.analysis.analyzer import (
    PYCG_CONFIG,
    Analyzer,
    AnalyzerConfig,
    AnalyzerError,
    AnalyzerResult,
    AnalyzerType,
    RawCallEdge,
)

logger = logging.getLogger(__name__)


class PyCGAnalyzer(Analyzer):
    """PyCG-based analyzer for Python call graphs.

    PyCG (Python Call Graphs) is a static analysis tool that generates
    call graphs for Python programs. It handles:
    - Direct function/method calls
    - Dynamic dispatch (method resolution)
    - Higher-order functions
    - Module-level code

    PyCG can be used as a library or via command line. This implementation
    supports both modes with library usage preferred.

    Usage:
        analyzer = PyCGAnalyzer(PYCG_CONFIG)
        if await analyzer.check_available():
            result = await analyzer.analyze(Path("/path/to/codebase"))
    """

    def __init__(self, config: AnalyzerConfig | None = None) -> None:
        """Initialize the PyCG analyzer.

        Args:
            config: Analyzer configuration (uses PYCG_CONFIG defaults if None)
        """
        if config is None:
            config = PYCG_CONFIG.model_copy()
        super().__init__(config)
        self._pycg_available: bool | None = None
        self._version: str | None = None

    async def check_available(self) -> bool:
        """Check if PyCG is available.

        Attempts to import pycg library first, then falls back to
        checking command-line availability.

        Returns:
            True if PyCG is available for use
        """
        if self._pycg_available is not None:
            return self._pycg_available

        # Try library import first
        try:
            import pycg  # noqa: F401

            self._pycg_available = True
            logger.debug("PyCG available as library")
            return True
        except ImportError:
            pass

        # Fall back to command line
        try:
            result = await asyncio.create_subprocess_exec(
                "pycg",
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await result.communicate()
            self._pycg_available = result.returncode == 0
        except FileNotFoundError:
            self._pycg_available = False

        if self._pycg_available:
            logger.debug("PyCG available via command line")
        else:
            logger.warning("PyCG not available")

        return self._pycg_available

    async def get_version(self) -> str | None:
        """Get PyCG version.

        Returns:
            Version string or None if not available
        """
        if self._version is not None:
            return self._version

        if not await self.check_available():
            return None

        try:
            # Try to get version from library
            import pycg

            self._version = getattr(pycg, "__version__", "unknown")
            return self._version
        except ImportError:
            pass

        # Fall back to command line
        try:
            result = await asyncio.create_subprocess_exec(
                "pycg",
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await result.communicate()
            if result.returncode == 0:
                self._version = stdout.decode().strip()
            else:
                self._version = "unknown"
        except Exception:
            self._version = "unknown"

        return self._version

    async def analyze(self, codebase_path: Path) -> AnalyzerResult:
        """Run PyCG analysis on the codebase.

        Analyzes Python source files and extracts call graph edges.

        Args:
            codebase_path: Path to the codebase root

        Returns:
            AnalyzerResult with raw call edges

        Raises:
            AnalyzerError: If analysis fails
            FileNotFoundError: If codebase_path doesn't exist
            TimeoutError: If analysis exceeds timeout
        """
        if not codebase_path.exists():
            raise FileNotFoundError(f"Codebase path not found: {codebase_path}")

        if not await self.check_available():
            raise AnalyzerError(
                "PyCG is not available. Install with: pip install pycg",
                analyzer_type=AnalyzerType.PYCG,
            )

        start_time = time.time()
        logger.info(f"Starting PyCG analysis on {codebase_path}")

        # Get files to analyze
        files = self._filter_files(codebase_path)
        if not files:
            logger.warning(f"No Python files found in {codebase_path}")
            return AnalyzerResult(
                analyzer_type=AnalyzerType.PYCG,
                codebase_path=str(codebase_path),
                warnings=["No Python files found matching patterns"],
            )

        logger.debug(f"Found {len(files)} Python files to analyze")

        # Run analysis
        try:
            raw_output = await self._run_pycg_analysis(codebase_path, files)
            raw_edges = self.parse_output(raw_output)
        except TimeoutError:
            raise TimeoutError(f"PyCG analysis timed out after {self._config.timeout_seconds}s")
        except Exception as e:
            raise AnalyzerError(
                f"PyCG analysis failed: {e}",
                analyzer_type=AnalyzerType.PYCG,
                stderr=str(e),
            )

        # Collect all nodes
        nodes = set()
        for edge in raw_edges:
            nodes.add(edge.caller)
            nodes.add(edge.callee)

        execution_time = time.time() - start_time
        logger.info(
            f"PyCG analysis complete: {len(raw_edges)} edges, "
            f"{len(nodes)} nodes in {execution_time:.2f}s"
        )

        return AnalyzerResult(
            analyzer_type=AnalyzerType.PYCG,
            codebase_path=str(codebase_path),
            raw_edges=raw_edges,
            nodes=nodes,
            execution_time_seconds=execution_time,
            timestamp=datetime.utcnow(),
            metadata={
                "files_analyzed": len(files),
                "pycg_version": await self.get_version(),
            },
        )

    async def _run_pycg_analysis(
        self,
        codebase_path: Path,
        files: list[Path],
    ) -> str:
        """Run PyCG analysis and get JSON output.

        Tries library-based analysis first, then falls back to subprocess.

        Args:
            codebase_path: Root path of the codebase
            files: List of files to analyze

        Returns:
            JSON string with call graph data
        """
        # Try library-based analysis first (more efficient)
        try:
            return await self._run_pycg_library(codebase_path, files)
        except ImportError:
            logger.debug("PyCG library not available, using subprocess")
        except Exception as e:
            logger.warning(f"Library analysis failed, falling back to subprocess: {e}")

        # Fall back to subprocess
        return await self._run_pycg_subprocess(codebase_path, files)

    async def _run_pycg_library(
        self,
        codebase_path: Path,
        files: list[Path],
    ) -> str:
        """Run PyCG using the library API.

        Args:
            codebase_path: Root path of the codebase
            files: List of files to analyze

        Returns:
            JSON string with call graph data
        """
        from pycg import pycg as pycg_module

        def run_analysis() -> dict[str, list[str]]:
            """Run PyCG analysis in a blocking manner."""
            # Convert file paths to strings
            file_strs = [str(f) for f in files]

            # Create PyCG instance
            cg = pycg_module.CallGraphGenerator(
                file_strs,
                str(codebase_path),
                max_iter=self._config.max_iterations,
                operation="call-graph",
            )
            cg.analyze()

            return cg.output()

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, run_analysis),
            timeout=self._config.timeout_seconds,
        )

        return json.dumps(result)

    async def _run_pycg_subprocess(
        self,
        codebase_path: Path,
        files: list[Path],
    ) -> str:
        """Run PyCG using subprocess.

        Args:
            codebase_path: Root path of the codebase
            files: List of files to analyze

        Returns:
            JSON string with call graph data
        """
        # Create a temp file for output
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
        ) as f:
            output_path = f.name

        try:
            # Build command
            cmd = self._build_command(codebase_path)
            cmd.extend(["--output", output_path])

            # Add entry points (all Python files)
            for file_path in files:
                cmd.append(str(file_path))

            logger.debug(f"Running PyCG command: {' '.join(cmd)}")

            # Run subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(codebase_path),
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self._config.timeout_seconds,
                )
            except TimeoutError:
                process.kill()
                raise

            if process.returncode != 0:
                raise AnalyzerError(
                    f"PyCG exited with code {process.returncode}",
                    analyzer_type=AnalyzerType.PYCG,
                    exit_code=process.returncode,
                    stderr=stderr.decode() if stderr else None,
                )

            # Read output
            output_file = Path(output_path)
            if output_file.exists():
                return output_file.read_text()
            else:
                # PyCG might have written to stdout instead
                return stdout.decode()

        finally:
            # Clean up temp file
            try:
                Path(output_path).unlink(missing_ok=True)
            except Exception:
                pass

    def _build_command(self, codebase_path: Path) -> list[str]:
        """Build the PyCG command line.

        Args:
            codebase_path: Path to codebase

        Returns:
            Command as list of strings
        """
        cmd = ["pycg"]

        # Add package path for module resolution
        cmd.extend(["--package", str(codebase_path)])

        # Add max iterations
        cmd.extend(["--max-iter", str(self._config.max_iterations)])

        # Add any extra arguments
        cmd.extend(self._config.extra_args)

        return cmd

    def parse_output(self, raw_output: str) -> list[RawCallEdge]:
        """Parse PyCG JSON output to RawCallEdge list.

        PyCG produces output in the format:
        {
            "module.function": ["module2.function2", "module3.Class.method"],
            ...
        }

        Where keys are callers and values are lists of callees.

        Args:
            raw_output: JSON string from PyCG

        Returns:
            List of RawCallEdge objects
        """
        if not raw_output or not raw_output.strip():
            logger.warning("Empty PyCG output")
            return []

        try:
            data = json.loads(raw_output)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse PyCG JSON output: {e}")
            return []

        edges = []
        for caller, callees in data.items():
            if not isinstance(callees, list):
                logger.warning(f"Unexpected callee format for {caller}: {callees}")
                continue

            for callee in callees:
                if not isinstance(callee, str):
                    continue

                edges.append(
                    RawCallEdge(
                        caller=caller,
                        callee=callee,
                    )
                )

        return edges


def create_pycg_analyzer(config: AnalyzerConfig | None = None) -> PyCGAnalyzer:
    """Factory function for creating PyCG analyzer.

    Args:
        config: Optional configuration

    Returns:
        Configured PyCGAnalyzer instance
    """
    return PyCGAnalyzer(config)
