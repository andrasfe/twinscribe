"""
Beads CLI Client.

Wrapper for interacting with Beads via the bd CLI.
See https://github.com/steveyegge/beads for Beads documentation.
"""

import asyncio
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


class BeadsClientConfig(BaseModel):
    """Configuration for Beads client.

    Attributes:
        directory: Beads directory (relative to repo root)
        labels: Default labels for created issues
        timeout_seconds: CLI command timeout
        max_retries: Maximum retry attempts
    """

    directory: str = Field(
        default=".beads",
        description="Beads directory",
    )
    labels: list[str] = Field(
        default_factory=lambda: ["ai-documentation", "twinscribe"],
        description="Default labels for issues",
    )
    timeout_seconds: int = Field(
        default=30,
        ge=1,
        description="CLI command timeout",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Max retries",
    )


@dataclass
class BeadsIssue:
    """Representation of a Beads issue.

    Attributes:
        id: Issue ID (e.g., bd-a1b2)
        title: Issue title/summary
        description: Issue description
        status: Current status
        priority: Priority (0=highest)
        labels: List of labels
        created: Creation timestamp
        updated: Last update timestamp
        parent_id: Parent issue ID (for subtasks)
        dependencies: List of blocking issue IDs
        metadata: Additional metadata
    """

    id: str
    title: str
    description: str = ""
    status: str = "open"
    priority: int = 1
    labels: list[str] = field(default_factory=list)
    created: Optional[datetime] = None
    updated: Optional[datetime] = None
    parent_id: Optional[str] = None
    dependencies: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        """Check if issue is open."""
        return self.status.lower() in ("open", "in_progress", "pending")

    @property
    def is_closed(self) -> bool:
        """Check if issue is closed."""
        return self.status.lower() in ("closed", "done", "resolved")


@dataclass
class CreateIssueRequest:
    """Request to create a new issue.

    Attributes:
        title: Issue title
        description: Issue description
        priority: Priority (0=highest)
        labels: Labels to add
        parent_id: Parent issue ID (for subtasks)
        dependencies: Blocking issue IDs
    """

    title: str
    description: str = ""
    priority: int = 1
    labels: list[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    dependencies: list[str] = field(default_factory=list)


class BeadsClient:
    """Client for interacting with Beads via CLI.

    Wraps the bd CLI commands for async Python usage.
    """

    def __init__(self, config: Optional[BeadsClientConfig] = None) -> None:
        """Initialize the client.

        Args:
            config: Client configuration (uses defaults if None)
        """
        self._config = config or BeadsClientConfig()
        self._initialized = False
        self._bd_path: Optional[str] = None

    @property
    def config(self) -> BeadsClientConfig:
        """Get client configuration."""
        return self._config

    @property
    def is_initialized(self) -> bool:
        """Check if client is initialized."""
        return self._initialized

    async def initialize(self) -> None:
        """Initialize the client and verify bd is available.

        Raises:
            BeadsError: If bd CLI not found or not initialized
        """
        # Check if bd is available
        self._bd_path = shutil.which("bd")
        if not self._bd_path:
            raise BeadsError(
                "Beads CLI (bd) not found. Install via: ./scripts/install-beads.sh"
            )

        # Check if beads is initialized in the repo
        beads_dir = Path(self._config.directory)
        if not beads_dir.exists():
            # Try to initialize
            await self._run_command(["init"])

        self._initialized = True

    async def close(self) -> None:
        """Close the client (sync with git if needed)."""
        if self._initialized:
            try:
                await self.sync()
            except BeadsError:
                pass  # Ignore sync errors on close

    async def _run_command(
        self,
        args: list[str],
        capture_output: bool = True,
    ) -> str:
        """Run a bd CLI command.

        Args:
            args: Command arguments (without 'bd' prefix)
            capture_output: Whether to capture and return output

        Returns:
            Command output if capture_output=True

        Raises:
            BeadsError: If command fails
        """
        cmd = [self._bd_path or "bd"] + args

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE if capture_output else None,
                stderr=asyncio.subprocess.PIPE if capture_output else None,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self._config.timeout_seconds,
            )

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise BeadsError(
                    f"bd command failed: {' '.join(args)}\n{error_msg}",
                    exit_code=process.returncode,
                )

            return stdout.decode() if stdout else ""

        except asyncio.TimeoutError as e:
            raise BeadsError(f"bd command timed out: {' '.join(args)}") from e

    async def create_issue(self, request: CreateIssueRequest) -> BeadsIssue:
        """Create a new issue.

        Args:
            request: Issue creation request

        Returns:
            Created issue

        Raises:
            BeadsError: If creation fails
        """
        args = ["create", request.title, "-p", str(request.priority)]

        # Add labels
        for label in request.labels or self._config.labels:
            args.extend(["-l", label])

        output = await self._run_command(args)

        # Parse the created issue ID from output
        # Expected format: "Created issue bd-xxxx"
        issue_id = self._parse_issue_id(output)

        # If description provided, update the issue
        if request.description:
            # bd doesn't have direct description support in create,
            # so we add it as a comment or metadata
            pass

        # Add dependencies if specified
        for dep_id in request.dependencies:
            await self._run_command(["dep", "add", issue_id, dep_id])

        return await self.get_issue(issue_id)

    async def get_issue(self, issue_id: str) -> BeadsIssue:
        """Get issue details.

        Args:
            issue_id: Issue ID (e.g., bd-a1b2)

        Returns:
            Issue details

        Raises:
            NotFoundError: If issue doesn't exist
        """
        try:
            output = await self._run_command(["show", issue_id, "--json"])
            data = json.loads(output)

            return BeadsIssue(
                id=data.get("id", issue_id),
                title=data.get("title", ""),
                description=data.get("description", ""),
                status=data.get("status", "open"),
                priority=data.get("priority", 1),
                labels=data.get("labels", []),
                created=self._parse_datetime(data.get("created")),
                updated=self._parse_datetime(data.get("updated")),
                parent_id=data.get("parent_id"),
                dependencies=data.get("dependencies", []),
                metadata=data.get("metadata", {}),
            )
        except BeadsError as e:
            if "not found" in str(e).lower():
                raise NotFoundError(f"Issue not found: {issue_id}") from e
            raise

    async def update_issue(
        self,
        issue_id: str,
        status: Optional[str] = None,
        priority: Optional[int] = None,
        **kwargs: Any,
    ) -> BeadsIssue:
        """Update an issue.

        Args:
            issue_id: Issue ID
            status: New status
            priority: New priority
            **kwargs: Additional fields to update

        Returns:
            Updated issue

        Raises:
            NotFoundError: If issue doesn't exist
        """
        args = ["update", issue_id]

        if status:
            args.extend(["--status", status])

        if priority is not None:
            args.extend(["-p", str(priority)])

        await self._run_command(args)
        return await self.get_issue(issue_id)

    async def close_issue(self, issue_id: str) -> BeadsIssue:
        """Close an issue.

        Args:
            issue_id: Issue ID

        Returns:
            Closed issue
        """
        await self._run_command(["close", issue_id])
        return await self.get_issue(issue_id)

    async def list_ready(self) -> list[BeadsIssue]:
        """List issues with no blocking dependencies.

        Returns:
            List of ready issues
        """
        output = await self._run_command(["ready", "--json"])
        try:
            data = json.loads(output)
            return [
                BeadsIssue(
                    id=item.get("id", ""),
                    title=item.get("title", ""),
                    status=item.get("status", "open"),
                    priority=item.get("priority", 1),
                    labels=item.get("labels", []),
                )
                for item in data
            ]
        except json.JSONDecodeError:
            return []

    async def sync(self) -> None:
        """Sync with git."""
        await self._run_command(["sync"])

    async def add_dependency(self, issue_id: str, depends_on: str) -> None:
        """Add a dependency relationship.

        Args:
            issue_id: Issue that depends on another
            depends_on: Issue being depended on
        """
        await self._run_command(["dep", "add", issue_id, depends_on])

    def _parse_issue_id(self, output: str) -> str:
        """Parse issue ID from command output.

        Args:
            output: Command output

        Returns:
            Issue ID
        """
        # Look for pattern like "bd-xxxx" in output
        import re

        match = re.search(r"(bd-[a-z0-9]+)", output, re.IGNORECASE)
        if match:
            return match.group(1)

        # Fallback: return first word that looks like an ID
        for word in output.split():
            if word.startswith("bd-"):
                return word

        raise BeadsError(f"Could not parse issue ID from output: {output}")

    def _parse_datetime(self, value: Optional[str]) -> Optional[datetime]:
        """Parse datetime from string.

        Args:
            value: ISO format datetime string

        Returns:
            Parsed datetime or None
        """
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None


class BeadsError(Exception):
    """Base exception for Beads client errors."""

    def __init__(
        self,
        message: str,
        exit_code: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class NotFoundError(BeadsError):
    """Raised when an issue is not found."""

    pass
