"""
Beads API Client.

Low-level client for interacting with Beads/Jira REST API.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, SecretStr


class BeadsClientConfig(BaseModel):
    """Configuration for Beads API client.

    Attributes:
        server: Beads/Jira server URL
        username: Username for authentication
        api_token: API token (from SecretStr)
        timeout_seconds: Request timeout
        max_retries: Maximum retry attempts
        verify_ssl: Whether to verify SSL certificates
    """

    server: str = Field(
        ...,
        description="Beads server URL",
        examples=["https://your-org.atlassian.net"],
    )
    username: str = Field(
        ...,
        description="Username for authentication",
    )
    api_token: SecretStr = Field(
        ...,
        description="API token",
    )
    timeout_seconds: int = Field(
        default=30,
        ge=1,
        description="Request timeout",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Max retries",
    )
    verify_ssl: bool = Field(
        default=True,
        description="Verify SSL certificates",
    )


@dataclass
class BeadsIssue:
    """Representation of a Beads/Jira issue.

    Attributes:
        key: Issue key (e.g., LEGACY-123)
        id: Numeric issue ID
        summary: Issue summary/title
        description: Issue description
        status: Current status name
        priority: Priority name
        issue_type: Issue type name
        labels: List of labels
        created: Creation timestamp
        updated: Last update timestamp
        resolution: Resolution name if resolved
        assignee: Assignee username
        reporter: Reporter username
        comments: List of comments
        custom_fields: Custom field values
    """

    key: str
    id: str
    summary: str
    description: str
    status: str
    priority: str
    issue_type: str
    labels: list[str] = field(default_factory=list)
    created: Optional[datetime] = None
    updated: Optional[datetime] = None
    resolution: Optional[str] = None
    assignee: Optional[str] = None
    reporter: Optional[str] = None
    comments: list["BeadsComment"] = field(default_factory=list)
    custom_fields: dict[str, Any] = field(default_factory=dict)

    @property
    def is_resolved(self) -> bool:
        """Check if issue is resolved."""
        return self.resolution is not None

    @property
    def is_open(self) -> bool:
        """Check if issue is open."""
        return self.status.lower() in ("open", "to do", "new", "in progress")


@dataclass
class BeadsComment:
    """Representation of a Beads/Jira comment.

    Attributes:
        id: Comment ID
        author: Comment author
        body: Comment body text
        created: Creation timestamp
        updated: Last update timestamp
    """

    id: str
    author: str
    body: str
    created: Optional[datetime] = None
    updated: Optional[datetime] = None


@dataclass
class CreateIssueRequest:
    """Request to create a new issue.

    Attributes:
        project: Project key
        issue_type: Issue type name
        summary: Issue summary
        description: Issue description
        priority: Priority name
        labels: Labels to add
        assignee: Assignee username
        custom_fields: Custom field values
    """

    project: str
    issue_type: str
    summary: str
    description: str
    priority: str = "Medium"
    labels: list[str] = field(default_factory=list)
    assignee: Optional[str] = None
    custom_fields: dict[str, Any] = field(default_factory=dict)


class BeadsClient(ABC):
    """Abstract base class for Beads/Jira API client.

    Provides async interface for common Beads operations.
    Implementations can use jira-python, atlassian-python-api, or direct REST.
    """

    def __init__(self, config: BeadsClientConfig) -> None:
        """Initialize the client.

        Args:
            config: Client configuration
        """
        self._config = config
        self._initialized = False

    @property
    def config(self) -> BeadsClientConfig:
        """Get client configuration."""
        return self._config

    @property
    def is_initialized(self) -> bool:
        """Check if client is initialized."""
        return self._initialized

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the client and verify connection.

        Raises:
            ConnectionError: If cannot connect to server
            AuthenticationError: If credentials invalid
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the client and release resources."""
        pass

    @abstractmethod
    async def create_issue(self, request: CreateIssueRequest) -> BeadsIssue:
        """Create a new issue.

        Args:
            request: Issue creation request

        Returns:
            Created issue

        Raises:
            BeadsError: If creation fails
        """
        pass

    @abstractmethod
    async def get_issue(self, key: str) -> BeadsIssue:
        """Get an issue by key.

        Args:
            key: Issue key (e.g., LEGACY-123)

        Returns:
            Issue details

        Raises:
            NotFoundError: If issue doesn't exist
        """
        pass

    @abstractmethod
    async def update_issue(
        self,
        key: str,
        fields: dict[str, Any],
    ) -> BeadsIssue:
        """Update an issue.

        Args:
            key: Issue key
            fields: Fields to update

        Returns:
            Updated issue

        Raises:
            NotFoundError: If issue doesn't exist
        """
        pass

    @abstractmethod
    async def add_comment(self, key: str, body: str) -> BeadsComment:
        """Add a comment to an issue.

        Args:
            key: Issue key
            body: Comment body

        Returns:
            Created comment
        """
        pass

    @abstractmethod
    async def get_comments(self, key: str) -> list[BeadsComment]:
        """Get all comments for an issue.

        Args:
            key: Issue key

        Returns:
            List of comments
        """
        pass

    @abstractmethod
    async def transition_issue(
        self,
        key: str,
        transition_name: str,
        resolution: Optional[str] = None,
    ) -> BeadsIssue:
        """Transition an issue to a new status.

        Args:
            key: Issue key
            transition_name: Name of transition to execute
            resolution: Resolution to set (for closing transitions)

        Returns:
            Updated issue
        """
        pass

    @abstractmethod
    async def search_issues(
        self,
        jql: str,
        max_results: int = 50,
        fields: Optional[list[str]] = None,
    ) -> list[BeadsIssue]:
        """Search issues using JQL.

        Args:
            jql: JQL query string
            max_results: Maximum results to return
            fields: Fields to include (None = all)

        Returns:
            List of matching issues
        """
        pass

    async def get_open_issues_by_label(
        self,
        project: str,
        label: str,
    ) -> list[BeadsIssue]:
        """Get open issues with a specific label.

        Args:
            project: Project key
            label: Label to filter by

        Returns:
            List of open issues with the label
        """
        jql = (
            f'project = "{project}" AND '
            f'labels = "{label}" AND '
            f'status NOT IN (Done, Resolved, Closed)'
        )
        return await self.search_issues(jql)


class BeadsError(Exception):
    """Base exception for Beads client errors."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class AuthenticationError(BeadsError):
    """Raised when authentication fails."""

    pass


class NotFoundError(BeadsError):
    """Raised when a resource is not found."""

    pass


class PermissionError(BeadsError):
    """Raised when permission is denied."""

    pass
