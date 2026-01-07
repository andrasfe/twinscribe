"""
Unit tests for BeadsLifecycleManager (documentation lifecycle).

Tests the lifecycle management functionality for documentation tickets:
- Creating documentation tickets
- Updating ticket status
- Recording validation results
- Linking dependencies
- Closing tickets
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from twinscribe.beads.client import BeadsError, BeadsIssue
from twinscribe.beads.lifecycle import (
    BeadsLifecycleManager,
    CloseReason,
    ConvergenceMetrics,
    DocumentationTicketStatus,
    LifecycleManagerConfig,
    ValidationSummary,
)
from twinscribe.beads.tracker import TicketStatus


@pytest.fixture
def mock_client():
    """Create a mock BeadsClient."""
    client = AsyncMock()
    client.is_initialized = True
    client.initialize = AsyncMock()
    client.close = AsyncMock()
    client.sync = AsyncMock()
    return client


@pytest.fixture
def mock_issue():
    """Create a mock BeadsIssue."""
    return BeadsIssue(
        id="twinscribe-abc",
        title="[DOC] Document function: test_func",
        description="Test description",
        status="open",
        priority=1,
        labels=["ai-documentation", "twinscribe"],
    )


@pytest.fixture
def manager_config():
    """Create test configuration."""
    return LifecycleManagerConfig(
        beads_directory=".beads",
        default_labels=["ai-documentation", "twinscribe"],
        auto_sync=False,  # Disable for testing
        update_on_progress=True,
        max_retries=3,
    )


@pytest.fixture
async def manager(mock_client, manager_config):
    """Create an initialized manager with mock client."""
    mgr = BeadsLifecycleManager(config=manager_config, client=mock_client)
    await mgr.initialize()
    return mgr


@pytest.fixture
def mock_component():
    """Create a mock source component."""
    component = MagicMock()
    component.name = "test_function"
    component.type = "function"
    component.file_path = "src/utils/helpers.py"
    component.component_id = "utils.helpers.test_function"
    component.signature = "def test_function(x: int, y: str) -> bool"
    component.existing_docstring = "Existing docstring."
    return component


@pytest.fixture
def mock_validation_result():
    """Create a mock ValidationResult."""
    result = MagicMock()
    result.component_id = "utils.helpers.test_function"
    result.validation_result = MagicMock()
    result.validation_result.value = "pass"
    result.completeness = MagicMock()
    result.completeness.score = 0.95
    result.completeness.missing_elements = []
    result.call_graph_accuracy = MagicMock()
    result.call_graph_accuracy.score = 0.98
    result.call_graph_accuracy.false_callees = []
    result.total_corrections = 2
    return result


class TestBeadsLifecycleManagerInit:
    """Tests for BeadsLifecycleManager initialization."""

    @pytest.mark.asyncio
    async def test_init_with_config(self, manager_config):
        """Test manager initializes with provided config."""
        manager = BeadsLifecycleManager(config=manager_config)
        assert manager.config == manager_config
        assert not manager.is_initialized

    @pytest.mark.asyncio
    async def test_init_default_config(self):
        """Test manager initializes with default config."""
        manager = BeadsLifecycleManager()
        assert manager.config is not None
        assert manager.config.beads_directory == ".beads"

    @pytest.mark.asyncio
    async def test_initialize(self, mock_client, manager_config):
        """Test manager initialization."""
        manager = BeadsLifecycleManager(config=manager_config, client=mock_client)
        await manager.initialize()
        assert manager.is_initialized

    @pytest.mark.asyncio
    async def test_double_initialize_is_safe(self, mock_client, manager_config):
        """Test calling initialize twice is safe."""
        manager = BeadsLifecycleManager(config=manager_config, client=mock_client)
        await manager.initialize()
        await manager.initialize()  # Should not raise
        assert manager.is_initialized

    @pytest.mark.asyncio
    async def test_close(self, manager):
        """Test manager close."""
        await manager.close()
        assert not manager.is_initialized


class TestCreateDocumentationTicket:
    """Tests for create_documentation_ticket method."""

    @pytest.mark.asyncio
    async def test_create_ticket_success(self, manager, mock_client, mock_component, mock_issue):
        """Test successful ticket creation."""
        mock_client.create_issue.return_value = mock_issue

        ticket_id = await manager.create_documentation_ticket(mock_component)

        assert ticket_id == "twinscribe-abc"
        mock_client.create_issue.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_ticket_sets_correct_title(
        self, manager, mock_client, mock_component, mock_issue
    ):
        """Test ticket title is formatted correctly."""
        mock_client.create_issue.return_value = mock_issue

        await manager.create_documentation_ticket(mock_component)

        call_args = mock_client.create_issue.call_args
        request = call_args[0][0]
        assert "test_function" in request.title
        assert "[DOC]" in request.title

    @pytest.mark.asyncio
    async def test_create_ticket_includes_labels(
        self, manager, mock_client, mock_component, mock_issue
    ):
        """Test ticket includes configured labels."""
        mock_client.create_issue.return_value = mock_issue

        await manager.create_documentation_ticket(mock_component, labels=["custom-label"])

        call_args = mock_client.create_issue.call_args
        request = call_args[0][0]
        assert "ai-documentation" in request.labels
        assert "custom-label" in request.labels

    @pytest.mark.asyncio
    async def test_create_ticket_tracks_in_tracker(
        self, manager, mock_client, mock_component, mock_issue
    ):
        """Test created ticket is tracked."""
        mock_client.create_issue.return_value = mock_issue

        ticket_id = await manager.create_documentation_ticket(mock_component)

        tracked = manager.tracker.get(ticket_id)
        assert tracked is not None
        assert tracked.ticket_key == ticket_id

    @pytest.mark.asyncio
    async def test_create_ticket_stores_component_mapping(
        self, manager, mock_client, mock_component, mock_issue
    ):
        """Test component to ticket mapping is stored."""
        mock_client.create_issue.return_value = mock_issue

        ticket_id = await manager.create_documentation_ticket(mock_component)

        component_ticket = await manager.get_component_ticket(mock_component.component_id)
        assert component_ticket == ticket_id

    @pytest.mark.asyncio
    async def test_create_ticket_records_history(
        self, manager, mock_client, mock_component, mock_issue
    ):
        """Test ticket creation is recorded in history."""
        mock_client.create_issue.return_value = mock_issue

        ticket_id = await manager.create_documentation_ticket(mock_component)

        history = await manager.get_ticket_history(ticket_id)
        assert len(history) == 1
        assert history[0]["action"] == "created"

    @pytest.mark.asyncio
    async def test_create_ticket_not_initialized_raises(
        self, mock_client, manager_config, mock_component
    ):
        """Test creating ticket before initialization raises error."""
        manager = BeadsLifecycleManager(config=manager_config, client=mock_client)

        with pytest.raises(RuntimeError, match="not initialized"):
            await manager.create_documentation_ticket(mock_component)


class TestUpdateTicketStatus:
    """Tests for update_ticket_status method."""

    @pytest.mark.asyncio
    async def test_update_status_success(self, manager, mock_client, mock_component, mock_issue):
        """Test successful status update."""
        mock_client.create_issue.return_value = mock_issue
        mock_client.update_issue.return_value = mock_issue

        ticket_id = await manager.create_documentation_ticket(mock_component)
        await manager.update_ticket_status(ticket_id, DocumentationTicketStatus.IN_PROGRESS)

        mock_client.update_issue.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_status_with_string(
        self, manager, mock_client, mock_component, mock_issue
    ):
        """Test status update with string value."""
        mock_client.create_issue.return_value = mock_issue
        mock_client.update_issue.return_value = mock_issue

        ticket_id = await manager.create_documentation_ticket(mock_component)
        await manager.update_ticket_status(ticket_id, "in_progress")

        mock_client.update_issue.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_status_records_history(
        self, manager, mock_client, mock_component, mock_issue
    ):
        """Test status update is recorded in history."""
        mock_client.create_issue.return_value = mock_issue
        mock_client.update_issue.return_value = mock_issue

        ticket_id = await manager.create_documentation_ticket(mock_component)
        await manager.update_ticket_status(
            ticket_id, DocumentationTicketStatus.DOCUMENTING, message="Starting documentation"
        )

        history = await manager.get_ticket_history(ticket_id)
        assert len(history) == 2
        assert history[1]["action"] == "status_update"
        assert history[1]["status"] == "documenting"
        assert history[1]["message"] == "Starting documentation"

    @pytest.mark.asyncio
    async def test_update_status_updates_tracker(
        self, manager, mock_client, mock_component, mock_issue
    ):
        """Test status update modifies tracker."""
        mock_client.create_issue.return_value = mock_issue
        mock_client.update_issue.return_value = mock_issue

        ticket_id = await manager.create_documentation_ticket(mock_component)
        await manager.update_ticket_status(ticket_id, DocumentationTicketStatus.IN_PROGRESS)

        tracked = manager.tracker.get(ticket_id)
        assert tracked.status == TicketStatus.IN_PROGRESS


class TestRecordValidationResult:
    """Tests for record_validation_result method."""

    @pytest.mark.asyncio
    async def test_record_validation_success(
        self, manager, mock_client, mock_component, mock_issue, mock_validation_result
    ):
        """Test recording validation result."""
        mock_client.create_issue.return_value = mock_issue
        mock_client.update_issue.return_value = mock_issue

        ticket_id = await manager.create_documentation_ticket(mock_component)
        await manager.record_validation_result(ticket_id, mock_validation_result)

        # Should record in history
        history = await manager.get_ticket_history(ticket_id)
        validation_entries = [h for h in history if h["action"] == "validation_recorded"]
        assert len(validation_entries) == 1

    @pytest.mark.asyncio
    async def test_record_validation_includes_scores(
        self, manager, mock_client, mock_component, mock_issue, mock_validation_result
    ):
        """Test validation scores are recorded."""
        mock_client.create_issue.return_value = mock_issue
        mock_client.update_issue.return_value = mock_issue

        ticket_id = await manager.create_documentation_ticket(mock_component)
        await manager.record_validation_result(ticket_id, mock_validation_result)

        history = await manager.get_ticket_history(ticket_id)
        validation_entry = [h for h in history if h["action"] == "validation_recorded"][0]

        assert validation_entry["validation"]["completeness_score"] == 0.95
        assert validation_entry["validation"]["call_graph_accuracy"] == 0.98


class TestRecordConvergenceMetrics:
    """Tests for record_convergence_metrics method."""

    @pytest.mark.asyncio
    async def test_record_metrics_success(self, manager, mock_client, mock_component, mock_issue):
        """Test recording convergence metrics."""
        mock_client.create_issue.return_value = mock_issue

        ticket_id = await manager.create_documentation_ticket(mock_component)

        metrics = ConvergenceMetrics(
            iteration=3,
            call_graph_match_rate=0.95,
            documentation_similarity=0.92,
            discrepancies_remaining=2,
        )

        await manager.record_convergence_metrics(ticket_id, metrics)

        history = await manager.get_ticket_history(ticket_id)
        metrics_entries = [h for h in history if h["action"] == "convergence_metrics"]
        assert len(metrics_entries) == 1
        assert metrics_entries[0]["metrics"]["iteration"] == 3

    @pytest.mark.asyncio
    async def test_record_metrics_updates_tracker(
        self, manager, mock_client, mock_component, mock_issue
    ):
        """Test metrics update tracker metadata."""
        mock_client.create_issue.return_value = mock_issue

        ticket_id = await manager.create_documentation_ticket(mock_component)

        metrics = ConvergenceMetrics(
            iteration=5,
            call_graph_match_rate=0.99,
            documentation_similarity=0.98,
            discrepancies_remaining=0,
        )

        await manager.record_convergence_metrics(ticket_id, metrics)

        tracked = manager.tracker.get(ticket_id)
        assert "last_convergence" in tracked.metadata
        assert tracked.metadata["last_convergence"]["iteration"] == 5


class TestLinkDependencies:
    """Tests for link_dependencies method."""

    @pytest.mark.asyncio
    async def test_link_dependencies_success(
        self, manager, mock_client, mock_component, mock_issue
    ):
        """Test linking dependencies."""
        mock_client.create_issue.return_value = mock_issue
        mock_client.add_dependency.return_value = None

        ticket_id = await manager.create_documentation_ticket(mock_component)
        await manager.link_dependencies(ticket_id, ["dep-1", "dep-2"])

        assert mock_client.add_dependency.call_count == 2

    @pytest.mark.asyncio
    async def test_link_dependencies_records_history(
        self, manager, mock_client, mock_component, mock_issue
    ):
        """Test dependency linking is recorded."""
        mock_client.create_issue.return_value = mock_issue
        mock_client.add_dependency.return_value = None

        ticket_id = await manager.create_documentation_ticket(mock_component)
        await manager.link_dependencies(ticket_id, ["dep-1", "dep-2"])

        history = await manager.get_ticket_history(ticket_id)
        link_entries = [h for h in history if h["action"] == "dependencies_linked"]
        assert len(link_entries) == 1
        assert link_entries[0]["depends_on"] == ["dep-1", "dep-2"]

    @pytest.mark.asyncio
    async def test_link_dependencies_handles_errors(
        self, manager, mock_client, mock_component, mock_issue
    ):
        """Test dependency linking continues on errors."""
        mock_client.create_issue.return_value = mock_issue
        mock_client.add_dependency.side_effect = [
            BeadsError("Failed"),
            None,  # Second call succeeds
        ]

        ticket_id = await manager.create_documentation_ticket(mock_component)
        await manager.link_dependencies(ticket_id, ["dep-1", "dep-2"])

        # Should have recorded the failure
        history = await manager.get_ticket_history(ticket_id)
        failure_entries = [h for h in history if h["action"] == "dependency_link_failed"]
        assert len(failure_entries) == 1


class TestCloseTicket:
    """Tests for close_ticket method."""

    @pytest.mark.asyncio
    async def test_close_ticket_success(self, manager, mock_client, mock_component, mock_issue):
        """Test closing a ticket."""
        mock_client.create_issue.return_value = mock_issue
        mock_client.close_issue.return_value = mock_issue

        ticket_id = await manager.create_documentation_ticket(mock_component)
        await manager.close_ticket(ticket_id, CloseReason.COMPLETED)

        mock_client.close_issue.assert_called_once_with(ticket_id)

    @pytest.mark.asyncio
    async def test_close_ticket_with_string_reason(
        self, manager, mock_client, mock_component, mock_issue
    ):
        """Test closing with string reason."""
        mock_client.create_issue.return_value = mock_issue
        mock_client.close_issue.return_value = mock_issue

        ticket_id = await manager.create_documentation_ticket(mock_component)
        await manager.close_ticket(ticket_id, "converged")

        mock_client.close_issue.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_ticket_records_history(
        self, manager, mock_client, mock_component, mock_issue
    ):
        """Test ticket closure is recorded."""
        mock_client.create_issue.return_value = mock_issue
        mock_client.close_issue.return_value = mock_issue

        ticket_id = await manager.create_documentation_ticket(mock_component)
        await manager.close_ticket(
            ticket_id, CloseReason.COMPLETED, summary="Documentation completed successfully"
        )

        history = await manager.get_ticket_history(ticket_id)
        close_entries = [h for h in history if h["action"] == "closed"]
        assert len(close_entries) == 1
        assert close_entries[0]["reason"] == "completed"
        assert close_entries[0]["summary"] == "Documentation completed successfully"

    @pytest.mark.asyncio
    async def test_close_ticket_updates_tracker(
        self, manager, mock_client, mock_component, mock_issue
    ):
        """Test closing updates tracker status."""
        mock_client.create_issue.return_value = mock_issue
        mock_client.close_issue.return_value = mock_issue

        ticket_id = await manager.create_documentation_ticket(mock_component)
        await manager.close_ticket(ticket_id, CloseReason.COMPLETED)

        tracked = manager.tracker.get(ticket_id)
        assert tracked.status == TicketStatus.APPLIED
        assert tracked.metadata["close_reason"] == "completed"


class TestGetStatistics:
    """Tests for get_statistics method."""

    @pytest.mark.asyncio
    async def test_get_statistics(self, manager, mock_client, mock_component, mock_issue):
        """Test statistics gathering."""
        mock_client.create_issue.return_value = mock_issue

        await manager.create_documentation_ticket(mock_component)

        stats = manager.get_statistics()

        assert "total_documentation_tickets" in stats
        assert stats["total_documentation_tickets"] >= 0
        assert "tracker" in stats
        assert stats["initialized"] is True


class TestGetOpenDocumentationTickets:
    """Tests for get_open_documentation_tickets method."""

    @pytest.mark.asyncio
    async def test_get_open_tickets(self, manager, mock_client, mock_component, mock_issue):
        """Test getting open documentation tickets."""
        mock_client.create_issue.return_value = mock_issue

        await manager.create_documentation_ticket(mock_component)

        open_tickets = await manager.get_open_documentation_tickets()

        assert len(open_tickets) == 1
        assert open_tickets[0].ticket_key == "twinscribe-abc"


class TestValidationSummary:
    """Tests for ValidationSummary dataclass."""

    def test_to_dict(self):
        """Test ValidationSummary serialization."""
        summary = ValidationSummary(
            component_id="test.component",
            status="pass",
            completeness_score=0.95,
            call_graph_accuracy=0.98,
            corrections_count=1,
            errors=["minor error"],
        )

        result = summary.to_dict()

        assert result["component_id"] == "test.component"
        assert result["status"] == "pass"
        assert result["completeness_score"] == 0.95
        assert result["errors"] == ["minor error"]


class TestConvergenceMetrics:
    """Tests for ConvergenceMetrics dataclass."""

    def test_to_dict(self):
        """Test ConvergenceMetrics serialization."""
        metrics = ConvergenceMetrics(
            iteration=5,
            call_graph_match_rate=0.99,
            documentation_similarity=0.97,
            discrepancies_remaining=0,
        )

        result = metrics.to_dict()

        assert result["iteration"] == 5
        assert result["call_graph_match_rate"] == 0.99
        assert "timestamp" in result
