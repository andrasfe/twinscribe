# Beads Integration Design Document

## Overview

The Beads integration module provides human-in-the-loop resolution for edge cases that cannot be automatically resolved through ground truth validation or model consensus. It integrates with Jira/Beads issue tracking systems to create, monitor, and apply resolutions from human reviewers.

## Architecture

```
+-------------------------------------------------------------------+
|                    Beads Integration Module                        |
|                                                                   |
|  +-------------------------------------------------------------+  |
|  |                 BeadsLifecycleManager                       |  |
|  |                                                             |  |
|  |  - Creates discrepancy tickets                              |  |
|  |  - Creates rebuild tickets                                  |  |
|  |  - Monitors for resolutions                                 |  |
|  |  - Applies resolutions                                      |  |
|  |  - Handles timeouts                                         |  |
|  +-------------------------------------------------------------+  |
|           |              |                |                       |
|           v              v                v                       |
|  +--------------+  +--------------+  +--------------+             |
|  | BeadsClient  |  |TicketTracker |  |TemplateEngine|             |
|  |              |  |              |  |              |             |
|  | - API calls  |  | - Track state|  | - Render     |             |
|  | - Auth       |  | - Query      |  |   content    |             |
|  | - CRUD ops   |  | - Serialize  |  | - Parse      |             |
|  +--------------+  +--------------+  +--------------+             |
|           |                                                       |
|           v                                                       |
|  +-------------------------------------------------------------+  |
|  |                  Jira/Beads REST API                        |  |
|  +-------------------------------------------------------------+  |
+-------------------------------------------------------------------+
```

## Module Structure

```
src/twinscribe/beads/
+-- __init__.py        # Public exports
+-- client.py          # Low-level API client
+-- tracker.py         # Ticket tracking and state
+-- templates.py       # Ticket content rendering
+-- manager.py         # Lifecycle coordination
```

## Component Details

### BeadsClient (client.py)

Abstract base class for interacting with the Beads/Jira REST API.

```python
class BeadsClient(ABC):
    """Abstract base class for Beads/Jira API client."""

    async def initialize() -> None
    async def close() -> None
    async def create_issue(request: CreateIssueRequest) -> BeadsIssue
    async def get_issue(key: str) -> BeadsIssue
    async def update_issue(key: str, fields: dict) -> BeadsIssue
    async def add_comment(key: str, body: str) -> BeadsComment
    async def get_comments(key: str) -> list[BeadsComment]
    async def transition_issue(key: str, transition_name: str) -> BeadsIssue
    async def search_issues(jql: str) -> list[BeadsIssue]
```

#### Data Classes

| Class | Purpose |
|-------|---------|
| `BeadsClientConfig` | API configuration (server, auth, timeouts) |
| `BeadsIssue` | Issue representation with key, summary, status, etc. |
| `BeadsComment` | Comment representation with author, body, timestamps |
| `CreateIssueRequest` | Request object for issue creation |

#### Exception Hierarchy

```
BeadsError
+-- AuthenticationError    # Invalid credentials
+-- NotFoundError          # Issue/resource not found
+-- PermissionError        # Access denied
```

### TicketTracker (tracker.py)

Tracks Beads tickets and maps them to discrepancies and components.

```python
class TicketTracker:
    """Tracks Beads tickets and their resolution status."""

    def track(ticket_key, ticket_type, ...) -> TrackedTicket
    def get(ticket_key: str) -> Optional[TrackedTicket]
    def get_by_discrepancy(discrepancy_id: str) -> Optional[TrackedTicket]
    def get_by_component(component_id: str) -> list[TrackedTicket]
    def query(query: TicketQuery) -> list[TrackedTicket]
    def update_resolution(ticket_key, text, action) -> TrackedTicket
    def mark_applied(ticket_key: str) -> TrackedTicket
    def expire_ticket(ticket_key: str) -> TrackedTicket
```

#### Ticket States

```
                    +----------+
                    |  PENDING |
                    +----+-----+
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
    +-----------+  +-----------+  +-----------+
    |IN_PROGRESS|  | RESOLVED  |  |  EXPIRED  |
    +-----------+  +-----+-----+  +-----------+
          |              |
          v              v
    +-----------+  +-----------+
    | RESOLVED  |  |  APPLIED  |
    +-----------+  +-----------+
          |
          v
    +-----------+
    |  APPLIED  |
    +-----------+
```

| State | Description |
|-------|-------------|
| `PENDING` | Ticket created, awaiting human review |
| `IN_PROGRESS` | Human is actively working on resolution |
| `RESOLVED` | Human provided resolution, not yet applied |
| `APPLIED` | Resolution applied to documentation |
| `EXPIRED` | Timeout reached without resolution |
| `CANCELLED` | Ticket was cancelled |

#### TrackedTicket

```python
@dataclass
class TrackedTicket:
    ticket_key: str                # e.g., "LEGACY-123"
    ticket_type: TicketType        # DISCREPANCY or REBUILD
    status: TicketStatus
    discrepancy_id: Optional[str]  # Links to Discrepancy model
    component_id: Optional[str]    # Links to Component model
    created_at: datetime
    updated_at: datetime
    resolution_text: Optional[str]
    resolution_action: Optional[str]
    timeout_at: Optional[datetime]
    metadata: dict[str, Any]
```

### TicketTemplateEngine (templates.py)

Renders ticket content from templates using variable substitution.

```python
class TicketTemplateEngine:
    """Renders ticket content from templates."""

    def render_discrepancy(data, template_name) -> tuple[str, str]
    def render_rebuild(data, template_name) -> tuple[str, str]
    def register_template(template: TicketTemplate) -> None
    def get_labels(data, template_name) -> list[str]
    def get_priority(data, template_name) -> str
```

#### Template Data Classes

**DiscrepancyTemplateData**
```python
@dataclass
class DiscrepancyTemplateData:
    discrepancy_id: str
    component_name: str
    component_type: str
    file_path: str
    discrepancy_type: str
    stream_a_value: str
    stream_b_value: str
    static_analysis_value: Optional[str]
    context: str
    iteration: int
    previous_attempts: list[str]
    labels: list[str]
    priority: str
```

**RebuildTemplateData**
```python
@dataclass
class RebuildTemplateData:
    component_name: str
    component_type: str
    file_path: str
    documentation: str
    call_graph: dict[str, list[str]]
    dependencies: list[str]
    dependents: list[str]
    complexity_score: float
    rebuild_priority: int
    suggested_approach: str
    labels: list[str]
    epic_key: Optional[str]
```

#### Default Templates

The module provides default Jira-formatted templates for both ticket types:

1. **Discrepancy Template**: Shows both stream interpretations, static analysis value if available, context, and instructions for resolution
2. **Rebuild Template**: Shows component documentation, dependencies, call graph, and acceptance criteria

#### Resolution Parsing

```python
class ResolutionParser:
    """Parses human resolutions from ticket comments."""

    def parse(comment_text: str) -> Optional[tuple[str, str]]
    def is_resolution_comment(comment_text: str) -> bool
```

Expected resolution format in comments:
```
RESOLUTION: <accept_a|accept_b|merge|manual>
<explanation and/or corrected content>
```

### BeadsLifecycleManager (manager.py)

Coordinates the full lifecycle of Beads tickets.

```python
class BeadsLifecycleManager:
    """Manages the lifecycle of Beads tickets."""

    async def initialize() -> None
    async def close() -> None

    # Ticket Creation
    async def create_discrepancy_ticket(data) -> TrackedTicket
    async def create_rebuild_ticket(data) -> TrackedTicket
    async def create_rebuild_epic(name, components) -> str

    # Resolution Monitoring
    async def check_for_resolution(ticket_key) -> Optional[TicketResolution]
    async def wait_for_resolution(ticket_key, timeout) -> Optional[TicketResolution]
    async def start_monitoring() -> None
    async def stop_monitoring() -> None

    # Resolution Application
    async def apply_resolution(resolution, apply_func) -> ResolutionResult

    # State Management
    async def sync_from_beads(jql: str) -> int
    def get_statistics() -> dict
```

#### Configuration

```python
class ManagerConfig(BaseModel):
    project: str = "LEGACY_DOC"          # Discrepancy project
    rebuild_project: str = "REBUILD"      # Rebuild project
    poll_interval_seconds: int = 60       # Resolution poll interval
    timeout_hours: int = 48               # Ticket timeout
    auto_create_tickets: bool = True
    default_labels: list[str] = ["ai-documentation"]
    max_concurrent_polls: int = 10
```

## Data Flow

### Discrepancy Ticket Flow

```
Comparison Result      DiscrepancyTemplateData       CreateIssueRequest
(discrepancy) -------> (render template) ----------> (Beads API) ----+
                                                                     |
                                                                     v
                                                              BeadsIssue
                                                                     |
                                                                     v
      TrackedTicket <------------------------------------------+
           |
           | (monitoring loop)
           v
      Check Comments ----> Parse Resolution ----> TicketResolution
                                                        |
                                                        v
                                                Apply to Documentation
                                                        |
                                                        v
                                                  Mark APPLIED
```

### Rebuild Ticket Flow

```
Final Documentation     RebuildTemplateData         CreateIssueRequest
(component docs) -----> (render template) --------> (Beads API) ----+
                                                                     |
                                                                     v
                                                              BeadsIssue
                                                                     |
                                                                     v
      TrackedTicket <------------------------------------------+
           |
           | (optional: link to epic)
           v
      Ready for Manual Rebuild
```

## Integration Points

### With Convergence Module

The manager integrates with convergence when:
1. Comparison finds blocking discrepancies that need human resolution
2. Convergence criteria include waiting for Beads resolution
3. Timeout handling affects convergence decisions

```python
# In convergence loop
if discrepancy.requires_human_resolution:
    ticket = await beads_manager.create_discrepancy_ticket(
        DiscrepancyTemplateData(
            discrepancy_id=discrepancy.id,
            component_name=component.name,
            ...
        )
    )

    # Option 1: Wait synchronously
    resolution = await beads_manager.wait_for_resolution(
        ticket.ticket_key,
        timeout_seconds=config.beads_ticket_timeout_hours * 3600
    )

    # Option 2: Continue and check later
    beads_manager.on_resolution(handle_resolution_callback)
    await beads_manager.start_monitoring()
```

### With Configuration Module

Beads configuration is loaded from `BeadsConfig`:

```python
class BeadsConfig(BaseModel):
    enabled: bool = True
    server: str = ""
    project: str = "LEGACY_DOC"
    rebuild_project: str = "REBUILD"
    username: str = ""
    api_token_env: str = "JIRA_API_TOKEN"
    ticket_labels: list[str] = ["ai-documentation"]
    auto_create_tickets: bool = True
```

### With Output Module

Final documentation can be exported as rebuild tickets:

```python
# After convergence complete
for component in final_documentation.components:
    await beads_manager.create_rebuild_ticket(
        RebuildTemplateData(
            component_name=component.name,
            documentation=component.documentation,
            call_graph=component.call_graph,
            ...
        )
    )
```

## Resolution Actions

| Action | Description | Effect |
|--------|-------------|--------|
| `accept_a` | Use Stream A's interpretation | Apply A's value directly |
| `accept_b` | Use Stream B's interpretation | Apply B's value directly |
| `merge` | Combine both interpretations | Use provided merged content |
| `manual` | Override with manual content | Use provided manual content |

## Error Handling

### Transient Errors

- Network timeouts: Retry with exponential backoff
- Rate limiting: Respect Retry-After headers
- Server errors (5xx): Retry up to max_retries

### Permanent Errors

- Authentication failures: Raise `AuthenticationError`, require reconfiguration
- Permission denied: Raise `PermissionError`, may need project access
- Not found: Raise `NotFoundError`, ticket may have been deleted

### Timeout Handling

When a ticket times out without resolution:
1. Mark ticket as `EXPIRED` in tracker
2. Log warning with ticket details
3. Return to convergence loop for fallback handling
4. Options: auto-select based on static analysis, escalate, or fail

## Serialization

The tracker supports serialization for persistence and recovery:

```python
# Save state
state = tracker.to_dict()
with open("tracker_state.json", "w") as f:
    json.dump(state, f)

# Restore state
with open("tracker_state.json") as f:
    state = json.load(f)
tracker = TicketTracker.from_dict(state)
```

## Usage Examples

### Basic Discrepancy Workflow

```python
from twinscribe.beads import (
    BeadsLifecycleManager,
    ManagerConfig,
    DiscrepancyTemplateData,
)
from twinscribe.beads.client import BeadsClientConfig

# Initialize
client_config = BeadsClientConfig(
    server="https://your-org.atlassian.net",
    username="user@example.com",
    api_token=SecretStr("token"),
)
client = JiraBeadsClient(client_config)  # Concrete implementation

manager_config = ManagerConfig(
    project="LEGACY_DOC",
    timeout_hours=24,
)
manager = BeadsLifecycleManager(client, manager_config)
await manager.initialize()

# Create discrepancy ticket
data = DiscrepancyTemplateData(
    discrepancy_id="disc-001",
    component_name="calculate_total",
    component_type="function",
    file_path="/src/billing/calculator.py",
    discrepancy_type="call_graph",
    stream_a_value="calls: process_payment, log_transaction",
    stream_b_value="calls: process_payment, send_receipt",
    static_analysis_value="calls: process_payment, log_transaction",
    context="Stream B may have hallucinated send_receipt call",
)

ticket = await manager.create_discrepancy_ticket(data)
print(f"Created ticket: {ticket.ticket_key}")

# Wait for resolution
resolution = await manager.wait_for_resolution(ticket.ticket_key)
if resolution:
    print(f"Resolved with action: {resolution.action}")
    # Apply resolution...
else:
    print("Ticket timed out")
```

### Async Monitoring

```python
# Register callback for resolutions
def handle_resolution(resolution: TicketResolution):
    print(f"Ticket {resolution.ticket_key} resolved: {resolution.action}")
    # Queue for processing...

manager.on_resolution(handle_resolution)

# Start background monitoring
await manager.start_monitoring()

# ... continue with other work ...

# Stop when done
await manager.stop_monitoring()
```

### Creating Rebuild Tickets

```python
from twinscribe.beads import RebuildTemplateData

# Create epic for the rebuild project
epic_key = await manager.create_rebuild_epic(
    "Legacy Billing Module Rebuild",
    components=[...],
)

# Create individual rebuild tickets
for component in components:
    data = RebuildTemplateData(
        component_name=component.name,
        component_type=component.type.value,
        file_path=component.file_path,
        documentation=component.final_documentation,
        call_graph={
            "calls": component.call_graph.get_callees(component.id),
            "called_by": component.call_graph.get_callers(component.id),
        },
        dependencies=component.dependencies,
        rebuild_priority=component.rebuild_order,
        complexity_score=component.complexity,
        epic_key=epic_key,
    )

    await manager.create_rebuild_ticket(data)
```

## Security Considerations

1. **API Token Storage**: Tokens stored in environment variables, never in code or logs
2. **SecretStr Usage**: Pydantic `SecretStr` prevents accidental token exposure
3. **TLS Verification**: SSL verification enabled by default
4. **Minimal Permissions**: Client should use token with minimal required permissions
5. **Audit Trail**: All ticket operations are logged for audit purposes

## Testing Strategy

### Unit Tests
- Template rendering with various data combinations
- Resolution parsing with valid and invalid formats
- Tracker state management and serialization
- Error handling for various failure modes

### Integration Tests
- Mock Jira API for end-to-end workflows
- Timeout handling verification
- Concurrent polling behavior
- State recovery after restart

### End-to-End Tests
- Real Jira instance (test project)
- Full discrepancy workflow
- Rebuild ticket creation
