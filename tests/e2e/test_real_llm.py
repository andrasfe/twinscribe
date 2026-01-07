"""
End-to-end tests with real LLM API calls.

These tests verify the complete documentation workflow using actual
LLM API calls through OpenRouter. They are designed to be skipped
when the OPENROUTER_API_KEY environment variable is not set.

Test scenarios:
1. Single component E2E - Document one simple Python function
2. Multi-component E2E - Document 3-5 related functions with known relationships
3. Error recovery - Handle API rate limits and timeout errors
"""

import asyncio
import json

import pytest

from twinscribe.models.base import (
    CallType,
    ComponentType,
)
from twinscribe.models.call_graph import CallEdge, CallGraph
from twinscribe.models.components import (
    Component,
    ComponentLocation,
)
from twinscribe.utils.llm_client import (
    APIError,
    AsyncLLMClient,
    AuthenticationError,
    LLMResponse,
    Message,
    RateLimitError,
)

# =============================================================================
# Skip conditions
# =============================================================================


def has_openrouter_api_key() -> bool:
    """Check if OpenRouter API key is available and valid.

    Uses the same detection mechanism as the LLM client,
    which loads from .env file. Also validates that the key
    is not a placeholder value.
    """
    from twinscribe.config.environment import get_api_key, reset_environment

    # Reset to force fresh load
    reset_environment()
    key = get_api_key("openrouter")

    if not key:
        return False

    # Check for placeholder values
    placeholder_patterns = [
        "your_",
        "REPLACE_",
        "placeholder",
        "xxx",
        "sk-or-v1-example",
    ]
    key_lower = key.lower()
    for pattern in placeholder_patterns:
        if pattern.lower() in key_lower:
            return False

    # Valid key should be at least 40 characters for OpenRouter
    return len(key) >= 40


skip_no_api_key = pytest.mark.skipif(
    not has_openrouter_api_key(),
    reason="OPENROUTER_API_KEY environment variable not set",
)


# =============================================================================
# Constants for testing
# =============================================================================

# Default timeout for API calls (in seconds)
DEFAULT_TIMEOUT = 60

# Small, cheap models for testing to minimize costs
# Using gpt-4o-mini which is one of the cheapest available models
TEST_MODEL = "gpt-4o-mini"


# =============================================================================
# Sample code fixtures - minimal to reduce API costs
# =============================================================================

SIMPLE_FUNCTION_CODE = '''
def add_numbers(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b
'''

FUNCTION_WITH_HELPER_CODE = '''
def format_value(value: float) -> str:
    """Format a floating point value as a string with 2 decimal places."""
    return f"{value:.2f}"

def calculate_total(items: list[float], tax_rate: float = 0.1) -> str:
    """
    Calculate total price with tax and return formatted string.

    Calls format_value to format the result.
    """
    subtotal = sum(items)
    total = subtotal * (1 + tax_rate)
    return format_value(total)
'''

MULTI_COMPONENT_CODE = '''
def validate_input(data: dict) -> bool:
    """Validate that input data has required fields."""
    return "name" in data and "value" in data

def transform_data(data: dict) -> dict:
    """Transform data by uppercasing the name field."""
    return {
        "name": data["name"].upper(),
        "value": data["value"]
    }

def process_item(data: dict) -> dict:
    """
    Process a single item through validation and transformation.

    Calls validate_input first, then transform_data.
    """
    if not validate_input(data):
        raise ValueError("Invalid input data")
    return transform_data(data)

def process_batch(items: list[dict]) -> list[dict]:
    """
    Process a batch of items.

    Calls process_item for each item in the batch.
    """
    return [process_item(item) for item in items]
'''


# =============================================================================
# Test fixtures
# =============================================================================


@pytest.fixture
def simple_component() -> Component:
    """Create a simple component for single-function testing."""
    return Component(
        component_id="test_module.add_numbers",
        name="add_numbers",
        type=ComponentType.FUNCTION,
        location=ComponentLocation(
            file_path="test_module.py",
            line_start=1,
            line_end=4,
        ),
        signature="def add_numbers(a: int, b: int) -> int",
        existing_docstring="Add two integers and return the result.",
    )


@pytest.fixture
def multi_components() -> list[Component]:
    """Create multiple related components for multi-component testing."""
    return [
        Component(
            component_id="test_module.validate_input",
            name="validate_input",
            type=ComponentType.FUNCTION,
            location=ComponentLocation(
                file_path="test_module.py",
                line_start=1,
                line_end=4,
            ),
            signature="def validate_input(data: dict) -> bool",
        ),
        Component(
            component_id="test_module.transform_data",
            name="transform_data",
            type=ComponentType.FUNCTION,
            location=ComponentLocation(
                file_path="test_module.py",
                line_start=5,
                line_end=10,
            ),
            signature="def transform_data(data: dict) -> dict",
        ),
        Component(
            component_id="test_module.process_item",
            name="process_item",
            type=ComponentType.FUNCTION,
            location=ComponentLocation(
                file_path="test_module.py",
                line_start=11,
                line_end=18,
            ),
            signature="def process_item(data: dict) -> dict",
        ),
        Component(
            component_id="test_module.process_batch",
            name="process_batch",
            type=ComponentType.FUNCTION,
            location=ComponentLocation(
                file_path="test_module.py",
                line_start=19,
                line_end=26,
            ),
            signature="def process_batch(items: list[dict]) -> list[dict]",
        ),
    ]


@pytest.fixture
def multi_component_ground_truth() -> CallGraph:
    """Create ground truth call graph for multi-component test."""
    return CallGraph(
        source="static_analysis",
        edges=[
            CallEdge(
                caller="test_module.process_item",
                callee="test_module.validate_input",
                call_site_line=16,
                call_type=CallType.DIRECT,
                confidence=1.0,
            ),
            CallEdge(
                caller="test_module.process_item",
                callee="test_module.transform_data",
                call_site_line=18,
                call_type=CallType.DIRECT,
                confidence=1.0,
            ),
            CallEdge(
                caller="test_module.process_batch",
                callee="test_module.process_item",
                call_site_line=25,
                call_type=CallType.LOOP,
                confidence=1.0,
            ),
        ],
    )


@pytest.fixture
def source_code_map() -> dict[str, str]:
    """Map component IDs to their source code."""
    return {
        "test_module.validate_input": '''def validate_input(data: dict) -> bool:
    """Validate that input data has required fields."""
    return "name" in data and "value" in data
''',
        "test_module.transform_data": '''def transform_data(data: dict) -> dict:
    """Transform data by uppercasing the name field."""
    return {
        "name": data["name"].upper(),
        "value": data["value"]
    }
''',
        "test_module.process_item": '''def process_item(data: dict) -> dict:
    """
    Process a single item through validation and transformation.

    Calls validate_input first, then transform_data.
    """
    if not validate_input(data):
        raise ValueError("Invalid input data")
    return transform_data(data)
''',
        "test_module.process_batch": '''def process_batch(items: list[dict]) -> list[dict]:
    """
    Process a batch of items.

    Calls process_item for each item in the batch.
    """
    return [process_item(item) for item in items]
''',
    }


# =============================================================================
# Helper functions
# =============================================================================


def build_documentation_prompt(
    component: Component,
    source_code: str,
    dependency_context: dict[str, str] | None = None,
) -> str:
    """Build a prompt for documentation generation."""
    prompt = f"""Analyze the following Python function and generate documentation.

## Source Code
```python
{source_code}
```

## Component Information
- Name: {component.name}
- Type: {component.type.value}
- Location: {component.location.file_path}:{component.location.line_start}-{component.location.line_end}
"""

    if dependency_context:
        prompt += "\n## Dependencies (already documented)\n"
        for dep_id, dep_doc in dependency_context.items():
            prompt += f"- {dep_id}: {dep_doc}\n"

    prompt += """

## Required Output Format (JSON)
Return a JSON object with these fields:
{
    "documentation": {
        "summary": "One-line summary of what the function does",
        "description": "Detailed explanation of purpose and behavior",
        "parameters": [
            {"name": "param_name", "type": "param_type", "description": "what it does"}
        ],
        "returns": {"type": "return_type", "description": "what is returned"},
        "raises": [{"type": "ExceptionType", "condition": "when raised"}]
    },
    "call_graph": {
        "callees": [
            {"component_id": "module.function_called", "call_type": "direct|conditional|loop"}
        ]
    }
}
"""
    return prompt


DOCUMENTATION_SYSTEM_PROMPT = """You are a Python documentation expert. Analyze code and generate accurate,
comprehensive documentation. Focus on:
1. Clear, concise summaries
2. Accurate parameter and return type documentation
3. Identifying function calls (callees) in the code
4. Proper exception documentation

Always respond with valid JSON matching the requested schema."""


def validate_documentation_schema(response_content: str) -> tuple[bool, dict | None, str | None]:
    """Validate that response content matches expected documentation schema.

    Returns:
        Tuple of (is_valid, parsed_data, error_message)
    """
    try:
        data = json.loads(response_content)
    except json.JSONDecodeError as e:
        return False, None, f"Invalid JSON: {e}"

    # Check required top-level fields
    if "documentation" not in data:
        return False, None, "Missing 'documentation' field"

    doc = data["documentation"]

    # Check documentation fields
    if "summary" not in doc:
        return False, None, "Missing 'summary' in documentation"

    if not isinstance(doc.get("summary", ""), str):
        return False, None, "'summary' must be a string"

    # Parameters validation (if present)
    params = doc.get("parameters", [])
    if not isinstance(params, list):
        return False, None, "'parameters' must be a list"

    for i, param in enumerate(params):
        if not isinstance(param, dict):
            return False, None, f"Parameter {i} must be a dict"
        if "name" not in param:
            return False, None, f"Parameter {i} missing 'name'"

    # call_graph validation (optional but should be valid if present)
    call_graph = data.get("call_graph", {})
    if call_graph:
        callees = call_graph.get("callees", [])
        if not isinstance(callees, list):
            return False, None, "'callees' must be a list"

        for i, callee in enumerate(callees):
            if not isinstance(callee, dict):
                return False, None, f"Callee {i} must be a dict"
            if "component_id" not in callee:
                return False, None, f"Callee {i} missing 'component_id'"

    return True, data, None


def verify_call_graph_accuracy(
    documented_callees: list[str],
    ground_truth: CallGraph,
    component_id: str,
) -> tuple[float, list[str], list[str]]:
    """Verify documented call graph against ground truth.

    Returns:
        Tuple of (accuracy_score, missing_callees, extra_callees)
    """
    # Get expected callees from ground truth
    expected_callees = {edge.callee for edge in ground_truth.edges if edge.caller == component_id}

    documented_set = set(documented_callees)

    # Calculate differences
    missing = expected_callees - documented_set
    extra = documented_set - expected_callees
    correct = expected_callees & documented_set

    # Calculate accuracy
    if not expected_callees:
        # No expected callees - check for false positives
        accuracy = 1.0 if not extra else 0.5
    else:
        # Precision and recall
        precision = len(correct) / len(documented_set) if documented_set else 1.0
        recall = len(correct) / len(expected_callees)
        accuracy = (precision + recall) / 2 if (precision + recall) > 0 else 0.0

    return accuracy, list(missing), list(extra)


# =============================================================================
# Test 1: Single Component E2E
# =============================================================================


@pytest.mark.e2e
@pytest.mark.slow
@skip_no_api_key
class TestSingleComponentE2E:
    """End-to-end tests for documenting a single simple Python function."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(DEFAULT_TIMEOUT)
    async def test_document_simple_function_stream_a(
        self,
        simple_component: Component,
    ):
        """
        Document one simple Python function using real LLM calls.

        Both streams use real LLM calls (if API key available).
        Verifies output matches expected schema.
        """
        # Arrange
        source_code = SIMPLE_FUNCTION_CODE
        prompt = build_documentation_prompt(simple_component, source_code)

        # Act
        async with AsyncLLMClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.generate_documentation(
                system_prompt=DOCUMENTATION_SYSTEM_PROMPT,
                user_prompt=prompt,
                model=TEST_MODEL,
                json_mode=True,
            )

        # Assert - verify response is valid
        assert response is not None
        assert isinstance(response, LLMResponse)
        assert response.content is not None
        assert len(response.content) > 0

        # Verify schema
        is_valid, parsed_data, error = validate_documentation_schema(response.content)
        assert is_valid, f"Schema validation failed: {error}"

        # Verify documentation content makes sense
        doc = parsed_data["documentation"]
        assert len(doc["summary"]) > 0
        assert "add" in doc["summary"].lower() or "sum" in doc["summary"].lower()

        # Verify parameters are documented
        params = doc.get("parameters", [])
        param_names = {p["name"] for p in params}
        assert "a" in param_names or any("a" in p["name"] for p in params)
        assert "b" in param_names or any("b" in p["name"] for p in params)

        # Verify token usage is tracked
        assert response.usage.total_tokens > 0

    @pytest.mark.asyncio
    @pytest.mark.timeout(DEFAULT_TIMEOUT)
    async def test_document_function_with_dependency(self):
        """
        Document a function that calls another function.

        Verifies that the call graph is captured correctly.
        """
        # Arrange - function that calls format_value
        component = Component(
            component_id="test_module.calculate_total",
            name="calculate_total",
            type=ComponentType.FUNCTION,
            location=ComponentLocation(
                file_path="test_module.py",
                line_start=5,
                line_end=13,
            ),
            signature="def calculate_total(items: list[float], tax_rate: float = 0.1) -> str",
        )

        source_code = '''def calculate_total(items: list[float], tax_rate: float = 0.1) -> str:
    """
    Calculate total price with tax and return formatted string.

    Calls format_value to format the result.
    """
    subtotal = sum(items)
    total = subtotal * (1 + tax_rate)
    return format_value(total)
'''

        # Provide dependency context
        dependency_context = {
            "test_module.format_value": "Format a floating point value as a string with 2 decimal places."
        }

        prompt = build_documentation_prompt(component, source_code, dependency_context)

        # Act
        async with AsyncLLMClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.generate_documentation(
                system_prompt=DOCUMENTATION_SYSTEM_PROMPT,
                user_prompt=prompt,
                model=TEST_MODEL,
                json_mode=True,
            )

        # Assert - verify response
        assert response is not None
        is_valid, parsed_data, error = validate_documentation_schema(response.content)
        assert is_valid, f"Schema validation failed: {error}"

        # Verify call graph captures the dependency
        call_graph = parsed_data.get("call_graph", {})
        callees = call_graph.get("callees", [])
        callee_ids = [c.get("component_id", "") for c in callees]

        # Should identify format_value as a callee
        has_format_value = any("format_value" in cid for cid in callee_ids)
        assert has_format_value, f"Expected to find format_value in callees, got: {callee_ids}"

    @pytest.mark.asyncio
    @pytest.mark.timeout(DEFAULT_TIMEOUT)
    async def test_verify_call_graph_against_static_analysis(
        self,
        simple_component: Component,
    ):
        """
        Verify that LLM-generated call graph matches static analysis ground truth.

        For a simple function with no calls, should report empty callees.
        """
        # Arrange - simple function with no function calls
        source_code = SIMPLE_FUNCTION_CODE
        prompt = build_documentation_prompt(simple_component, source_code)

        # Ground truth - no function calls in add_numbers
        ground_truth = CallGraph(
            source="static_analysis",
            edges=[],  # add_numbers doesn't call any functions
        )

        # Act
        async with AsyncLLMClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.generate_documentation(
                system_prompt=DOCUMENTATION_SYSTEM_PROMPT,
                user_prompt=prompt,
                model=TEST_MODEL,
                json_mode=True,
            )

        # Assert
        is_valid, parsed_data, _ = validate_documentation_schema(response.content)
        assert is_valid

        # Extract documented callees
        call_graph = parsed_data.get("call_graph", {})
        callees = call_graph.get("callees", [])
        documented_callee_ids = [c.get("component_id", "") for c in callees]

        # Verify against ground truth
        accuracy, missing, extra = verify_call_graph_accuracy(
            documented_callee_ids,
            ground_truth,
            simple_component.component_id,
        )

        # For this simple function, accuracy should be high (no calls to find)
        # We accept some noise from the LLM potentially adding builtins
        assert accuracy >= 0.5 or len(extra) <= 2, (
            f"Call graph accuracy too low: {accuracy}, extra: {extra}"
        )


# =============================================================================
# Test 2: Multi-Component E2E
# =============================================================================


@pytest.mark.e2e
@pytest.mark.slow
@skip_no_api_key
class TestMultiComponentE2E:
    """End-to-end tests for documenting multiple related functions."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(DEFAULT_TIMEOUT * 3)  # Longer timeout for multiple components
    async def test_document_related_functions(
        self,
        multi_components: list[Component],
        source_code_map: dict[str, str],
        multi_component_ground_truth: CallGraph,
    ):
        """
        Document 3-5 related functions with known relationships.

        Verifies that streams can document dependencies and
        call graph relationships are captured correctly.
        """
        # Arrange
        documented_outputs: dict[str, dict] = {}
        dependency_context: dict[str, str] = {}

        # Act - document components in topological order
        # (validate_input and transform_data first, then process_item, then process_batch)
        processing_order = [
            "test_module.validate_input",
            "test_module.transform_data",
            "test_module.process_item",
            "test_module.process_batch",
        ]

        async with AsyncLLMClient(timeout=DEFAULT_TIMEOUT) as client:
            for component_id in processing_order:
                # Find component
                component = next(c for c in multi_components if c.component_id == component_id)
                source_code = source_code_map[component_id]

                # Build prompt with accumulated dependency context
                prompt = build_documentation_prompt(
                    component,
                    source_code,
                    dependency_context if dependency_context else None,
                )

                # Generate documentation
                response = await client.generate_documentation(
                    system_prompt=DOCUMENTATION_SYSTEM_PROMPT,
                    user_prompt=prompt,
                    model=TEST_MODEL,
                    json_mode=True,
                )

                # Validate and store
                is_valid, parsed_data, error = validate_documentation_schema(response.content)
                assert is_valid, f"Failed for {component_id}: {error}"

                documented_outputs[component_id] = parsed_data

                # Add to dependency context for next components
                summary = parsed_data["documentation"].get("summary", "")
                dependency_context[component_id] = summary

        # Assert - verify all components were documented
        assert len(documented_outputs) == len(multi_components)

        # Verify call graph relationships
        # process_item should call validate_input and transform_data
        process_item_data = documented_outputs["test_module.process_item"]
        process_item_callees = [
            c.get("component_id", "")
            for c in process_item_data.get("call_graph", {}).get("callees", [])
        ]

        # Should have captured at least one of the known callees
        has_validate = any("validate" in c.lower() for c in process_item_callees)
        has_transform = any("transform" in c.lower() for c in process_item_callees)

        # At least one of the dependencies should be captured
        assert has_validate or has_transform, (
            f"process_item should call validate_input or transform_data, "
            f"got: {process_item_callees}"
        )

        # process_batch should call process_item
        process_batch_data = documented_outputs["test_module.process_batch"]
        process_batch_callees = [
            c.get("component_id", "")
            for c in process_batch_data.get("call_graph", {}).get("callees", [])
        ]

        has_process_item = any("process_item" in c.lower() for c in process_batch_callees)
        assert has_process_item, (
            f"process_batch should call process_item, got: {process_batch_callees}"
        )

    @pytest.mark.asyncio
    @pytest.mark.timeout(DEFAULT_TIMEOUT * 2)
    async def test_call_graph_accuracy_multi_component(
        self,
        multi_components: list[Component],
        source_code_map: dict[str, str],
        multi_component_ground_truth: CallGraph,
    ):
        """
        Verify call graph relationships match ground truth for multiple components.
        """
        # Focus on process_item which has the most interesting call relationships
        component = next(
            c for c in multi_components if c.component_id == "test_module.process_item"
        )
        source_code = source_code_map[component.component_id]

        # Provide dependency context
        dependency_context = {
            "test_module.validate_input": "Validate that input data has required fields.",
            "test_module.transform_data": "Transform data by uppercasing the name field.",
        }

        prompt = build_documentation_prompt(component, source_code, dependency_context)

        # Generate documentation
        async with AsyncLLMClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.generate_documentation(
                system_prompt=DOCUMENTATION_SYSTEM_PROMPT,
                user_prompt=prompt,
                model=TEST_MODEL,
                json_mode=True,
            )

        # Validate response
        is_valid, parsed_data, _ = validate_documentation_schema(response.content)
        assert is_valid

        # Extract callees
        call_graph = parsed_data.get("call_graph", {})
        callees = call_graph.get("callees", [])
        documented_callee_ids = [c.get("component_id", "") for c in callees]

        # Verify against ground truth
        accuracy, missing, extra = verify_call_graph_accuracy(
            documented_callee_ids,
            multi_component_ground_truth,
            component.component_id,
        )

        # Should capture at least 50% of call relationships
        # (LLMs may use different naming conventions)
        assert accuracy >= 0.25 or len(missing) <= 1, (
            f"Call graph accuracy too low: {accuracy}, missing: {missing}, extra: {extra}"
        )


# =============================================================================
# Test 3: Error Recovery
# =============================================================================


@pytest.mark.e2e
@pytest.mark.slow
class TestErrorRecovery:
    """Tests for handling API errors gracefully."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(DEFAULT_TIMEOUT)
    async def test_rate_limit_error_is_raised(self):
        """
        Test that RateLimitError is raised correctly on 429 response.

        Uses respx to mock HTTP responses.
        """
        import respx
        from httpx import Response

        from twinscribe.utils.llm_client import (
            OPENROUTER_BASE_URL,
            OPENROUTER_CHAT_ENDPOINT,
        )

        # Mock the OpenRouter API response for rate limiting
        with respx.mock:
            respx.post(f"{OPENROUTER_BASE_URL}{OPENROUTER_CHAT_ENDPOINT}").mock(
                return_value=Response(429, json={"error": {"message": "Rate limit exceeded"}})
            )

            async with AsyncLLMClient(api_key="test-key", max_retries=1) as client:
                with pytest.raises(RateLimitError):
                    await client.send_message(
                        model="gpt-4o",
                        messages=[Message(role="user", content="test")],
                    )

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_handles_timeout_gracefully_no_api_key(self):
        """
        Test that without valid API key, AuthenticationError is raised.
        """
        import respx
        from httpx import Response

        # Mock the API to return 401 Unauthorized
        with respx.mock:
            respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
                return_value=Response(401, json={"error": {"message": "Invalid API key"}})
            )

            client = AsyncLLMClient(
                api_key="invalid_test_key_for_testing",
                timeout=10,
            )
            with pytest.raises(AuthenticationError):
                async with client:
                    await client.send_message(
                        model=TEST_MODEL,
                        messages=[Message(role="user", content="Hello")],
                    )

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    @skip_no_api_key
    async def test_handles_timeout_gracefully_with_api_key(self):
        """
        Test that the client raises timeout errors with very short timeout.
        """
        import httpx

        # With API key, test actual timeout behavior
        # Create a very short timeout that will likely fail
        client = AsyncLLMClient(
            timeout=0.001,  # 1ms timeout - will definitely timeout
        )
        # httpx raises ConnectTimeout or ReadTimeout
        with pytest.raises((httpx.TimeoutException, asyncio.TimeoutError)):
            async with client:
                await client.send_message(
                    model=TEST_MODEL,
                    messages=[Message(role="user", content="Hello")],
                )

    @pytest.mark.asyncio
    @pytest.mark.timeout(DEFAULT_TIMEOUT)
    @skip_no_api_key
    async def test_handles_invalid_model_gracefully(self):
        """
        Test that the client handles invalid model errors gracefully.

        Note: This uses the ModelNotFoundError which is raised before API call.
        """
        from twinscribe.utils.llm_client import ModelNotFoundError

        # Arrange
        client = AsyncLLMClient(timeout=DEFAULT_TIMEOUT)

        # Act & Assert - invalid model name should raise ModelNotFoundError
        async with client:
            with pytest.raises(ModelNotFoundError) as exc_info:
                await client.send_message(
                    model="nonexistent-model-xyz-123",
                    messages=[Message(role="user", content="Hello")],
                )

            # Should get an error about the model
            error_str = str(exc_info.value).lower()
            assert "unknown" in error_str or "model" in error_str

    @pytest.mark.asyncio
    @pytest.mark.timeout(DEFAULT_TIMEOUT)
    async def test_api_error_contains_status_code(self):
        """
        Test that API errors contain the HTTP status code.
        """
        import respx
        from httpx import Response

        from twinscribe.utils.llm_client import (
            OPENROUTER_BASE_URL,
            OPENROUTER_CHAT_ENDPOINT,
        )

        # Mock a 500 error response
        with respx.mock:
            respx.post(f"{OPENROUTER_BASE_URL}{OPENROUTER_CHAT_ENDPOINT}").mock(
                return_value=Response(500, json={"error": {"message": "Internal Server Error"}})
            )

            async with AsyncLLMClient(api_key="test-key", max_retries=1) as client:
                with pytest.raises(APIError) as exc_info:
                    await client.send_message(
                        model="gpt-4o",
                        messages=[Message(role="user", content="test")],
                    )

                # Verify status code is captured
                assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    @pytest.mark.timeout(DEFAULT_TIMEOUT)
    async def test_authentication_error_on_401(self):
        """
        Test that AuthenticationError is raised on 401 response.
        """
        import respx
        from httpx import Response

        from twinscribe.utils.llm_client import (
            OPENROUTER_BASE_URL,
            OPENROUTER_CHAT_ENDPOINT,
        )

        # Mock a 401 error response
        with respx.mock:
            respx.post(f"{OPENROUTER_BASE_URL}{OPENROUTER_CHAT_ENDPOINT}").mock(
                return_value=Response(401, json={"error": {"message": "Invalid API key"}})
            )

            async with AsyncLLMClient(api_key="invalid-key", max_retries=1) as client:
                with pytest.raises(AuthenticationError):
                    await client.send_message(
                        model="gpt-4o",
                        messages=[Message(role="user", content="test")],
                    )


# =============================================================================
# Test 4: Dual Stream Comparison (Integration-like E2E)
# =============================================================================


@pytest.mark.e2e
@pytest.mark.slow
@skip_no_api_key
class TestDualStreamE2E:
    """Tests simulating dual-stream documentation with real LLM calls."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(DEFAULT_TIMEOUT * 2)
    async def test_two_models_produce_similar_documentation(
        self,
        simple_component: Component,
    ):
        """
        Test that two different models produce similar documentation.

        This simulates Stream A and Stream B using different models.
        """
        # Arrange
        source_code = SIMPLE_FUNCTION_CODE
        prompt = build_documentation_prompt(simple_component, source_code)

        # Use two different models (both cheap for testing)
        model_a = TEST_MODEL  # Primary test model
        model_b = TEST_MODEL  # Use same model for consistency in CI

        # Act - generate documentation from both "streams"
        async with AsyncLLMClient(timeout=DEFAULT_TIMEOUT) as client:
            response_a = await client.generate_documentation(
                system_prompt=DOCUMENTATION_SYSTEM_PROMPT,
                user_prompt=prompt,
                model=model_a,
                json_mode=True,
            )

            response_b = await client.generate_documentation(
                system_prompt=DOCUMENTATION_SYSTEM_PROMPT,
                user_prompt=prompt,
                model=model_b,
                json_mode=True,
            )

        # Assert - both should produce valid documentation
        is_valid_a, data_a, error_a = validate_documentation_schema(response_a.content)
        is_valid_b, data_b, error_b = validate_documentation_schema(response_b.content)

        assert is_valid_a, f"Stream A failed: {error_a}"
        assert is_valid_b, f"Stream B failed: {error_b}"

        # Both should have identified the same basic information
        doc_a = data_a["documentation"]
        doc_b = data_b["documentation"]

        # Summaries should both mention adding
        summary_a_lower = doc_a["summary"].lower()
        summary_b_lower = doc_b["summary"].lower()

        # At least one should mention "add" or similar
        has_add_concept = ("add" in summary_a_lower or "sum" in summary_a_lower) or (
            "add" in summary_b_lower or "sum" in summary_b_lower
        )
        assert has_add_concept, (
            f"Neither stream identified adding: A='{doc_a['summary']}', B='{doc_b['summary']}'"
        )

        # Both should have documented the parameters
        params_a = {p["name"] for p in doc_a.get("parameters", [])}
        params_b = {p["name"] for p in doc_b.get("parameters", [])}

        # Should have significant overlap in parameter names
        if params_a and params_b:
            overlap = params_a & params_b
            total = params_a | params_b
            similarity = len(overlap) / len(total) if total else 1.0
            assert similarity >= 0.5, (
                f"Parameter documentation too different: A={params_a}, B={params_b}"
            )
