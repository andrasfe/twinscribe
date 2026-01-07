"""
CrossCheck Verification Framework Test Fixtures

This module provides pytest fixtures specifically for testing the verification
strategies. All fixtures are designed to avoid real API calls and provide
deterministic, reproducible test behavior.

Fixture Categories:
- Mock LLM Clients: Simulated LLM responses for verification strategies
- Sample Code Components: Python code snippets for verification testing
- Documentation Samples: Sample documentation for verification testing
- Question/Answer Fixtures: Pre-computed Q&A data for interrogation tests
- Mask/Reconstruction Fixtures: Masked code and expected reconstructions
- Scenario Fixtures: Execution scenarios for walkthrough tests
- Mutation Fixtures: Code mutations for detection tests
- Expected Results: Pre-computed verification results for assertions
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# =============================================================================
# Data Classes for Verification Test Models
# =============================================================================


@dataclass
class MockCodeComponent:
    """Mock code component for verification testing."""

    id: str
    name: str
    language: str = "python"
    source_code: str = ""
    file_path: str = "sample_module.py"
    line_start: int = 1
    line_end: int = 10


@dataclass
class MockQuestion:
    """Mock question for Q&A interrogation testing."""

    id: str
    text: str
    category: str
    correct_answer: str
    documentation_gap: str = ""


@dataclass
class MockQAResult:
    """Mock Q&A evaluation result."""

    team_a_correct: bool
    team_b_correct: bool
    documentation_gap: str | None
    severity: str = "low"


@dataclass
class MockMask:
    """Mock mask for reconstruction testing."""

    id: str
    start: int
    end: int
    original: str
    mask_type: str


@dataclass
class MockMaskedChallenge:
    """Mock masked code challenge."""

    component_id: str
    masked_code: str
    masks: list[MockMask]
    original_code: str


@dataclass
class MockScenario:
    """Mock execution scenario for walkthrough testing."""

    id: str
    description: str
    input_values: dict[str, Any]
    expected_call_sequence: list[str]
    expected_return: Any
    expected_side_effects: list[str]


@dataclass
class MockMutation:
    """Mock code mutation for detection testing."""

    id: str
    original_code: str
    mutated_code: str
    mutation_type: str
    location: str
    expected_detectable: bool


@dataclass
class MockImpactChallenge:
    """Mock impact analysis challenge."""

    component_id: str
    change_description: str
    change_type: str
    actual_impacted: list[str]


@dataclass
class MockDocumentationGap:
    """Mock documentation gap finding."""

    mask_type: str
    original_value: str
    severity: str
    recommendation: str


@dataclass
class MockVerificationScores:
    """Mock verification scores for testing."""

    qa_score: float = 0.0
    reconstruction_score: float = 0.0
    scenario_score: float = 0.0
    mutation_score: float = 0.0
    impact_score: float = 0.0
    adversarial_findings: int = 0
    test_pass_rate: float = 0.0


# =============================================================================
# Mock LLM Client Fixtures for Verification
# =============================================================================


class MockVerificationLLMResponse:
    """Mock LLM response for verification strategy calls."""

    def __init__(self, content: str, model: str = "claude-opus-4-5"):
        self.content = content
        self.model = model
        self.id = "mock-verification-response-id"
        self.created = int(datetime.now(UTC).timestamp())
        self.usage = MagicMock(prompt_tokens=200, completion_tokens=100, total_tokens=300)


@pytest.fixture
def mock_examiner_client() -> MagicMock:
    """
    Create a mock LLM client for the examiner (Agent C) role.

    The examiner generates questions, evaluates answers, and creates
    verification challenges.
    """
    client = MagicMock()

    async def mock_generate(prompt: str) -> str:
        """Return mock responses based on prompt content."""
        if "generate" in prompt.lower() and "question" in prompt.lower():
            return json.dumps(
                [
                    {
                        "id": "q1",
                        "text": "What happens if process_data() receives an empty list?",
                        "category": "edge_case",
                        "correct_answer": "Returns an empty dictionary",
                        "documentation_gap": "Edge case handling not documented",
                    }
                ]
            )
        elif "evaluate" in prompt.lower():
            return json.dumps(
                {
                    "team_a_correct": True,
                    "team_b_correct": False,
                    "documentation_gap": "Missing edge case documentation",
                    "severity": "medium",
                }
            )
        else:
            return json.dumps({"status": "success"})

    client.generate = AsyncMock(side_effect=mock_generate)
    return client


@pytest.fixture
def mock_team_client() -> MagicMock:
    """
    Create a mock LLM client for team (Agent A/B) responses.

    Teams answer questions and reconstruct code using only their documentation.
    """
    client = MagicMock()

    async def mock_answer(prompt: str) -> str:
        """Return mock team answers."""
        return json.dumps(
            {
                "answer": "Returns an empty dictionary when given an empty list",
                "confidence": 0.85,
                "documentation_reference": "Section 3.2: Edge Cases",
            }
        )

    client.generate = AsyncMock(side_effect=mock_answer)
    return client


# =============================================================================
# Sample Code Component Fixtures
# =============================================================================


@pytest.fixture
def sample_verification_function() -> MockCodeComponent:
    """Return a sample function for verification testing."""
    return MockCodeComponent(
        id="sample_module.calculate_discount",
        name="calculate_discount",
        language="python",
        source_code='''def calculate_discount(price: float, customer_type: str, quantity: int) -> float:
    """Calculate final price with discounts applied.

    Args:
        price: Base price of the item
        customer_type: Either 'premium' or 'standard'
        quantity: Number of items purchased

    Returns:
        Final price after applying discounts

    Raises:
        ValueError: If price is negative or quantity is less than 1
    """
    if price < 0:
        raise ValueError("Price cannot be negative")
    if quantity < 1:
        raise ValueError("Quantity must be at least 1")

    if customer_type == "premium":
        base_discount = 0.2
    else:
        base_discount = 0.1

    if quantity > 100:
        volume_bonus = 0.05
    else:
        volume_bonus = 0

    return price * quantity * (1 - base_discount - volume_bonus)
''',
        file_path="sample_module.py",
        line_start=1,
        line_end=28,
    )


@pytest.fixture
def sample_verification_class() -> MockCodeComponent:
    """Return a sample class for verification testing."""
    return MockCodeComponent(
        id="sample_module.OrderProcessor",
        name="OrderProcessor",
        language="python",
        source_code='''class OrderProcessor:
    """Process customer orders with validation and tracking."""

    def __init__(self, db_connection, logger=None):
        self.db = db_connection
        self.logger = logger or default_logger
        self._order_count = 0

    def process_order(self, order: dict) -> dict:
        """Process a customer order.

        Args:
            order: Order dictionary with 'items', 'customer_id', 'payment_method'

        Returns:
            Order result with 'order_id', 'status', 'total'

        Raises:
            ValidationError: If order is malformed
            PaymentError: If payment processing fails
        """
        self._validate_order(order)
        total = self._calculate_total(order['items'])
        payment_result = self._process_payment(order['payment_method'], total)

        if payment_result['success']:
            order_id = self._save_order(order, total)
            self._order_count += 1
            self.logger.info(f"Order {order_id} processed successfully")
            return {'order_id': order_id, 'status': 'completed', 'total': total}
        else:
            raise PaymentError(payment_result['error'])

    def _validate_order(self, order: dict) -> None:
        """Validate order structure."""
        required = ['items', 'customer_id', 'payment_method']
        for field in required:
            if field not in order:
                raise ValidationError(f"Missing required field: {field}")
        if not order['items']:
            raise ValidationError("Order must have at least one item")

    def _calculate_total(self, items: list) -> float:
        """Calculate order total from items."""
        return sum(item['price'] * item['quantity'] for item in items)

    def _process_payment(self, method: str, amount: float) -> dict:
        """Process payment through payment gateway."""
        # Payment gateway integration
        return self.db.process_payment(method, amount)

    def _save_order(self, order: dict, total: float) -> str:
        """Save order to database."""
        return self.db.insert_order({**order, 'total': total})
''',
        file_path="order_processor.py",
        line_start=1,
        line_end=52,
    )


@pytest.fixture
def sample_async_component() -> MockCodeComponent:
    """Return a sample async function for verification testing."""
    return MockCodeComponent(
        id="sample_module.fetch_user_data",
        name="fetch_user_data",
        language="python",
        source_code='''async def fetch_user_data(user_id: str, include_orders: bool = False) -> dict:
    """Fetch user data from the API.

    Args:
        user_id: The unique user identifier
        include_orders: Whether to include order history

    Returns:
        User data dictionary with profile and optionally orders

    Raises:
        UserNotFoundError: If user_id doesn't exist
        APITimeoutError: If API request times out
    """
    async with aiohttp.ClientSession() as session:
        user_response = await session.get(f"/api/users/{user_id}")
        if user_response.status == 404:
            raise UserNotFoundError(f"User {user_id} not found")

        user_data = await user_response.json()

        if include_orders:
            orders_response = await session.get(f"/api/users/{user_id}/orders")
            user_data['orders'] = await orders_response.json()

        return user_data
''',
        file_path="api_client.py",
        line_start=1,
        line_end=26,
    )


# =============================================================================
# Documentation Sample Fixtures
# =============================================================================


@pytest.fixture
def sample_documentation_complete() -> dict:
    """Return complete documentation for the sample function."""
    return {
        "component_id": "sample_module.calculate_discount",
        "summary": "Calculate final price with discounts applied.",
        "description": (
            "Applies customer-type based discounts and volume bonuses "
            "to calculate the final price. Premium customers receive "
            "20% discount, standard customers receive 10%. Orders over "
            "100 items get an additional 5% volume bonus."
        ),
        "parameters": [
            {
                "name": "price",
                "type": "float",
                "description": "Base price of the item (must be non-negative)",
            },
            {
                "name": "customer_type",
                "type": "str",
                "description": "Either 'premium' or 'standard'",
            },
            {
                "name": "quantity",
                "type": "int",
                "description": "Number of items purchased (must be at least 1)",
            },
        ],
        "returns": {"type": "float", "description": "Final price after applying all discounts"},
        "raises": [
            {"type": "ValueError", "condition": "If price is negative"},
            {"type": "ValueError", "condition": "If quantity is less than 1"},
        ],
        "edge_cases": [
            "Empty order (quantity=0) raises ValueError",
            "Negative price raises ValueError",
            "Volume bonus only applies when quantity > 100",
        ],
    }


@pytest.fixture
def sample_documentation_incomplete() -> dict:
    """Return incomplete documentation with gaps for testing."""
    return {
        "component_id": "sample_module.calculate_discount",
        "summary": "Calculates discounts for orders.",
        "description": "Applies discounts based on customer type.",
        "parameters": [
            {"name": "price", "type": "float", "description": "Item price"},
            {"name": "customer_type", "type": "str", "description": "Type of customer"},
            {"name": "quantity", "type": "int", "description": "Number of items"},
        ],
        "returns": {"type": "float", "description": "Discounted price"},
        "raises": [],  # Missing exception documentation
        "edge_cases": [],  # Missing edge case documentation
    }


@pytest.fixture
def sample_documentation_incorrect() -> dict:
    """Return documentation with deliberate errors for testing."""
    return {
        "component_id": "sample_module.calculate_discount",
        "summary": "Calculate discounts for premium customers only.",  # Incorrect
        "description": (
            "Applies a flat 15% discount to all orders. "  # Incorrect discount values
            "Volume bonus of 10% applies for orders over 50 items."  # Wrong threshold
        ),
        "parameters": [
            {"name": "price", "type": "int", "description": "Item price"},  # Wrong type
            {"name": "customer_type", "type": "str", "description": "Customer level"},
            {"name": "quantity", "type": "int", "description": "Order quantity"},
        ],
        "returns": {
            "type": "int",  # Wrong return type
            "description": "Discounted total",
        },
        "raises": [
            {"type": "TypeError", "condition": "If inputs are wrong type"}  # Wrong exception
        ],
        "edge_cases": [],
    }


# =============================================================================
# Q&A Interrogation Fixtures
# =============================================================================


@pytest.fixture
def sample_questions() -> list[MockQuestion]:
    """Return sample questions for Q&A interrogation testing."""
    return [
        MockQuestion(
            id="q1",
            text="What happens if calculate_discount() receives a negative price?",
            category="error_handling",
            correct_answer="Raises ValueError with message 'Price cannot be negative'",
            documentation_gap="Exception handling not fully documented",
        ),
        MockQuestion(
            id="q2",
            text="What discount does a premium customer receive?",
            category="return_value",
            correct_answer="20% base discount (0.2)",
            documentation_gap="",
        ),
        MockQuestion(
            id="q3",
            text="At what quantity threshold is the volume bonus applied?",
            category="edge_case",
            correct_answer="Volume bonus of 5% applies when quantity > 100",
            documentation_gap="Threshold value not clearly specified",
        ),
        MockQuestion(
            id="q4",
            text="What is the total discount for a premium customer with 150 items?",
            category="edge_case",
            correct_answer="25% (20% base + 5% volume bonus)",
            documentation_gap="",
        ),
        MockQuestion(
            id="q5",
            text="What functions does calculate_discount call internally?",
            category="call_flow",
            correct_answer="No internal function calls, only uses built-in operators",
            documentation_gap="",
        ),
    ]


@pytest.fixture
def sample_qa_results() -> list[MockQAResult]:
    """Return sample Q&A evaluation results."""
    return [
        MockQAResult(
            team_a_correct=True, team_b_correct=True, documentation_gap=None, severity="low"
        ),
        MockQAResult(
            team_a_correct=True,
            team_b_correct=False,
            documentation_gap="Team B incorrectly stated 15% discount",
            severity="medium",
        ),
        MockQAResult(
            team_a_correct=False,
            team_b_correct=False,
            documentation_gap="Neither team documented the volume threshold",
            severity="high",
        ),
    ]


# =============================================================================
# Masked Reconstruction Fixtures
# =============================================================================


@pytest.fixture
def sample_masked_challenge() -> MockMaskedChallenge:
    """Return a sample masked code challenge."""
    return MockMaskedChallenge(
        component_id="sample_module.calculate_discount",
        masked_code="""def calculate_discount(price: float, customer_type: str, quantity: int) -> float:
    if price < 0:
        raise ValueError("Price cannot be negative")
    if quantity < 1:
        raise ValueError("Quantity must be at least 1")

    if customer_type == "premium":
        base_discount = MASKED_VALUE_1
    else:
        base_discount = MASKED_VALUE_2

    if quantity > MASKED_THRESHOLD:
        volume_bonus = MASKED_VALUE_3
    else:
        volume_bonus = 0

    return MASKED_EXPRESSION
""",
        masks=[
            MockMask(id="m1", start=280, end=283, original="0.2", mask_type="constants"),
            MockMask(id="m2", start=340, end=343, original="0.1", mask_type="constants"),
            MockMask(id="m3", start=380, end=383, original="100", mask_type="constants"),
            MockMask(id="m4", start=420, end=424, original="0.05", mask_type="constants"),
            MockMask(
                id="m5",
                start=500,
                end=550,
                original="price * quantity * (1 - base_discount - volume_bonus)",
                mask_type="returns",
            ),
        ],
        original_code="""def calculate_discount(price: float, customer_type: str, quantity: int) -> float:
    if price < 0:
        raise ValueError("Price cannot be negative")
    if quantity < 1:
        raise ValueError("Quantity must be at least 1")

    if customer_type == "premium":
        base_discount = 0.2
    else:
        base_discount = 0.1

    if quantity > 100:
        volume_bonus = 0.05
    else:
        volume_bonus = 0

    return price * quantity * (1 - base_discount - volume_bonus)
""",
    )


@pytest.fixture
def sample_reconstruction_team_a() -> dict:
    """Return Team A's reconstruction attempt."""
    return {
        "m1": "0.2",  # Correct
        "m2": "0.1",  # Correct
        "m3": "100",  # Correct
        "m4": "0.05",  # Correct
        "m5": "price * quantity * (1 - base_discount - volume_bonus)",  # Correct
    }


@pytest.fixture
def sample_reconstruction_team_b() -> dict:
    """Return Team B's reconstruction attempt (with errors)."""
    return {
        "m1": "0.15",  # Incorrect
        "m2": "0.1",  # Correct
        "m3": "50",  # Incorrect
        "m4": "0.1",  # Incorrect
        "m5": "price * (1 - base_discount)",  # Incorrect - missing quantity and volume
    }


# =============================================================================
# Scenario Walkthrough Fixtures
# =============================================================================


@pytest.fixture
def sample_scenarios() -> list[MockScenario]:
    """Return sample execution scenarios for walkthrough testing."""
    return [
        MockScenario(
            id="scenario_1",
            description="Premium customer with small order",
            input_values={"price": 100.0, "customer_type": "premium", "quantity": 10},
            expected_call_sequence=["calculate_discount"],
            expected_return=800.0,  # 100 * 10 * (1 - 0.2)
            expected_side_effects=[],
        ),
        MockScenario(
            id="scenario_2",
            description="Standard customer with large order",
            input_values={"price": 50.0, "customer_type": "standard", "quantity": 150},
            expected_call_sequence=["calculate_discount"],
            expected_return=6375.0,  # 50 * 150 * (1 - 0.1 - 0.05)
            expected_side_effects=[],
        ),
        MockScenario(
            id="scenario_3",
            description="Invalid input - negative price",
            input_values={"price": -10.0, "customer_type": "premium", "quantity": 5},
            expected_call_sequence=["calculate_discount"],
            expected_return=None,  # Raises exception
            expected_side_effects=["raises ValueError"],
        ),
        MockScenario(
            id="scenario_4",
            description="Boundary case - exactly 100 items",
            input_values={"price": 25.0, "customer_type": "standard", "quantity": 100},
            expected_call_sequence=["calculate_discount"],
            expected_return=2250.0,  # 25 * 100 * (1 - 0.1) - no volume bonus
            expected_side_effects=[],
        ),
    ]


@pytest.fixture
def sample_order_scenarios() -> list[MockScenario]:
    """Return scenarios for the OrderProcessor class."""
    return [
        MockScenario(
            id="order_scenario_1",
            description="Valid order with successful payment",
            input_values={
                "order": {
                    "items": [{"price": 100.0, "quantity": 2}],
                    "customer_id": "cust_123",
                    "payment_method": "credit_card",
                }
            },
            expected_call_sequence=[
                "process_order",
                "_validate_order",
                "_calculate_total",
                "_process_payment",
                "_save_order",
            ],
            expected_return={"order_id": "ord_xyz", "status": "completed", "total": 200.0},
            expected_side_effects=["logger.info called", "_order_count incremented"],
        ),
        MockScenario(
            id="order_scenario_2",
            description="Order with missing customer_id",
            input_values={
                "order": {"items": [{"price": 50.0, "quantity": 1}], "payment_method": "debit"}
            },
            expected_call_sequence=["process_order", "_validate_order"],
            expected_return=None,
            expected_side_effects=["raises ValidationError"],
        ),
    ]


# =============================================================================
# Mutation Detection Fixtures
# =============================================================================


@pytest.fixture
def sample_mutations() -> list[MockMutation]:
    """Return sample code mutations for detection testing."""
    return [
        MockMutation(
            id="mut_1",
            original_code="if quantity > 100:",
            mutated_code="if quantity >= 100:",
            mutation_type="boundary",
            location="line 12",
            expected_detectable=True,  # Good docs should catch this
        ),
        MockMutation(
            id="mut_2",
            original_code="base_discount = 0.2",
            mutated_code="base_discount = 0.25",
            mutation_type="constant",
            location="line 8",
            expected_detectable=True,  # Specific value should be documented
        ),
        MockMutation(
            id="mut_3",
            original_code="if price < 0:",
            mutated_code="if price <= 0:",
            mutation_type="boundary",
            location="line 2",
            expected_detectable=True,  # Zero case behavior matters
        ),
        MockMutation(
            id="mut_4",
            original_code="volume_bonus = 0.05",
            mutated_code="volume_bonus = 0.10",
            mutation_type="constant",
            location="line 14",
            expected_detectable=True,
        ),
        MockMutation(
            id="mut_5",
            original_code="return price * quantity * (1 - base_discount - volume_bonus)",
            mutated_code="return price * quantity * (1 - base_discount)",
            mutation_type="missing_call",
            location="line 18",
            expected_detectable=True,  # Volume bonus removal should be caught
        ),
    ]


@pytest.fixture
def sample_mutation_responses() -> dict[str, dict]:
    """Return expected mutation detection responses."""
    return {
        "team_a_complete": {
            "mut_1": {
                "detectable": True,
                "reason": "Docs specify 'over 100' not 'at least 100'",
                "confidence": 0.95,
            },
            "mut_2": {
                "detectable": True,
                "reason": "Docs state premium discount is 20%",
                "confidence": 0.99,
            },
            "mut_3": {
                "detectable": True,
                "reason": "Docs say negative price raises error, zero behavior documented",
                "confidence": 0.85,
            },
        },
        "team_b_incomplete": {
            "mut_1": {
                "detectable": False,
                "reason": "Docs only say 'large orders get bonus' without threshold",
                "confidence": 0.30,
            },
            "mut_2": {
                "detectable": False,
                "reason": "Docs say 'premium discount' without specific value",
                "confidence": 0.40,
            },
            "mut_3": {
                "detectable": False,
                "reason": "Error handling not documented",
                "confidence": 0.20,
            },
        },
    }


# =============================================================================
# Impact Analysis Fixtures
# =============================================================================


@pytest.fixture
def sample_impact_challenges() -> list[MockImpactChallenge]:
    """Return sample impact analysis challenges."""
    return [
        MockImpactChallenge(
            component_id="sample_module.calculate_discount",
            change_description="Add required 'currency' parameter to calculate_discount",
            change_type="signature",
            actual_impacted=[
                "sample_module.OrderProcessor.process_order",
                "sample_module.CartService.get_total",
                "tests.test_discount.TestCalculateDiscount",
            ],
        ),
        MockImpactChallenge(
            component_id="sample_module.OrderProcessor._validate_order",
            change_description="Change _validate_order to return bool instead of raising",
            change_type="return_type",
            actual_impacted=["sample_module.OrderProcessor.process_order"],
        ),
        MockImpactChallenge(
            component_id="sample_module.calculate_discount",
            change_description="Delete calculate_discount function",
            change_type="removal",
            actual_impacted=[
                "sample_module.OrderProcessor._calculate_total",
                "sample_module.CartService.get_total",
                "sample_module.PriceService.get_discounted_price",
                "tests.test_discount.TestCalculateDiscount",
                "tests.test_pricing.TestPricing",
            ],
        ),
    ]


# =============================================================================
# Adversarial Review Fixtures
# =============================================================================


@pytest.fixture
def sample_adversarial_findings() -> list[dict]:
    """Return sample findings from adversarial review."""
    return [
        {
            "finding_id": "adv_1",
            "reviewed_team": "B",
            "component_id": "sample_module.calculate_discount",
            "issue_type": "incorrect_value",
            "description": "Documentation states 15% discount for premium, code shows 20%",
            "severity": "high",
            "verified": True,
        },
        {
            "finding_id": "adv_2",
            "reviewed_team": "A",
            "component_id": "sample_module.OrderProcessor.process_order",
            "issue_type": "missing_exception",
            "description": "PaymentError not documented in raises section",
            "severity": "medium",
            "verified": True,
        },
        {
            "finding_id": "adv_3",
            "reviewed_team": "B",
            "component_id": "sample_module.calculate_discount",
            "issue_type": "missing_side_effect",
            "description": "Documentation doesn't mention thread-safety behavior",
            "severity": "low",
            "verified": False,  # Actually, function is stateless
        },
    ]


# =============================================================================
# Test Generation Fixtures
# =============================================================================


@pytest.fixture
def sample_generated_tests() -> str:
    """Return sample pytest code generated from documentation."""
    return '''
import pytest
from sample_module import calculate_discount


class TestCalculateDiscountFromDocs:
    """Tests generated from documentation."""

    def test_premium_customer_base_discount(self):
        """Test that premium customers receive 20% discount."""
        result = calculate_discount(100.0, "premium", 1)
        assert result == 80.0  # 100 * (1 - 0.2)

    def test_standard_customer_base_discount(self):
        """Test that standard customers receive 10% discount."""
        result = calculate_discount(100.0, "standard", 1)
        assert result == 90.0  # 100 * (1 - 0.1)

    def test_volume_bonus_applied_over_100(self):
        """Test volume bonus for orders over 100 items."""
        result = calculate_discount(10.0, "standard", 150)
        expected = 10.0 * 150 * (1 - 0.1 - 0.05)  # 1275.0
        assert result == expected

    def test_no_volume_bonus_at_100(self):
        """Test no volume bonus at exactly 100 items."""
        result = calculate_discount(10.0, "standard", 100)
        expected = 10.0 * 100 * (1 - 0.1)  # 900.0
        assert result == expected

    def test_negative_price_raises_value_error(self):
        """Test that negative price raises ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            calculate_discount(-10.0, "premium", 5)

    def test_zero_quantity_raises_value_error(self):
        """Test that quantity < 1 raises ValueError."""
        with pytest.raises(ValueError, match="at least 1"):
            calculate_discount(100.0, "premium", 0)
'''


@pytest.fixture
def sample_test_validation_result() -> dict:
    """Return sample test validation result."""
    return {
        "total_tests": 6,
        "passed": 5,
        "failed": 1,
        "documentation_errors": [
            {
                "test": "test_no_volume_bonus_at_100",
                "expected": 900.0,
                "actual": 900.0,
                "issue": "Test passed but documentation was ambiguous about boundary",
            }
        ],
        "coverage": {
            "happy_path": 2,
            "error_handling": 2,
            "edge_cases": 2,
            "boundary_conditions": 2,
        },
    }


# =============================================================================
# Verification Score Fixtures
# =============================================================================


@pytest.fixture
def sample_verification_scores_high() -> MockVerificationScores:
    """Return high-quality verification scores (Grade A)."""
    return MockVerificationScores(
        qa_score=0.95,
        reconstruction_score=0.92,
        scenario_score=0.98,
        mutation_score=0.90,
        impact_score=0.94,
        adversarial_findings=2,
        test_pass_rate=0.96,
    )


@pytest.fixture
def sample_verification_scores_medium() -> MockVerificationScores:
    """Return medium-quality verification scores (Grade B)."""
    return MockVerificationScores(
        qa_score=0.85,
        reconstruction_score=0.80,
        scenario_score=0.88,
        mutation_score=0.75,
        impact_score=0.82,
        adversarial_findings=5,
        test_pass_rate=0.87,
    )


@pytest.fixture
def sample_verification_scores_low() -> MockVerificationScores:
    """Return low-quality verification scores (Grade C/F)."""
    return MockVerificationScores(
        qa_score=0.60,
        reconstruction_score=0.55,
        scenario_score=0.65,
        mutation_score=0.50,
        impact_score=0.58,
        adversarial_findings=12,
        test_pass_rate=0.62,
    )


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def verification_config() -> dict:
    """Return a mock verification configuration."""
    return {
        "verification": {
            "enabled_strategies": [
                "qa_interrogation",
                "masked_reconstruction",
                "scenario_walkthrough",
                "mutation_detection",
                "impact_analysis",
                "adversarial_review",
                "test_generation",
            ],
            "thresholds": {
                "min_overall_quality": 0.85,
                "min_qa_score": 0.80,
                "min_reconstruction_score": 0.75,
                "min_scenario_score": 0.85,
                "min_test_pass_rate": 0.90,
            },
            "qa_interrogation": {
                "questions_per_component": 5,
                "categories": ["return_value", "error_handling", "edge_case", "call_flow"],
            },
            "masked_reconstruction": {
                "mask_ratio": 0.3,
                "mask_types": ["constants", "conditions", "returns"],
            },
            "scenario_walkthrough": {
                "scenarios_per_component": 3,
                "types": ["happy_path", "error_path", "edge_case"],
            },
            "mutation_detection": {
                "mutations_per_component": 5,
                "mutation_types": ["boundary", "off_by_one", "null_handling"],
            },
            "adversarial_review": {"max_findings_per_component": 10},
            "test_generation": {"tests_per_component": 10, "run_generated_tests": True},
        }
    }


# =============================================================================
# Call Graph Fixtures for Impact Analysis
# =============================================================================


@pytest.fixture
def sample_dependency_graph() -> dict:
    """Return a sample dependency graph for impact analysis testing."""
    return {
        "edges": [
            {"caller": "OrderProcessor.process_order", "callee": "calculate_discount"},
            {"caller": "OrderProcessor.process_order", "callee": "OrderProcessor._validate_order"},
            {"caller": "OrderProcessor.process_order", "callee": "OrderProcessor._calculate_total"},
            {"caller": "OrderProcessor.process_order", "callee": "OrderProcessor._process_payment"},
            {"caller": "OrderProcessor._calculate_total", "callee": "calculate_discount"},
            {"caller": "CartService.get_total", "callee": "calculate_discount"},
            {"caller": "PriceService.get_discounted_price", "callee": "calculate_discount"},
        ],
        "nodes": [
            "calculate_discount",
            "OrderProcessor.process_order",
            "OrderProcessor._validate_order",
            "OrderProcessor._calculate_total",
            "OrderProcessor._process_payment",
            "CartService.get_total",
            "PriceService.get_discounted_price",
        ],
    }


# =============================================================================
# Pipeline Integration Fixtures
# =============================================================================


@pytest.fixture
def sample_pipeline_result() -> dict:
    """Return a sample complete pipeline execution result."""
    return {
        "component_id": "sample_module.calculate_discount",
        "verification_complete": True,
        "scores": {
            "qa_score": 0.90,
            "reconstruction_score": 0.85,
            "scenario_score": 0.92,
            "mutation_score": 0.80,
            "impact_score": 0.88,
            "adversarial_findings": 3,
            "test_pass_rate": 0.91,
            "overall_quality": 0.876,
            "quality_grade": "B",
        },
        "weakest_areas": ["Boundary Precision", "Implementation Details", "Dependency Tracking"],
        "documentation_gaps": [
            {
                "type": "constants",
                "severity": "medium",
                "recommendation": "Add specific threshold values to documentation",
            },
            {
                "type": "edge_case",
                "severity": "high",
                "recommendation": "Document behavior at boundary values",
            },
        ],
        "beads_tickets": [
            {"id": "bd-abc1", "title": "Document volume discount threshold", "priority": "medium"},
            {"id": "bd-abc2", "title": "Add boundary condition examples", "priority": "high"},
        ],
        "execution_time_ms": 5430,
        "timestamp": "2026-01-07T10:30:00Z",
    }


# =============================================================================
# Async Test Helpers
# =============================================================================


@pytest.fixture
def async_verification_mock() -> AsyncMock:
    """Return a fresh AsyncMock for testing async verification code."""
    return AsyncMock()


@pytest.fixture
def verification_event_loop():
    """Provide an event loop for verification tests."""
    import asyncio

    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
