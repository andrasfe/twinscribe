"""
Unit tests for the ScenarioWalker verification strategy.

Tests cover:
- Scenario creation from components
- Execution trace prediction
- Side effect identification
- Call sequence validation
- Different scenario types handling

Related Beads Tickets:
- twinscribe-8pw: Create unit tests for all verification strategies
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from twinscribe.verification import (
    DocumentationGap,
    ExecutionTrace,
    Scenario,
    ScenarioChallenge,
    ScenarioEvaluation,
    ScenarioResult,
    ScenarioType,
    ScenarioWalker,
    Severity,
    StrategyType,
    VerificationLevel,
)
from twinscribe.verification.strategies import LLMClient

# =============================================================================
# Helper Fixtures
# =============================================================================


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Create a mock LLM client that implements the LLMClient protocol."""
    client = MagicMock(spec=LLMClient)
    client.generate = AsyncMock()
    return client


@pytest.fixture
def sample_source_code() -> str:
    """Return sample source code for testing scenario generation."""
    return '''def process_order(order_id: int, items: list, customer_type: str) -> dict:
    """Process an order and apply appropriate discounts."""
    if not items:
        raise ValueError("Order must contain at least one item")

    total = sum(item['price'] * item['quantity'] for item in items)

    if customer_type == "premium":
        discount = total * 0.2
        log_discount_applied(order_id, discount)
    else:
        discount = 0

    final_total = total - discount

    if final_total > 1000:
        send_large_order_notification(order_id)

    save_order(order_id, final_total)
    return {"order_id": order_id, "total": final_total, "discount": discount}
'''


# =============================================================================
# TestScenarioWalkerInit
# =============================================================================


@pytest.mark.verification
class TestScenarioWalkerInit:
    """Tests for ScenarioWalker initialization."""

    def test_init_with_default_parameters(self, mock_llm_client):
        """Test initialization with default settings."""
        walker = ScenarioWalker(llm_client=mock_llm_client)

        assert walker._scenarios_per_component == 3
        assert ScenarioType.HAPPY_PATH in walker._scenario_types
        assert ScenarioType.ERROR_PATH in walker._scenario_types
        assert ScenarioType.EDGE_CASE in walker._scenario_types

    def test_init_with_custom_scenarios_per_component(self, mock_llm_client):
        """Test initialization with custom scenarios count."""
        walker = ScenarioWalker(llm_client=mock_llm_client, scenarios_per_component=5)
        assert walker._scenarios_per_component == 5

    def test_init_with_custom_scenario_types(self, mock_llm_client):
        """Test initialization with custom scenario types."""
        custom_types = [ScenarioType.HAPPY_PATH, ScenarioType.CONCURRENT]
        walker = ScenarioWalker(llm_client=mock_llm_client, scenario_types=custom_types)
        assert walker._scenario_types == custom_types

    def test_scenario_types_available(self, mock_llm_client):
        """Test that all scenario types are defined."""
        expected_types = [
            ScenarioType.HAPPY_PATH,
            ScenarioType.ERROR_PATH,
            ScenarioType.EDGE_CASE,
            ScenarioType.CONCURRENT,
            ScenarioType.STATE_DEPENDENT,
        ]
        for scenario_type in expected_types:
            assert scenario_type is not None

    def test_strategy_properties(self, mock_llm_client):
        """Test strategy type and level properties."""
        walker = ScenarioWalker(llm_client=mock_llm_client)

        assert walker.strategy_type == StrategyType.SCENARIO_WALKTHROUGH
        assert walker.level == VerificationLevel.ACTIVE
        assert (
            "behavioral" in walker.description.lower() or "execution" in walker.description.lower()
        )


# =============================================================================
# TestScenarioCreation
# =============================================================================


@pytest.mark.verification
class TestScenarioCreation:
    """Tests for creating execution scenarios."""

    @pytest.mark.asyncio
    async def test_generate_challenge_returns_scenario_challenge(
        self, mock_llm_client, sample_source_code
    ):
        """Test that generate_challenge returns a ScenarioChallenge."""
        mock_llm_client.generate.return_value = json.dumps(
            [
                {
                    "scenario_id": "scen_001",
                    "scenario_type": "happy_path",
                    "description": "Premium customer with valid order",
                    "inputs": {
                        "order_id": 123,
                        "items": [{"price": 100, "quantity": 2}],
                        "customer_type": "premium",
                    },
                    "expected_calls": ["log_discount_applied", "save_order"],
                    "expected_output": "Returns dict with order details",
                    "expected_side_effects": ["Order saved to database"],
                }
            ]
        )

        walker = ScenarioWalker(llm_client=mock_llm_client)
        challenge = await walker.generate_challenge(
            component_id="test.process_order", source_code=sample_source_code
        )

        assert isinstance(challenge, ScenarioChallenge)
        assert challenge.component_id == "test.process_order"
        assert len(challenge.scenarios) == 1
        assert challenge.challenge_id.startswith("chal_scen")

    @pytest.mark.asyncio
    async def test_generate_challenge_creates_scenarios_with_correct_fields(
        self, mock_llm_client, sample_source_code
    ):
        """Test that generated scenarios have all required fields."""
        mock_llm_client.generate.return_value = json.dumps(
            [
                {
                    "scenario_id": "scen_001",
                    "scenario_type": "error_path",
                    "description": "Empty items list causes error",
                    "inputs": {"order_id": 123, "items": [], "customer_type": "regular"},
                    "expected_calls": [],
                    "expected_output": "ValueError",
                    "expected_side_effects": [],
                }
            ]
        )

        walker = ScenarioWalker(llm_client=mock_llm_client)
        challenge = await walker.generate_challenge(
            component_id="test.func", source_code=sample_source_code
        )

        scenario = challenge.scenarios[0]
        assert isinstance(scenario, Scenario)
        assert scenario.scenario_id == "scen_001"
        assert scenario.scenario_type == ScenarioType.ERROR_PATH
        assert scenario.description == "Empty items list causes error"
        assert scenario.inputs == {"order_id": 123, "items": [], "customer_type": "regular"}
        assert scenario.expected_calls == []

    @pytest.mark.asyncio
    async def test_generate_challenge_respects_num_scenarios_kwarg(
        self, mock_llm_client, sample_source_code
    ):
        """Test that num_scenarios kwarg is passed to the prompt."""
        mock_llm_client.generate.return_value = json.dumps(
            [
                {
                    "scenario_id": f"scen_{i:03d}",
                    "scenario_type": "happy_path",
                    "description": f"Scenario {i}",
                    "inputs": {},
                    "expected_calls": [],
                    "expected_output": "result",
                }
                for i in range(5)
            ]
        )

        walker = ScenarioWalker(llm_client=mock_llm_client)
        challenge = await walker.generate_challenge(
            component_id="test.func", source_code=sample_source_code, num_scenarios=5
        )

        assert len(challenge.scenarios) == 5
        # Verify prompt was called with num_scenarios
        call_args = mock_llm_client.generate.call_args
        assert "5" in call_args[0][0]


# =============================================================================
# TestScenarioModel
# =============================================================================


@pytest.mark.verification
class TestScenarioModel:
    """Tests for Scenario data model."""

    def test_scenario_creation(self):
        """Test creating a Scenario instance."""
        scenario = Scenario(
            scenario_id="scen_001",
            scenario_type=ScenarioType.HAPPY_PATH,
            description="Standard order processing",
            inputs={"order_id": 123, "items": []},
            expected_calls=["validate", "process", "save"],
            expected_output="Success",
            expected_side_effects=["Database updated"],
        )

        assert scenario.scenario_id == "scen_001"
        assert scenario.scenario_type == ScenarioType.HAPPY_PATH
        assert scenario.description == "Standard order processing"
        assert len(scenario.expected_calls) == 3

    def test_scenario_json_serialization(self):
        """Test JSON serialization of Scenario."""
        scenario = Scenario(
            scenario_id="scen_001",
            scenario_type=ScenarioType.ERROR_PATH,
            description="Error scenario",
            inputs={},
            expected_calls=[],
        )

        json_str = scenario.model_dump_json()
        assert "scen_001" in json_str
        assert "error_path" in json_str.lower()


# =============================================================================
# TestExecutionTracePrediction
# =============================================================================


@pytest.mark.verification
class TestExecutionTracePrediction:
    """Tests for predicting execution traces from documentation."""

    @pytest.fixture
    def sample_challenge(self) -> ScenarioChallenge:
        """Create a sample challenge for testing."""
        return ScenarioChallenge(
            challenge_id="chal_test",
            component_id="test.component",
            scenarios=[
                Scenario(
                    scenario_id="scen_001",
                    scenario_type=ScenarioType.HAPPY_PATH,
                    description="Premium customer order",
                    inputs={"customer_type": "premium", "items": [{"price": 100}]},
                    expected_calls=["log_discount_applied", "save_order"],
                    expected_output="dict with order details",
                    expected_side_effects=["Order saved", "Discount logged"],
                ),
                Scenario(
                    scenario_id="scen_002",
                    scenario_type=ScenarioType.ERROR_PATH,
                    description="Empty order",
                    inputs={"customer_type": "regular", "items": []},
                    expected_calls=[],
                    expected_output="ValueError",
                    expected_side_effects=[],
                ),
            ],
        )

    @pytest.mark.asyncio
    async def test_evaluate_correct_predictions(self, mock_llm_client, sample_challenge):
        """Test evaluation when predictions match actual execution."""
        walker = ScenarioWalker(llm_client=mock_llm_client)

        team_a_response = json.dumps(
            [
                {
                    "scenario_id": "scen_001",
                    "predicted_calls": ["log_discount_applied", "save_order"],
                    "predicted_output": "dict with order details",
                    "predicted_side_effects": ["Order saved", "Discount logged"],
                },
                {
                    "scenario_id": "scen_002",
                    "predicted_calls": [],
                    "predicted_output": "ValueError",
                    "predicted_side_effects": [],
                },
            ]
        )
        team_b_response = json.dumps(
            [
                {
                    "scenario_id": "scen_001",
                    "predicted_calls": ["log_discount_applied", "save_order"],
                    "predicted_output": "dict with order details",
                    "predicted_side_effects": ["Order saved", "Discount logged"],
                },
                {
                    "scenario_id": "scen_002",
                    "predicted_calls": [],
                    "predicted_output": "ValueError",
                    "predicted_side_effects": [],
                },
            ]
        )

        result = await walker.evaluate(
            challenge=sample_challenge,
            team_a_response=team_a_response,
            team_b_response=team_b_response,
        )

        assert isinstance(result, ScenarioResult)
        assert result.team_a_score >= 0.7  # Should have high score
        assert result.team_b_score >= 0.7

    @pytest.mark.asyncio
    async def test_evaluate_incorrect_call_sequence(self, mock_llm_client, sample_challenge):
        """Test evaluation when call sequence is wrong."""
        walker = ScenarioWalker(llm_client=mock_llm_client)

        # Team A predicts correct sequence, Team B has wrong order
        team_a_response = json.dumps(
            [
                {
                    "scenario_id": "scen_001",
                    "predicted_calls": ["log_discount_applied", "save_order"],
                    "predicted_output": "dict with order details",
                    "predicted_side_effects": ["Order saved", "Discount logged"],
                },
                {
                    "scenario_id": "scen_002",
                    "predicted_calls": [],
                    "predicted_output": "ValueError",
                    "predicted_side_effects": [],
                },
            ]
        )
        team_b_response = json.dumps(
            [
                {
                    "scenario_id": "scen_001",
                    "predicted_calls": ["save_order", "log_discount_applied"],  # Wrong order
                    "predicted_output": "dict with order details",
                    "predicted_side_effects": ["Order saved", "Discount logged"],
                },
                {
                    "scenario_id": "scen_002",
                    "predicted_calls": [],
                    "predicted_output": "ValueError",
                    "predicted_side_effects": [],
                },
            ]
        )

        result = await walker.evaluate(
            challenge=sample_challenge,
            team_a_response=team_a_response,
            team_b_response=team_b_response,
        )

        # Team A should score higher due to correct sequence
        assert result.team_a_score >= result.team_b_score

    @pytest.mark.asyncio
    async def test_evaluate_missed_side_effects(self, mock_llm_client, sample_challenge):
        """Test evaluation when side effects are missed."""
        walker = ScenarioWalker(llm_client=mock_llm_client)

        # Both teams miss some side effects
        team_a_response = json.dumps(
            [
                {
                    "scenario_id": "scen_001",
                    "predicted_calls": ["log_discount_applied", "save_order"],
                    "predicted_output": "dict with order details",
                    "predicted_side_effects": ["Order saved"],  # Missing "Discount logged"
                },
                {
                    "scenario_id": "scen_002",
                    "predicted_calls": [],
                    "predicted_output": "ValueError",
                    "predicted_side_effects": [],
                },
            ]
        )
        team_b_response = json.dumps(
            [
                {
                    "scenario_id": "scen_001",
                    "predicted_calls": ["log_discount_applied", "save_order"],
                    "predicted_output": "dict with order details",
                    "predicted_side_effects": ["Order saved"],  # Same miss
                },
                {
                    "scenario_id": "scen_002",
                    "predicted_calls": [],
                    "predicted_output": "ValueError",
                    "predicted_side_effects": [],
                },
            ]
        )

        result = await walker.evaluate(
            challenge=sample_challenge,
            team_a_response=team_a_response,
            team_b_response=team_b_response,
        )

        # Should identify commonly missed side effects
        assert "Discount logged" in result.commonly_missed_side_effects


# =============================================================================
# TestExecutionTraceModel
# =============================================================================


@pytest.mark.verification
class TestExecutionTraceModel:
    """Tests for ExecutionTrace data model."""

    def test_execution_trace_creation(self):
        """Test creating an ExecutionTrace instance."""
        trace = ExecutionTrace(
            scenario_id="scen_001",
            predicted_calls=["func_a", "func_b"],
            predicted_output="Success",
            predicted_side_effects=["State changed"],
        )

        assert trace.scenario_id == "scen_001"
        assert len(trace.predicted_calls) == 2
        assert trace.predicted_output == "Success"

    def test_execution_trace_json_serialization(self):
        """Test JSON serialization of ExecutionTrace."""
        trace = ExecutionTrace(
            scenario_id="scen_001", predicted_calls=["func_a"], predicted_output="Result"
        )

        json_str = trace.model_dump_json()
        assert "scen_001" in json_str
        assert "func_a" in json_str


# =============================================================================
# TestWalkthroughEvaluation
# =============================================================================


@pytest.mark.verification
class TestWalkthroughEvaluation:
    """Tests for evaluating walkthrough predictions."""

    @pytest.fixture
    def single_scenario_challenge(self) -> ScenarioChallenge:
        """Create a challenge with a single scenario."""
        return ScenarioChallenge(
            challenge_id="chal_test",
            component_id="test.component",
            scenarios=[
                Scenario(
                    scenario_id="scen_001",
                    scenario_type=ScenarioType.HAPPY_PATH,
                    description="Test scenario",
                    inputs={},
                    expected_calls=["call_a", "call_b", "call_c"],
                    expected_output="success",
                    expected_side_effects=["effect_1", "effect_2"],
                )
            ],
        )

    @pytest.mark.asyncio
    async def test_evaluate_perfect_walkthrough(self, mock_llm_client, single_scenario_challenge):
        """Test evaluation for perfect walkthrough."""
        walker = ScenarioWalker(llm_client=mock_llm_client)

        team_response = json.dumps(
            [
                {
                    "scenario_id": "scen_001",
                    "predicted_calls": ["call_a", "call_b", "call_c"],
                    "predicted_output": "success",
                    "predicted_side_effects": ["effect_1", "effect_2"],
                }
            ]
        )

        result = await walker.evaluate(
            challenge=single_scenario_challenge,
            team_a_response=team_response,
            team_b_response=team_response,
        )

        assert result.team_a_score >= 0.9
        assert result.team_b_score >= 0.9
        assert len(result.scenarios_correct_both) == 1

    @pytest.mark.asyncio
    async def test_evaluate_partial_walkthrough(self, mock_llm_client, single_scenario_challenge):
        """Test evaluation for partial match."""
        walker = ScenarioWalker(llm_client=mock_llm_client)

        # Team A gets partial calls correct
        team_a_response = json.dumps(
            [
                {
                    "scenario_id": "scen_001",
                    "predicted_calls": ["call_a", "call_b"],  # Missing call_c
                    "predicted_output": "success",
                    "predicted_side_effects": ["effect_1"],  # Missing effect_2
                }
            ]
        )
        team_b_response = json.dumps(
            [
                {
                    "scenario_id": "scen_001",
                    "predicted_calls": ["call_a"],  # Missing two calls
                    "predicted_output": "success",
                    "predicted_side_effects": [],  # Missing both effects
                }
            ]
        )

        result = await walker.evaluate(
            challenge=single_scenario_challenge,
            team_a_response=team_a_response,
            team_b_response=team_b_response,
        )

        # Team A should score higher
        assert result.team_a_score > result.team_b_score

    @pytest.mark.asyncio
    async def test_evaluate_identifies_documentation_gaps(
        self, mock_llm_client, single_scenario_challenge
    ):
        """Test that documentation gaps are identified when both fail."""
        walker = ScenarioWalker(llm_client=mock_llm_client)

        # Both teams miss the same calls and effects
        team_response = json.dumps(
            [
                {
                    "scenario_id": "scen_001",
                    "predicted_calls": ["call_a"],  # Missing call_b, call_c
                    "predicted_output": "wrong output",
                    "predicted_side_effects": [],  # Missing all effects
                }
            ]
        )

        result = await walker.evaluate(
            challenge=single_scenario_challenge,
            team_a_response=team_response,
            team_b_response=team_response,
        )

        # Should have documentation gaps for shared failures
        assert len(result.documentation_gaps) > 0 or len(result.commonly_missed_calls) > 0


# =============================================================================
# TestScenarioEvaluationModel
# =============================================================================


@pytest.mark.verification
class TestScenarioEvaluationModel:
    """Tests for ScenarioEvaluation data model."""

    def test_scenario_evaluation_creation(self):
        """Test creating a ScenarioEvaluation instance."""
        evaluation = ScenarioEvaluation(
            scenario_id="scen_001",
            call_sequence_score=0.8,
            output_correct=True,
            side_effects_score=0.7,
            overall_score=0.75,
            missed_calls=["call_c"],
            missed_side_effects=["effect_2"],
        )

        assert evaluation.scenario_id == "scen_001"
        assert evaluation.call_sequence_score == 0.8
        assert evaluation.output_correct is True
        assert evaluation.overall_score == 0.75


# =============================================================================
# TestScenarioResultModel
# =============================================================================


@pytest.mark.verification
class TestScenarioResultModel:
    """Tests for ScenarioResult data model."""

    def test_scenario_result_creation(self):
        """Test creating a ScenarioResult instance."""
        result = ScenarioResult(
            result_id="res_001",
            challenge_id="chal_001",
            component_id="test.component",
            team_a_score=0.85,
            team_b_score=0.75,
            team_a_evaluations=[],
            team_b_evaluations=[],
            scenarios_correct_both=["scen_001"],
            scenarios_wrong_both=["scen_002"],
            commonly_missed_calls=["func_a"],
            commonly_missed_side_effects=["effect_1"],
        )

        assert result.result_id == "res_001"
        assert result.team_a_score == 0.85
        assert result.average_score == 0.8

    def test_scenario_result_json_serialization(self):
        """Test JSON serialization of ScenarioResult."""
        result = ScenarioResult(
            result_id="res_001",
            challenge_id="chal_001",
            component_id="test.component",
            team_a_score=0.8,
            team_b_score=0.7,
        )

        json_str = result.model_dump_json()
        assert "res_001" in json_str


# =============================================================================
# TestScenarioScoring
# =============================================================================


@pytest.mark.verification
class TestScenarioScoring:
    """Tests for scenario walkthrough score calculation."""

    def test_score_perfect_walkthrough(self):
        """Test score calculation for perfect walkthrough."""
        result = ScenarioResult(
            result_id="res_001",
            challenge_id="chal_001",
            component_id="test.component",
            team_a_score=1.0,
            team_b_score=1.0,
        )

        assert result.team_a_score == 1.0
        assert result.average_score == 1.0

    def test_score_partial_walkthrough(self):
        """Test score calculation for partial match."""
        result = ScenarioResult(
            result_id="res_001",
            challenge_id="chal_001",
            component_id="test.component",
            team_a_score=0.6,
            team_b_score=0.4,
        )

        assert result.average_score == 0.5

    def test_score_aggregation_across_scenarios(self):
        """Test aggregating scores across multiple scenarios."""
        result = ScenarioResult(
            result_id="res_001",
            challenge_id="chal_001",
            component_id="test.component",
            team_a_score=0.7,
            team_b_score=0.5,
            team_a_evaluations=[
                ScenarioEvaluation(
                    scenario_id="s1",
                    call_sequence_score=0.8,
                    output_correct=True,
                    side_effects_score=0.6,
                    overall_score=0.8,
                ),
                ScenarioEvaluation(
                    scenario_id="s2",
                    call_sequence_score=0.6,
                    output_correct=True,
                    side_effects_score=0.5,
                    overall_score=0.6,
                ),
            ],
        )

        # Average of evaluations should be reflected in team score
        assert len(result.team_a_evaluations) == 2


# =============================================================================
# TestScenarioWalkerEdgeCases
# =============================================================================


@pytest.mark.verification
class TestScenarioWalkerEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_call_sequence(self, mock_llm_client):
        """Test handling when no functions are called."""
        walker = ScenarioWalker(llm_client=mock_llm_client)

        challenge = ScenarioChallenge(
            challenge_id="chal_test",
            component_id="test.component",
            scenarios=[
                Scenario(
                    scenario_id="scen_001",
                    scenario_type=ScenarioType.ERROR_PATH,
                    description="Early return scenario",
                    inputs={},
                    expected_calls=[],
                    expected_output="None",
                )
            ],
        )

        team_response = json.dumps(
            [
                {
                    "scenario_id": "scen_001",
                    "predicted_calls": [],
                    "predicted_output": "None",
                    "predicted_side_effects": [],
                }
            ]
        )

        result = await walker.evaluate(
            challenge=challenge, team_a_response=team_response, team_b_response=team_response
        )

        # Should handle empty call sequences
        assert result is not None
        assert result.team_a_score >= 0.7  # Should score well for correct empty sequence

    @pytest.mark.asyncio
    async def test_missing_scenario_prediction(self, mock_llm_client):
        """Test handling when team doesn't provide prediction for a scenario."""
        walker = ScenarioWalker(llm_client=mock_llm_client)

        challenge = ScenarioChallenge(
            challenge_id="chal_test",
            component_id="test.component",
            scenarios=[
                Scenario(
                    scenario_id="scen_001",
                    scenario_type=ScenarioType.HAPPY_PATH,
                    description="First scenario",
                    inputs={},
                    expected_calls=["call_a"],
                    expected_output="result",
                ),
                Scenario(
                    scenario_id="scen_002",
                    scenario_type=ScenarioType.ERROR_PATH,
                    description="Second scenario",
                    inputs={},
                    expected_calls=[],
                    expected_output="error",
                ),
            ],
        )

        # Team A provides prediction for only first scenario
        team_a_response = json.dumps(
            [
                {
                    "scenario_id": "scen_001",
                    "predicted_calls": ["call_a"],
                    "predicted_output": "result",
                    "predicted_side_effects": [],
                }
            ]
        )
        # Team B provides both
        team_b_response = json.dumps(
            [
                {
                    "scenario_id": "scen_001",
                    "predicted_calls": ["call_a"],
                    "predicted_output": "result",
                    "predicted_side_effects": [],
                },
                {
                    "scenario_id": "scen_002",
                    "predicted_calls": [],
                    "predicted_output": "error",
                    "predicted_side_effects": [],
                },
            ]
        )

        result = await walker.evaluate(
            challenge=challenge, team_a_response=team_a_response, team_b_response=team_b_response
        )

        # Team A should score lower due to missing prediction
        assert result.team_b_score >= result.team_a_score

    @pytest.mark.asyncio
    async def test_llm_error_during_generation(self, mock_llm_client, sample_source_code):
        """Test handling LLM errors during scenario generation."""
        mock_llm_client.generate.side_effect = Exception("LLM API error")

        walker = ScenarioWalker(llm_client=mock_llm_client)

        with pytest.raises(Exception, match="LLM API error"):
            await walker.generate_challenge(
                component_id="test.func", source_code=sample_source_code
            )

    @pytest.mark.asyncio
    async def test_handle_malformed_llm_response(self, mock_llm_client, sample_source_code):
        """Test handling of malformed JSON responses from LLM."""
        mock_llm_client.generate.return_value = "not valid json"

        walker = ScenarioWalker(llm_client=mock_llm_client)

        with pytest.raises(json.JSONDecodeError):
            await walker.generate_challenge(
                component_id="test.func", source_code=sample_source_code
            )


# =============================================================================
# TestScenarioWalkerIntegration
# =============================================================================


@pytest.mark.verification
class TestScenarioWalkerIntegration:
    """Integration tests for ScenarioWalker with other components."""

    @pytest.mark.asyncio
    async def test_full_walkthrough_workflow(self, mock_llm_client, sample_source_code):
        """Test complete scenario walkthrough workflow."""
        # Step 1: Mock LLM for scenario generation
        mock_llm_client.generate.return_value = json.dumps(
            [
                {
                    "scenario_id": "scen_001",
                    "scenario_type": "happy_path",
                    "description": "Standard order",
                    "inputs": {"order_id": 123},
                    "expected_calls": ["save_order"],
                    "expected_output": "dict",
                    "expected_side_effects": ["Order saved"],
                }
            ]
        )

        walker = ScenarioWalker(llm_client=mock_llm_client)

        # Generate challenge
        challenge = await walker.generate_challenge(
            component_id="test.process_order", source_code=sample_source_code
        )
        assert len(challenge.scenarios) == 1

        # Step 2: Create team responses
        team_response = json.dumps(
            [
                {
                    "scenario_id": "scen_001",
                    "predicted_calls": ["save_order"],
                    "predicted_output": "dict",
                    "predicted_side_effects": ["Order saved"],
                }
            ]
        )

        # Step 3: Evaluate
        result = await walker.evaluate(
            challenge=challenge, team_a_response=team_response, team_b_response=team_response
        )

        # Step 4: Verify result
        assert isinstance(result, ScenarioResult)
        assert result.team_a_score > 0

    @pytest.mark.asyncio
    async def test_get_documentation_gaps_from_result(self, mock_llm_client):
        """Test extracting documentation gaps from scenario result."""
        walker = ScenarioWalker(llm_client=mock_llm_client)

        result = ScenarioResult(
            result_id="res_001",
            challenge_id="chal_001",
            component_id="test.component",
            team_a_score=0.5,
            team_b_score=0.5,
            documentation_gaps=[
                DocumentationGap(
                    gap_id="gap_001",
                    area="execution_behavior",
                    description="Both teams missed key function calls",
                    severity=Severity.HIGH,
                    recommendation="Document the call sequence",
                )
            ],
        )

        gaps = walker.get_documentation_gaps(result)

        assert len(gaps) == 1
        assert gaps[0]["area"] == "execution_behavior"
        assert gaps[0]["severity"] == "high"

    @pytest.mark.asyncio
    async def test_sequence_similarity_calculation(self, mock_llm_client):
        """Test that sequence similarity is calculated correctly."""
        walker = ScenarioWalker(llm_client=mock_llm_client)

        # Test with exact match
        score_exact = walker._sequence_similarity(["a", "b", "c"], ["a", "b", "c"])
        assert score_exact == 1.0

        # Test with partial match
        score_partial = walker._sequence_similarity(
            ["a", "b", "c"],
            ["a", "c"],  # Missing b
        )
        assert 0 < score_partial < 1.0

        # Test with no match
        score_none = walker._sequence_similarity(["a", "b", "c"], ["x", "y", "z"])
        assert score_none == 0.0

        # Test with empty sequences
        score_empty = walker._sequence_similarity([], [])
        assert score_empty == 1.0
