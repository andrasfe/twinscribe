"""
Unit tests for the MutationDetector verification strategy.

Tests cover:
- Mutation creation from source code
- Detection evaluation based on documentation
- Different mutation types handling
- Bug detection confidence scoring
- Documentation precision assessment

Related Beads Tickets:
- twinscribe-8pw: Create unit tests for all verification strategies
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from twinscribe.verification import (
    DocumentationGap,
    Mutation,
    MutationAssessment,
    MutationChallenge,
    MutationDetector,
    MutationEvaluation,
    MutationResult,
    MutationType,
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
    """Return sample source code for testing mutation generation."""
    return '''def calculate_discount(price: float, customer_type: str, quantity: int) -> float:
    """Calculate final price with discounts applied."""
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
'''


# =============================================================================
# TestMutationDetectorInit
# =============================================================================


@pytest.mark.verification
class TestMutationDetectorInit:
    """Tests for MutationDetector initialization."""

    def test_init_with_default_parameters(self, mock_llm_client):
        """Test initialization with default settings."""
        detector = MutationDetector(llm_client=mock_llm_client)

        assert detector._mutations_per_component == 5
        assert MutationType.BOUNDARY in detector._mutation_types
        assert MutationType.OFF_BY_ONE in detector._mutation_types
        assert MutationType.NULL_HANDLING in detector._mutation_types

    def test_init_with_custom_mutations_per_component(self, mock_llm_client):
        """Test initialization with custom mutations count."""
        detector = MutationDetector(llm_client=mock_llm_client, mutations_per_component=10)
        assert detector._mutations_per_component == 10

    def test_init_with_custom_mutation_types(self, mock_llm_client):
        """Test initialization with custom mutation types."""
        custom_types = [MutationType.BOUNDARY, MutationType.WRONG_VARIABLE]
        detector = MutationDetector(llm_client=mock_llm_client, mutation_types=custom_types)
        assert detector._mutation_types == custom_types

    def test_mutation_types_available(self, mock_llm_client):
        """Test that all mutation types are defined."""
        expected_types = [
            MutationType.BOUNDARY,
            MutationType.OFF_BY_ONE,
            MutationType.WRONG_VARIABLE,
            MutationType.MISSING_CALL,
            MutationType.WRONG_ORDER,
            MutationType.NULL_HANDLING,
        ]
        for mutation_type in expected_types:
            assert mutation_type is not None

    def test_strategy_properties(self, mock_llm_client):
        """Test strategy type and level properties."""
        detector = MutationDetector(llm_client=mock_llm_client)

        assert detector.strategy_type == StrategyType.MUTATION_DETECTION
        assert detector.level == VerificationLevel.BEHAVIORAL
        assert (
            "precision" in detector.description.lower()
            or "mutation" in detector.description.lower()
        )


# =============================================================================
# TestMutationCreation
# =============================================================================


@pytest.mark.verification
class TestMutationCreation:
    """Tests for creating code mutations."""

    @pytest.mark.asyncio
    async def test_generate_challenge_returns_mutation_challenge(
        self, mock_llm_client, sample_source_code
    ):
        """Test that generate_challenge returns a MutationChallenge."""
        mock_llm_client.generate.return_value = json.dumps(
            [
                {
                    "mutation_id": "mut_001",
                    "mutation_type": "boundary",
                    "original_code": "quantity > 100",
                    "mutated_code": "quantity >= 100",
                    "description": "Changed > to >= in quantity check",
                    "line_number": 13,
                    "detection_hint": "Document the exact threshold value",
                }
            ]
        )

        detector = MutationDetector(llm_client=mock_llm_client)
        challenge = await detector.generate_challenge(
            component_id="test.calculate_discount", source_code=sample_source_code
        )

        assert isinstance(challenge, MutationChallenge)
        assert challenge.component_id == "test.calculate_discount"
        assert len(challenge.mutations) == 1
        assert challenge.challenge_id.startswith("chal_mut")

    @pytest.mark.asyncio
    async def test_generate_challenge_creates_mutations_with_correct_fields(
        self, mock_llm_client, sample_source_code
    ):
        """Test that generated mutations have all required fields."""
        mock_llm_client.generate.return_value = json.dumps(
            [
                {
                    "mutation_id": "mut_001",
                    "mutation_type": "off_by_one",
                    "original_code": "quantity > 100",
                    "mutated_code": "quantity > 99",
                    "description": "Off by one in quantity threshold",
                    "line_number": 13,
                    "detection_hint": "Document exact boundary values",
                }
            ]
        )

        detector = MutationDetector(llm_client=mock_llm_client)
        challenge = await detector.generate_challenge(
            component_id="test.func", source_code=sample_source_code
        )

        mutation = challenge.mutations[0]
        assert isinstance(mutation, Mutation)
        assert mutation.mutation_id == "mut_001"
        assert mutation.mutation_type == MutationType.OFF_BY_ONE
        assert mutation.original_code == "quantity > 100"
        assert mutation.mutated_code == "quantity > 99"
        assert mutation.line_number == 13

    @pytest.mark.asyncio
    async def test_generate_challenge_respects_num_mutations_kwarg(
        self, mock_llm_client, sample_source_code
    ):
        """Test that num_mutations kwarg is passed to the prompt."""
        mock_llm_client.generate.return_value = json.dumps(
            [
                {
                    "mutation_id": f"mut_{i:03d}",
                    "mutation_type": "boundary",
                    "original_code": f"code_{i}",
                    "mutated_code": f"mutated_{i}",
                    "description": f"Mutation {i}",
                }
                for i in range(7)
            ]
        )

        detector = MutationDetector(llm_client=mock_llm_client)
        challenge = await detector.generate_challenge(
            component_id="test.func", source_code=sample_source_code, num_mutations=7
        )

        assert len(challenge.mutations) == 7
        # Verify prompt was called with num_mutations
        call_args = mock_llm_client.generate.call_args
        assert "7" in call_args[0][0]


# =============================================================================
# TestMutationModel
# =============================================================================


@pytest.mark.verification
class TestMutationModel:
    """Tests for Mutation data model."""

    def test_mutation_creation(self):
        """Test creating a Mutation instance."""
        mutation = Mutation(
            mutation_id="mut_001",
            mutation_type=MutationType.BOUNDARY,
            original_code="x > 10",
            mutated_code="x >= 10",
            description="Boundary change",
            line_number=5,
            detection_hint="Document boundary condition",
        )

        assert mutation.mutation_id == "mut_001"
        assert mutation.mutation_type == MutationType.BOUNDARY
        assert mutation.original_code == "x > 10"
        assert mutation.mutated_code == "x >= 10"

    def test_mutation_json_serialization(self):
        """Test JSON serialization of Mutation."""
        mutation = Mutation(
            mutation_id="mut_001",
            mutation_type=MutationType.OFF_BY_ONE,
            original_code="x > 10",
            mutated_code="x > 9",
            description="Off by one",
            line_number=5,
        )

        json_str = mutation.model_dump_json()
        assert "mut_001" in json_str
        assert "off_by_one" in json_str.lower()


# =============================================================================
# TestMutationChallengeModel
# =============================================================================


@pytest.mark.verification
class TestMutationChallengeModel:
    """Tests for MutationChallenge data model."""

    def test_mutation_challenge_creation(self):
        """Test creating a MutationChallenge instance."""
        mutations = [
            Mutation(
                mutation_id="mut_001",
                mutation_type=MutationType.BOUNDARY,
                original_code="x > 10",
                mutated_code="x >= 10",
                description="Boundary",
                line_number=5,
            )
        ]
        challenge = MutationChallenge(
            challenge_id="chal_001",
            component_id="test.component",
            mutations=mutations,
            full_mutated_code="def test(): pass",
        )

        assert challenge.challenge_id == "chal_001"
        assert challenge.component_id == "test.component"
        assert len(challenge.mutations) == 1

    def test_mutation_challenge_json_serialization(self):
        """Test JSON serialization of MutationChallenge."""
        challenge = MutationChallenge(
            challenge_id="chal_001",
            component_id="test.component",
            mutations=[],
            full_mutated_code="",
        )

        json_str = challenge.model_dump_json()
        assert "chal_001" in json_str
        assert "test.component" in json_str


# =============================================================================
# TestDetectionEvaluation
# =============================================================================


@pytest.mark.verification
class TestDetectionEvaluation:
    """Tests for evaluating mutation detection capability."""

    @pytest.fixture
    def sample_challenge(self) -> MutationChallenge:
        """Create a sample challenge for testing."""
        return MutationChallenge(
            challenge_id="chal_test",
            component_id="test.component",
            mutations=[
                Mutation(
                    mutation_id="mut_001",
                    mutation_type=MutationType.BOUNDARY,
                    original_code="quantity > 100",
                    mutated_code="quantity >= 100",
                    description="Boundary change at 100",
                    line_number=13,
                    detection_hint="Document exact threshold",
                ),
                Mutation(
                    mutation_id="mut_002",
                    mutation_type=MutationType.NULL_HANDLING,
                    original_code="if items is not None:",
                    mutated_code="if items:",
                    description="Removed explicit None check",
                    line_number=5,
                    detection_hint="Document None handling",
                ),
            ],
            full_mutated_code="def test(): pass",
        )

    @pytest.mark.asyncio
    async def test_evaluate_detectable_mutation(self, mock_llm_client, sample_challenge):
        """Test evaluation when documentation would detect mutation."""
        # Mock LLM to say docs would help detect
        mock_llm_client.generate.return_value = json.dumps(
            {"docs_actually_help": True, "explanation": "Documentation specifies exact threshold"}
        )

        detector = MutationDetector(llm_client=mock_llm_client)

        team_a_response = json.dumps(
            [
                {
                    "mutation_id": "mut_001",
                    "would_detect": True,
                    "relevant_documentation": "Threshold is exactly 100",
                },
                {
                    "mutation_id": "mut_002",
                    "would_detect": True,
                    "relevant_documentation": "Items must not be None",
                },
            ]
        )
        team_b_response = json.dumps(
            [
                {
                    "mutation_id": "mut_001",
                    "would_detect": True,
                    "relevant_documentation": "Volume bonus at 100 units",
                },
                {
                    "mutation_id": "mut_002",
                    "would_detect": True,
                    "relevant_documentation": "Explicit None check required",
                },
            ]
        )

        result = await detector.evaluate(
            challenge=sample_challenge,
            team_a_response=team_a_response,
            team_b_response=team_b_response,
        )

        assert isinstance(result, MutationResult)
        assert result.team_a_score == 1.0
        assert result.team_b_score == 1.0
        assert len(result.detectable_mutations) == 2

    @pytest.mark.asyncio
    async def test_evaluate_undetectable_mutation(self, mock_llm_client, sample_challenge):
        """Test evaluation when documentation is too vague to detect."""
        # Mock LLM to say docs wouldn't help
        mock_llm_client.generate.return_value = json.dumps(
            {"docs_actually_help": False, "explanation": "Documentation is too vague"}
        )

        detector = MutationDetector(llm_client=mock_llm_client)

        team_a_response = json.dumps(
            [
                {
                    "mutation_id": "mut_001",
                    "would_detect": False,
                    "relevant_documentation": "Some discount applied",
                },
                {
                    "mutation_id": "mut_002",
                    "would_detect": False,
                    "relevant_documentation": "Items are checked",
                },
            ]
        )
        team_b_response = json.dumps(
            [
                {
                    "mutation_id": "mut_001",
                    "would_detect": False,
                    "relevant_documentation": "Discount for large orders",
                },
                {
                    "mutation_id": "mut_002",
                    "would_detect": False,
                    "relevant_documentation": "Items validated",
                },
            ]
        )

        result = await detector.evaluate(
            challenge=sample_challenge,
            team_a_response=team_a_response,
            team_b_response=team_b_response,
        )

        assert result.team_a_score == 0.0
        assert result.team_b_score == 0.0
        assert len(result.undetectable_mutations) == 2
        assert len(result.documentation_gaps) == 2

    @pytest.mark.asyncio
    async def test_evaluate_team_scores_with_consistent_responses(
        self, mock_llm_client, sample_challenge
    ):
        """Test evaluation scores with consistent LLM responses."""
        # Always return that docs would help detect (both teams have good docs)
        mock_llm_client.generate.return_value = json.dumps(
            {"docs_actually_help": True, "explanation": "Documentation is precise"}
        )

        detector = MutationDetector(llm_client=mock_llm_client)

        # Both teams claim they would detect
        team_a_response = json.dumps(
            [
                {
                    "mutation_id": "mut_001",
                    "would_detect": True,
                    "relevant_documentation": "Exact threshold: 100",
                },
                {
                    "mutation_id": "mut_002",
                    "would_detect": True,
                    "relevant_documentation": "Must check for None explicitly",
                },
            ]
        )
        team_b_response = json.dumps(
            [
                {
                    "mutation_id": "mut_001",
                    "would_detect": True,
                    "relevant_documentation": "Threshold is exactly 100",
                },
                {
                    "mutation_id": "mut_002",
                    "would_detect": True,
                    "relevant_documentation": "Explicit None check required",
                },
            ]
        )

        result = await detector.evaluate(
            challenge=sample_challenge,
            team_a_response=team_a_response,
            team_b_response=team_b_response,
        )

        # With docs actually helping, both teams should score well
        assert result.team_a_score == 1.0
        assert result.team_b_score == 1.0
        assert len(result.detectable_mutations) == 2


# =============================================================================
# TestMutationAssessmentModel
# =============================================================================


@pytest.mark.verification
class TestMutationAssessmentModel:
    """Tests for MutationAssessment data model."""

    def test_mutation_assessment_creation(self):
        """Test creating a MutationAssessment instance."""
        assessment = MutationAssessment(
            mutation_id="mut_001", would_detect=True, relevant_documentation="Threshold is 100"
        )

        assert assessment.mutation_id == "mut_001"
        assert assessment.would_detect is True
        assert assessment.relevant_documentation == "Threshold is 100"

    def test_mutation_assessment_json_serialization(self):
        """Test JSON serialization of MutationAssessment."""
        assessment = MutationAssessment(
            mutation_id="mut_001", would_detect=False, relevant_documentation="No specific docs"
        )

        json_str = assessment.model_dump_json()
        assert "mut_001" in json_str


# =============================================================================
# TestMutationEvaluationModel
# =============================================================================


@pytest.mark.verification
class TestMutationEvaluationModel:
    """Tests for MutationEvaluation data model."""

    def test_mutation_evaluation_creation(self):
        """Test creating a MutationEvaluation instance."""
        evaluation = MutationEvaluation(
            mutation_id="mut_001",
            assessment_accurate=True,
            docs_actually_help=True,
            precision_gap=None,
        )

        assert evaluation.mutation_id == "mut_001"
        assert evaluation.assessment_accurate is True
        assert evaluation.docs_actually_help is True

    def test_mutation_evaluation_with_gap(self):
        """Test MutationEvaluation when there's a precision gap."""
        evaluation = MutationEvaluation(
            mutation_id="mut_001",
            assessment_accurate=False,
            docs_actually_help=False,
            precision_gap="Need exact threshold value",
        )

        assert evaluation.docs_actually_help is False
        assert evaluation.precision_gap == "Need exact threshold value"


# =============================================================================
# TestMutationResultModel
# =============================================================================


@pytest.mark.verification
class TestMutationResultModel:
    """Tests for MutationResult data model."""

    def test_mutation_result_creation(self):
        """Test creating a MutationResult instance."""
        result = MutationResult(
            result_id="res_001",
            challenge_id="chal_001",
            component_id="test.component",
            team_a_score=0.8,
            team_b_score=0.6,
            team_a_evaluations=[],
            team_b_evaluations=[],
            detectable_mutations=["mut_001"],
            undetectable_mutations=["mut_002"],
            mutation_type_scores={"boundary": 0.7},
        )

        assert result.result_id == "res_001"
        assert result.team_a_score == 0.8
        assert result.team_b_score == 0.6
        assert result.average_score == 0.7

    def test_mutation_result_json_serialization(self):
        """Test JSON serialization of MutationResult."""
        result = MutationResult(
            result_id="res_001",
            challenge_id="chal_001",
            component_id="test.component",
            team_a_score=0.9,
            team_b_score=0.7,
        )

        json_str = result.model_dump_json()
        assert "res_001" in json_str


# =============================================================================
# TestMutationScoring
# =============================================================================


@pytest.mark.verification
class TestMutationScoring:
    """Tests for mutation detection score calculation."""

    def test_score_all_mutations_detected(self):
        """Test score when all mutations would be detected."""
        result = MutationResult(
            result_id="res_001",
            challenge_id="chal_001",
            component_id="test.component",
            team_a_score=1.0,
            team_b_score=1.0,
            team_a_evaluations=[
                MutationEvaluation(mutation_id="m1", docs_actually_help=True),
                MutationEvaluation(mutation_id="m2", docs_actually_help=True),
            ],
            team_b_evaluations=[
                MutationEvaluation(mutation_id="m1", docs_actually_help=True),
                MutationEvaluation(mutation_id="m2", docs_actually_help=True),
            ],
        )

        assert result.team_a_score == 1.0
        assert result.average_score == 1.0

    def test_score_no_mutations_detected(self):
        """Test score when no mutations would be detected."""
        result = MutationResult(
            result_id="res_001",
            challenge_id="chal_001",
            component_id="test.component",
            team_a_score=0.0,
            team_b_score=0.0,
        )

        assert result.team_a_score == 0.0
        assert result.average_score == 0.0

    def test_score_partial_detection(self):
        """Test score with mixed detection results."""
        result = MutationResult(
            result_id="res_001",
            challenge_id="chal_001",
            component_id="test.component",
            team_a_score=0.5,
            team_b_score=0.5,
        )

        assert result.average_score == 0.5

    def test_mutation_type_scores_tracked(self):
        """Test that mutation type scores are tracked."""
        result = MutationResult(
            result_id="res_001",
            challenge_id="chal_001",
            component_id="test.component",
            team_a_score=0.8,
            team_b_score=0.6,
            mutation_type_scores={"boundary": 0.9, "off_by_one": 0.7, "null_handling": 0.5},
        )

        assert result.mutation_type_scores["boundary"] == 0.9
        assert result.mutation_type_scores["off_by_one"] == 0.7


# =============================================================================
# TestMutationDetectorEdgeCases
# =============================================================================


@pytest.mark.verification
class TestMutationDetectorEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_missing_assessment_for_mutation(self, mock_llm_client):
        """Test handling when team doesn't provide assessment for a mutation."""
        # Mock LLM response (won't be called for missing assessments)
        mock_llm_client.generate.return_value = json.dumps(
            {"docs_actually_help": True, "explanation": "Docs help"}
        )

        detector = MutationDetector(llm_client=mock_llm_client)

        challenge = MutationChallenge(
            challenge_id="chal_test",
            component_id="test.component",
            mutations=[
                Mutation(
                    mutation_id="mut_001",
                    mutation_type=MutationType.BOUNDARY,
                    original_code="x > 10",
                    mutated_code="x >= 10",
                    description="Boundary change",
                    line_number=5,
                    detection_hint="Document threshold",
                )
            ],
            full_mutated_code="",
        )

        # Team A provides assessment, Team B doesn't
        team_a_response = json.dumps(
            [
                {
                    "mutation_id": "mut_001",
                    "would_detect": True,
                    "relevant_documentation": "Threshold is 10",
                }
            ]
        )
        team_b_response = json.dumps([])  # No assessments

        result = await detector.evaluate(
            challenge=challenge, team_a_response=team_a_response, team_b_response=team_b_response
        )

        # Team A should score better
        assert result.team_a_score >= result.team_b_score

    @pytest.mark.asyncio
    async def test_llm_error_during_generation(self, mock_llm_client, sample_source_code):
        """Test handling LLM errors during mutation generation."""
        mock_llm_client.generate.side_effect = Exception("LLM API error")

        detector = MutationDetector(llm_client=mock_llm_client)

        with pytest.raises(Exception, match="LLM API error"):
            await detector.generate_challenge(
                component_id="test.func", source_code=sample_source_code
            )

    @pytest.mark.asyncio
    async def test_handle_malformed_llm_response(self, mock_llm_client, sample_source_code):
        """Test handling of malformed JSON responses from LLM."""
        mock_llm_client.generate.return_value = "not valid json"

        detector = MutationDetector(llm_client=mock_llm_client)

        with pytest.raises(json.JSONDecodeError):
            await detector.generate_challenge(
                component_id="test.func", source_code=sample_source_code
            )

    @pytest.mark.asyncio
    async def test_llm_error_during_evaluation(self, mock_llm_client):
        """Test handling LLM errors during detection evaluation."""
        mock_llm_client.generate.side_effect = Exception("LLM eval error")

        detector = MutationDetector(llm_client=mock_llm_client)

        challenge = MutationChallenge(
            challenge_id="chal_test",
            component_id="test.component",
            mutations=[
                Mutation(
                    mutation_id="mut_001",
                    mutation_type=MutationType.BOUNDARY,
                    original_code="x > 10",
                    mutated_code="x >= 10",
                    description="Boundary",
                    line_number=5,
                )
            ],
            full_mutated_code="",
        )

        team_response = json.dumps(
            [{"mutation_id": "mut_001", "would_detect": True, "relevant_documentation": "Docs"}]
        )

        with pytest.raises(Exception, match="LLM eval error"):
            await detector.evaluate(
                challenge=challenge, team_a_response=team_response, team_b_response=team_response
            )


# =============================================================================
# TestMutationDetectorIntegration
# =============================================================================


@pytest.mark.verification
class TestMutationDetectorIntegration:
    """Integration tests for MutationDetector with other components."""

    @pytest.mark.asyncio
    async def test_full_mutation_workflow(self, mock_llm_client, sample_source_code):
        """Test complete mutation detection workflow."""
        # Step 1: Mock LLM for mutation generation
        mock_llm_client.generate.return_value = json.dumps(
            [
                {
                    "mutation_id": "mut_001",
                    "mutation_type": "boundary",
                    "original_code": "quantity > 100",
                    "mutated_code": "quantity >= 100",
                    "description": "Changed boundary",
                    "detection_hint": "Document threshold",
                }
            ]
        )

        detector = MutationDetector(llm_client=mock_llm_client)

        # Generate challenge
        challenge = await detector.generate_challenge(
            component_id="test.calculate_discount", source_code=sample_source_code
        )
        assert len(challenge.mutations) == 1

        # Step 2: Reset mock for evaluation phase
        mock_llm_client.generate.return_value = json.dumps(
            {"docs_actually_help": True, "explanation": "Documentation is precise"}
        )

        # Step 3: Create team responses
        team_response = json.dumps(
            [
                {
                    "mutation_id": "mut_001",
                    "would_detect": True,
                    "relevant_documentation": "Volume threshold is exactly 100",
                }
            ]
        )

        # Step 4: Evaluate
        result = await detector.evaluate(
            challenge=challenge, team_a_response=team_response, team_b_response=team_response
        )

        # Step 5: Verify result
        assert isinstance(result, MutationResult)
        assert result.team_a_score > 0

    @pytest.mark.asyncio
    async def test_get_documentation_gaps_from_result(self, mock_llm_client):
        """Test extracting documentation gaps from mutation result."""
        detector = MutationDetector(llm_client=mock_llm_client)

        result = MutationResult(
            result_id="res_001",
            challenge_id="chal_001",
            component_id="test.component",
            team_a_score=0.5,
            team_b_score=0.5,
            documentation_gaps=[
                DocumentationGap(
                    gap_id="gap_001",
                    area="boundary_precision_boundary",
                    description="Neither team's docs would catch boundary mutation",
                    severity=Severity.HIGH,
                    recommendation="Document exact threshold value",
                )
            ],
        )

        gaps = detector.get_documentation_gaps(result)

        assert len(gaps) == 1
        assert "boundary" in gaps[0]["area"]
        assert gaps[0]["severity"] == "high"

    @pytest.mark.asyncio
    async def test_mutation_type_coverage(self, mock_llm_client):
        """Test that all mutation types can be generated."""
        mutation_types_to_test = [
            ("boundary", MutationType.BOUNDARY),
            ("off_by_one", MutationType.OFF_BY_ONE),
            ("wrong_variable", MutationType.WRONG_VARIABLE),
            ("missing_call", MutationType.MISSING_CALL),
            ("wrong_order", MutationType.WRONG_ORDER),
            ("null_handling", MutationType.NULL_HANDLING),
        ]

        for type_str, type_enum in mutation_types_to_test:
            mock_llm_client.generate.return_value = json.dumps(
                [
                    {
                        "mutation_id": "mut_001",
                        "mutation_type": type_str,
                        "original_code": "code",
                        "mutated_code": "mutated",
                        "description": f"Test {type_str}",
                    }
                ]
            )

            detector = MutationDetector(llm_client=mock_llm_client)
            challenge = await detector.generate_challenge(
                component_id="test.func", source_code="def test(): pass"
            )

            assert challenge.mutations[0].mutation_type == type_enum

    @pytest.mark.asyncio
    async def test_documentation_quality_comparison(self, mock_llm_client):
        """Test how documentation quality affects detection capability."""
        detector = MutationDetector(llm_client=mock_llm_client)

        challenge = MutationChallenge(
            challenge_id="chal_test",
            component_id="test.component",
            mutations=[
                Mutation(
                    mutation_id="mut_001",
                    mutation_type=MutationType.BOUNDARY,
                    original_code="x > 100",
                    mutated_code="x >= 100",
                    description="Boundary change",
                    line_number=13,
                    detection_hint="Document exact threshold: 100",
                )
            ],
            full_mutated_code="",
        )

        # Mock different quality levels for each team
        call_count = [0]

        async def quality_mock(prompt):
            call_count[0] += 1
            if call_count[0] == 1:  # Team A - good docs
                return json.dumps(
                    {"docs_actually_help": True, "explanation": "Precise documentation"}
                )
            else:  # Team B - vague docs
                return json.dumps(
                    {"docs_actually_help": False, "explanation": "Vague documentation"}
                )

        mock_llm_client.generate.side_effect = quality_mock

        # Good documentation
        team_a_response = json.dumps(
            [
                {
                    "mutation_id": "mut_001",
                    "would_detect": True,
                    "relevant_documentation": "Threshold is exactly 100 (>100 for bonus)",
                }
            ]
        )

        # Vague documentation
        team_b_response = json.dumps(
            [
                {
                    "mutation_id": "mut_001",
                    "would_detect": False,
                    "relevant_documentation": "Large orders get bonuses",
                }
            ]
        )

        result = await detector.evaluate(
            challenge=challenge, team_a_response=team_a_response, team_b_response=team_b_response
        )

        # Team A should be able to detect, Team B shouldn't
        assert result.team_a_score > result.team_b_score
