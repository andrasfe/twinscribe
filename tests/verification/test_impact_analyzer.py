"""
Unit tests for the ImpactAnalyzer verification strategy.

Tests cover:
- Impact challenge creation from components
- Change impact prediction evaluation
- Dependency documentation validation
- True/false positive/negative analysis
- Precision and recall calculation

Related Beads Tickets:
- twinscribe-8pw: Create unit tests for all verification strategies
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from twinscribe.verification import (
    ChangeType,
    ImpactAnalyzer,
    ImpactChallenge,
    ImpactPrediction,
    ImpactResult,
    Severity,
    StrategyType,
    VerificationLevel,
)


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Create a mock LLM client for ImpactAnalyzer."""
    client = MagicMock()
    client.generate = AsyncMock(return_value='{"status": "success"}')
    return client


@pytest.fixture
def impact_analyzer(mock_llm_client) -> ImpactAnalyzer:
    """Create an ImpactAnalyzer instance with mock LLM."""
    return ImpactAnalyzer(llm_client=mock_llm_client)


@pytest.fixture
def impact_analyzer_with_types(mock_llm_client) -> ImpactAnalyzer:
    """Create an ImpactAnalyzer with specific change types."""
    return ImpactAnalyzer(
        llm_client=mock_llm_client,
        change_types=[ChangeType.SIGNATURE, ChangeType.REMOVAL],
    )


@pytest.mark.verification
class TestImpactAnalyzerInit:
    """Tests for ImpactAnalyzer initialization."""

    def test_init_with_default_settings(self, mock_llm_client):
        """Test initialization with default settings."""
        analyzer = ImpactAnalyzer(llm_client=mock_llm_client)

        assert analyzer.strategy_type == StrategyType.IMPACT_ANALYSIS
        assert analyzer.level == VerificationLevel.BEHAVIORAL
        assert "dependency documentation" in analyzer.description.lower()

    def test_init_with_custom_change_types(self, mock_llm_client):
        """Test initialization with custom change types."""
        custom_types = [ChangeType.SIGNATURE, ChangeType.BEHAVIOR]
        analyzer = ImpactAnalyzer(
            llm_client=mock_llm_client,
            change_types=custom_types,
        )

        assert analyzer._change_types == custom_types

    def test_change_types_available(self):
        """Test that all change types are defined."""
        expected_types = [
            ChangeType.SIGNATURE,
            ChangeType.RETURN_TYPE,
            ChangeType.BEHAVIOR,
            ChangeType.REMOVAL,
            ChangeType.RENAME,
        ]
        for change_type in expected_types:
            assert change_type in ChangeType


@pytest.mark.verification
class TestImpactChallengeCreation:
    """Tests for creating impact analysis challenges."""

    @pytest.mark.asyncio
    async def test_create_signature_change_challenge(
        self,
        impact_analyzer,
        sample_verification_function,
    ):
        """Test creating challenge for signature changes."""
        challenge = await impact_analyzer.generate_challenge(
            component_id=sample_verification_function.id,
            source_code=sample_verification_function.source_code,
            change_type=ChangeType.SIGNATURE,
            actual_impacted=["caller1", "caller2"],
        )

        assert isinstance(challenge, ImpactChallenge)
        assert challenge.change_type == ChangeType.SIGNATURE
        assert (
            "signature" in challenge.change_description.lower()
            or "parameter" in challenge.change_description.lower()
        )
        assert challenge.actual_impacted == ["caller1", "caller2"]

    @pytest.mark.asyncio
    async def test_create_return_type_change_challenge(
        self,
        impact_analyzer,
        sample_verification_function,
    ):
        """Test creating challenge for return type changes."""
        challenge = await impact_analyzer.generate_challenge(
            component_id=sample_verification_function.id,
            source_code=sample_verification_function.source_code,
            change_type=ChangeType.RETURN_TYPE,
        )

        assert challenge.change_type == ChangeType.RETURN_TYPE
        assert "return" in challenge.change_description.lower()

    @pytest.mark.asyncio
    async def test_create_behavior_change_challenge(
        self,
        impact_analyzer,
        sample_verification_function,
    ):
        """Test creating challenge for behavior changes."""
        challenge = await impact_analyzer.generate_challenge(
            component_id=sample_verification_function.id,
            source_code=sample_verification_function.source_code,
            change_type=ChangeType.BEHAVIOR,
        )

        assert challenge.change_type == ChangeType.BEHAVIOR

    @pytest.mark.asyncio
    async def test_create_removal_challenge(
        self,
        impact_analyzer,
        sample_verification_function,
    ):
        """Test creating challenge for component removal."""
        challenge = await impact_analyzer.generate_challenge(
            component_id=sample_verification_function.id,
            source_code=sample_verification_function.source_code,
            change_type=ChangeType.REMOVAL,
        )

        assert challenge.change_type == ChangeType.REMOVAL
        assert "delete" in challenge.change_description.lower()

    @pytest.mark.asyncio
    async def test_create_rename_challenge(
        self,
        impact_analyzer,
        sample_verification_function,
    ):
        """Test creating challenge for component rename."""
        challenge = await impact_analyzer.generate_challenge(
            component_id=sample_verification_function.id,
            source_code=sample_verification_function.source_code,
            change_type=ChangeType.RENAME,
        )

        assert challenge.change_type == ChangeType.RENAME
        assert "rename" in challenge.change_description.lower()


@pytest.mark.verification
class TestImpactChallengeModel:
    """Tests for ImpactChallenge data model."""

    def test_impact_challenge_creation(self):
        """Test creating an ImpactChallenge instance."""
        challenge = ImpactChallenge(
            challenge_id="chal_impact_001",
            component_id="module.function",
            change_type=ChangeType.SIGNATURE,
            change_description="Add required parameter",
            actual_impacted=["caller1", "caller2"],
        )

        assert challenge.challenge_id == "chal_impact_001"
        assert challenge.component_id == "module.function"
        assert challenge.change_type == ChangeType.SIGNATURE
        assert challenge.change_description == "Add required parameter"
        assert challenge.actual_impacted == ["caller1", "caller2"]

    def test_impact_challenge_attributes(self):
        """Test ImpactChallenge has required attributes."""
        challenge = ImpactChallenge(
            challenge_id="chal_impact_002",
            component_id="module.function",
            change_type=ChangeType.REMOVAL,
            change_description="Delete function",
            actual_impacted=[],
        )

        # Verify required attributes exist
        assert hasattr(challenge, "challenge_id")
        assert hasattr(challenge, "component_id")
        assert hasattr(challenge, "change_type")
        assert hasattr(challenge, "change_description")
        assert hasattr(challenge, "actual_impacted")

    def test_impact_challenge_json_serialization(self):
        """Test JSON serialization of ImpactChallenge."""
        challenge = ImpactChallenge(
            challenge_id="chal_impact_003",
            component_id="module.function",
            change_type=ChangeType.BEHAVIOR,
            change_description="Change error handling",
            actual_impacted=["caller1"],
        )

        json_str = challenge.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["challenge_id"] == "chal_impact_003"
        assert parsed["change_type"] == "behavior"


@pytest.mark.verification
class TestImpactPredictionEvaluation:
    """Tests for evaluating impact predictions."""

    @pytest.mark.asyncio
    async def test_evaluate_perfect_prediction(self, impact_analyzer):
        """Test evaluation when team predicts all impacts correctly."""
        challenge = ImpactChallenge(
            challenge_id="chal_impact_001",
            component_id="module.function",
            change_type=ChangeType.SIGNATURE,
            change_description="Add parameter",
            actual_impacted=["caller1", "caller2", "caller3"],
        )

        # Both teams predict correctly
        team_a_pred = ImpactPrediction(
            predicted_impacted=["caller1", "caller2", "caller3"],
            reasoning="These are the callers",
        )
        team_b_pred = ImpactPrediction(
            predicted_impacted=["caller1", "caller2", "caller3"],
            reasoning="All callers will break",
        )

        result = await impact_analyzer.evaluate(
            challenge=challenge,
            team_a_response=team_a_pred.model_dump_json(),
            team_b_response=team_b_pred.model_dump_json(),
        )

        assert result.team_a_precision == 1.0
        assert result.team_a_recall == 1.0
        assert result.team_b_precision == 1.0
        assert result.team_b_recall == 1.0
        assert result.team_a_score == 1.0  # F1 score
        assert result.team_b_score == 1.0

    @pytest.mark.asyncio
    async def test_evaluate_missed_impacts(self, impact_analyzer):
        """Test evaluation when team misses some impacts."""
        challenge = ImpactChallenge(
            challenge_id="chal_impact_002",
            component_id="module.function",
            change_type=ChangeType.REMOVAL,
            change_description="Delete function",
            actual_impacted=["caller1", "caller2", "caller3"],
        )

        # Team A misses caller3
        team_a_pred = ImpactPrediction(
            predicted_impacted=["caller1", "caller2"],
            reasoning="Found these callers",
        )
        team_b_pred = ImpactPrediction(
            predicted_impacted=["caller1", "caller2", "caller3"],
            reasoning="All callers",
        )

        result = await impact_analyzer.evaluate(
            challenge=challenge,
            team_a_response=team_a_pred.model_dump_json(),
            team_b_response=team_b_pred.model_dump_json(),
        )

        assert result.team_a_precision == 1.0  # All predicted are correct
        assert result.team_a_recall == pytest.approx(2 / 3)  # Missed one
        assert len(result.team_a_false_negatives) == 1
        assert "caller3" in result.team_a_false_negatives

    @pytest.mark.asyncio
    async def test_evaluate_overclaimed_impacts(self, impact_analyzer):
        """Test evaluation when team claims non-existent impacts."""
        challenge = ImpactChallenge(
            challenge_id="chal_impact_003",
            component_id="module.function",
            change_type=ChangeType.BEHAVIOR,
            change_description="Change behavior",
            actual_impacted=["caller1"],
        )

        # Team A claims extra callers
        team_a_pred = ImpactPrediction(
            predicted_impacted=["caller1", "caller2", "caller3"],
            reasoning="Might affect these",
        )
        team_b_pred = ImpactPrediction(
            predicted_impacted=["caller1"],
            reasoning="Only this caller",
        )

        result = await impact_analyzer.evaluate(
            challenge=challenge,
            team_a_response=team_a_pred.model_dump_json(),
            team_b_response=team_b_pred.model_dump_json(),
        )

        assert result.team_a_precision == pytest.approx(1 / 3)  # 1 correct out of 3
        assert result.team_a_recall == 1.0  # Found all actual
        assert len(result.team_a_false_positives) == 2
        assert result.team_b_precision == 1.0
        assert result.team_b_recall == 1.0

    @pytest.mark.asyncio
    async def test_evaluate_mixed_prediction(self, impact_analyzer):
        """Test evaluation with mix of correct and incorrect predictions."""
        challenge = ImpactChallenge(
            challenge_id="chal_impact_004",
            component_id="module.function",
            change_type=ChangeType.SIGNATURE,
            change_description="Change signature",
            actual_impacted=["caller1", "caller2", "caller3", "caller4"],
        )

        # Team A has mix: misses some, overclaims some
        team_a_pred = ImpactPrediction(
            predicted_impacted=["caller1", "caller2", "wrong1"],
            reasoning="Partial prediction",
        )
        team_b_pred = ImpactPrediction(
            predicted_impacted=["caller1", "caller3", "wrong2"],
            reasoning="Another partial",
        )

        result = await impact_analyzer.evaluate(
            challenge=challenge,
            team_a_response=team_a_pred.model_dump_json(),
            team_b_response=team_b_pred.model_dump_json(),
        )

        # Team A: 2 correct out of 3 predicted (precision = 2/3)
        # Team A: 2 correct out of 4 actual (recall = 2/4 = 0.5)
        assert result.team_a_precision == pytest.approx(2 / 3)
        assert result.team_a_recall == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_compare_team_predictions(self, impact_analyzer):
        """Test comparing Team A vs Team B impact predictions."""
        challenge = ImpactChallenge(
            challenge_id="chal_impact_005",
            component_id="module.function",
            change_type=ChangeType.REMOVAL,
            change_description="Remove function",
            actual_impacted=["c1", "c2", "c3", "c4"],
        )

        # Team B is better
        team_a_pred = ImpactPrediction(predicted_impacted=["c1"], reasoning="")
        team_b_pred = ImpactPrediction(predicted_impacted=["c1", "c2", "c3", "c4"], reasoning="")

        result = await impact_analyzer.evaluate(
            challenge=challenge,
            team_a_response=team_a_pred.model_dump_json(),
            team_b_response=team_b_pred.model_dump_json(),
        )

        assert result.team_b_score > result.team_a_score


@pytest.mark.verification
class TestImpactResultModel:
    """Tests for ImpactResult data model."""

    def test_impact_result_creation(self):
        """Test creating an ImpactResult instance."""
        team_a_pred = ImpactPrediction(predicted_impacted=["c1", "c2"], reasoning="test")
        team_b_pred = ImpactPrediction(predicted_impacted=["c1"], reasoning="test")

        result = ImpactResult(
            result_id="res_impact_001",
            challenge_id="chal_001",
            component_id="module.func",
            team_a_score=0.8,
            team_b_score=0.6,
            team_a_prediction=team_a_pred,
            team_b_prediction=team_b_pred,
            team_a_true_positives=["c1", "c2"],
            team_a_false_positives=[],
            team_a_false_negatives=["c3"],
            team_b_true_positives=["c1"],
            team_b_false_positives=[],
            team_b_false_negatives=["c2", "c3"],
            team_a_precision=1.0,
            team_a_recall=0.67,
            team_b_precision=1.0,
            team_b_recall=0.33,
        )

        assert result.result_id == "res_impact_001"
        assert result.team_a_score == 0.8

    def test_impact_result_attributes(self):
        """Test ImpactResult has required attributes."""
        team_a_pred = ImpactPrediction(predicted_impacted=[], reasoning="")
        team_b_pred = ImpactPrediction(predicted_impacted=[], reasoning="")

        result = ImpactResult(
            result_id="res_impact_002",
            challenge_id="chal_002",
            component_id="module.func",
            team_a_score=0.5,
            team_b_score=0.5,
            team_a_prediction=team_a_pred,
            team_b_prediction=team_b_pred,
            team_a_true_positives=[],
            team_a_false_positives=[],
            team_a_false_negatives=[],
            team_b_true_positives=[],
            team_b_false_positives=[],
            team_b_false_negatives=[],
            team_a_precision=0.0,
            team_a_recall=1.0,
            team_b_precision=0.0,
            team_b_recall=1.0,
        )

        # Verify required attributes
        assert hasattr(result, "team_a_true_positives")
        assert hasattr(result, "team_a_false_positives")
        assert hasattr(result, "team_a_false_negatives")
        assert hasattr(result, "team_a_precision")
        assert hasattr(result, "team_a_recall")
        assert hasattr(result, "documentation_gaps")

    def test_impact_result_precision_calculation(self):
        """Test precision calculation."""
        # precision = true_positives / (true_positives + false_positives)
        team_a_pred = ImpactPrediction(predicted_impacted=["c1", "c2", "wrong"], reasoning="")

        result = ImpactResult(
            result_id="res_001",
            challenge_id="chal_001",
            component_id="m.f",
            team_a_score=0.0,
            team_b_score=0.0,
            team_a_prediction=team_a_pred,
            team_b_prediction=team_a_pred,
            team_a_true_positives=["c1", "c2"],
            team_a_false_positives=["wrong"],
            team_a_false_negatives=[],
            team_b_true_positives=[],
            team_b_false_positives=[],
            team_b_false_negatives=[],
            team_a_precision=2 / 3,  # 2 / (2 + 1)
            team_a_recall=1.0,
            team_b_precision=0.0,
            team_b_recall=0.0,
        )

        assert result.team_a_precision == pytest.approx(2 / 3)

    def test_impact_result_recall_calculation(self):
        """Test recall calculation."""
        # recall = true_positives / (true_positives + false_negatives)
        team_a_pred = ImpactPrediction(predicted_impacted=["c1"], reasoning="")

        result = ImpactResult(
            result_id="res_002",
            challenge_id="chal_002",
            component_id="m.f",
            team_a_score=0.0,
            team_b_score=0.0,
            team_a_prediction=team_a_pred,
            team_b_prediction=team_a_pred,
            team_a_true_positives=["c1"],
            team_a_false_positives=[],
            team_a_false_negatives=["c2", "c3"],
            team_b_true_positives=[],
            team_b_false_positives=[],
            team_b_false_negatives=[],
            team_a_precision=1.0,
            team_a_recall=1 / 3,  # 1 / (1 + 2)
            team_b_precision=0.0,
            team_b_recall=0.0,
        )

        assert result.team_a_recall == pytest.approx(1 / 3)

    def test_impact_result_f1_score(self):
        """Test F1 score calculation (team_a_score is F1 score)."""
        # F1 = 2 * precision * recall / (precision + recall)
        precision = 0.8
        recall = 0.6
        expected_f1 = 2 * precision * recall / (precision + recall)

        team_a_pred = ImpactPrediction(predicted_impacted=[], reasoning="")

        result = ImpactResult(
            result_id="res_003",
            challenge_id="chal_003",
            component_id="m.f",
            team_a_score=expected_f1,
            team_b_score=0.0,
            team_a_prediction=team_a_pred,
            team_b_prediction=team_a_pred,
            team_a_true_positives=[],
            team_a_false_positives=[],
            team_a_false_negatives=[],
            team_b_true_positives=[],
            team_b_false_positives=[],
            team_b_false_negatives=[],
            team_a_precision=precision,
            team_a_recall=recall,
            team_b_precision=0.0,
            team_b_recall=0.0,
        )

        assert result.team_a_score == pytest.approx(expected_f1)


@pytest.mark.verification
class TestImpactScoring:
    """Tests for impact analysis score calculation."""

    @pytest.mark.asyncio
    async def test_score_perfect_impact_analysis(self, impact_analyzer):
        """Test score when all impacts correctly predicted."""
        challenge = ImpactChallenge(
            challenge_id="chal_score_1",
            component_id="m.f",
            change_type=ChangeType.SIGNATURE,
            change_description="Change",
            actual_impacted=["c1", "c2"],
        )

        team_pred = ImpactPrediction(predicted_impacted=["c1", "c2"], reasoning="")

        result = await impact_analyzer.evaluate(
            challenge=challenge,
            team_a_response=team_pred.model_dump_json(),
            team_b_response=team_pred.model_dump_json(),
        )

        # Perfect prediction = F1 of 1.0
        assert result.team_a_score == 1.0
        assert result.team_b_score == 1.0

    @pytest.mark.asyncio
    async def test_score_no_impacts_predicted(self, impact_analyzer):
        """Test score when no impacts predicted."""
        challenge = ImpactChallenge(
            challenge_id="chal_score_2",
            component_id="m.f",
            change_type=ChangeType.REMOVAL,
            change_description="Remove",
            actual_impacted=["c1", "c2"],
        )

        # Empty prediction
        team_pred = ImpactPrediction(predicted_impacted=[], reasoning="")

        result = await impact_analyzer.evaluate(
            challenge=challenge,
            team_a_response=team_pred.model_dump_json(),
            team_b_response=team_pred.model_dump_json(),
        )

        # No predictions = 0 precision (0/0 handled), 0 recall
        assert result.team_a_score == 0.0

    @pytest.mark.asyncio
    async def test_score_from_recall(self, impact_analyzer):
        """Test that score considers recall (not missing impacts)."""
        challenge = ImpactChallenge(
            challenge_id="chal_score_3",
            component_id="m.f",
            change_type=ChangeType.BEHAVIOR,
            change_description="Change",
            actual_impacted=["c1", "c2", "c3", "c4"],
        )

        # High precision, low recall
        team_a_pred = ImpactPrediction(predicted_impacted=["c1"], reasoning="")
        # Balanced precision and recall
        team_b_pred = ImpactPrediction(predicted_impacted=["c1", "c2", "c3", "c4"], reasoning="")

        result = await impact_analyzer.evaluate(
            challenge=challenge,
            team_a_response=team_a_pred.model_dump_json(),
            team_b_response=team_b_pred.model_dump_json(),
        )

        # Team A: precision=1.0, recall=0.25 -> F1 = 0.4
        # Team B: precision=1.0, recall=1.0 -> F1 = 1.0
        assert result.team_b_score > result.team_a_score


@pytest.mark.verification
class TestImpactAnalyzerWithCallGraph:
    """Tests for impact analysis using call graph data."""

    @pytest.mark.asyncio
    async def test_analyze_with_call_graph(
        self,
        impact_analyzer,
        sample_verification_function,
        sample_dependency_graph,
    ):
        """Test impact analysis using static call graph."""
        # Extract callers from dependency graph
        callers = [
            edge["caller"]
            for edge in sample_dependency_graph["edges"]
            if edge["callee"] == "calculate_discount"
        ]

        challenge = await impact_analyzer.generate_challenge(
            component_id="calculate_discount",
            source_code=sample_verification_function.source_code,
            change_type=ChangeType.SIGNATURE,
            actual_impacted=callers,
        )

        assert len(challenge.actual_impacted) == len(callers)


@pytest.mark.verification
class TestImpactAnalyzerEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_no_dependencies(self, impact_analyzer):
        """Test handling of component with no dependencies."""
        challenge = ImpactChallenge(
            challenge_id="chal_edge_1",
            component_id="m.isolated_func",
            change_type=ChangeType.SIGNATURE,
            change_description="Change signature",
            actual_impacted=[],  # No callers
        )

        team_pred = ImpactPrediction(predicted_impacted=[], reasoning="No callers")

        result = await impact_analyzer.evaluate(
            challenge=challenge,
            team_a_response=team_pred.model_dump_json(),
            team_b_response=team_pred.model_dump_json(),
        )

        # Both correctly predicted empty set
        # With no actual impacted, recall is 1.0 (nothing to miss)
        assert result.team_a_recall == 1.0
        assert result.team_b_recall == 1.0

    @pytest.mark.asyncio
    async def test_empty_team_response(self, impact_analyzer):
        """Test handling when team provides no impact predictions."""
        challenge = ImpactChallenge(
            challenge_id="chal_edge_2",
            component_id="m.f",
            change_type=ChangeType.REMOVAL,
            change_description="Remove",
            actual_impacted=["c1", "c2"],
        )

        empty_pred = ImpactPrediction(predicted_impacted=[], reasoning="")

        result = await impact_analyzer.evaluate(
            challenge=challenge,
            team_a_response=empty_pred.model_dump_json(),
            team_b_response=empty_pred.model_dump_json(),
        )

        # Missed all impacts
        assert len(result.team_a_false_negatives) == 2
        assert result.team_a_recall == 0.0


@pytest.mark.verification
class TestDocumentationGaps:
    """Tests for documentation gap identification."""

    @pytest.mark.asyncio
    async def test_both_teams_missed_impacts_creates_gap(self, impact_analyzer):
        """Test that impacts missed by both teams create documentation gaps."""
        challenge = ImpactChallenge(
            challenge_id="chal_gap_1",
            component_id="m.f",
            change_type=ChangeType.SIGNATURE,
            change_description="Change",
            actual_impacted=["c1", "c2", "c3"],
        )

        # Both teams miss c3
        team_a_pred = ImpactPrediction(predicted_impacted=["c1", "c2"], reasoning="")
        team_b_pred = ImpactPrediction(predicted_impacted=["c1"], reasoning="")

        result = await impact_analyzer.evaluate(
            challenge=challenge,
            team_a_response=team_a_pred.model_dump_json(),
            team_b_response=team_b_pred.model_dump_json(),
        )

        # c3 was missed by both
        assert len(result.documentation_gaps) > 0
        gap = result.documentation_gaps[0]
        assert gap.severity == Severity.CRITICAL
        assert gap.affects_team_a is True
        assert gap.affects_team_b is True

    def test_get_documentation_gaps(self, impact_analyzer):
        """Test extracting documentation gaps from result."""
        from twinscribe.verification.models import DocumentationGap

        team_pred = ImpactPrediction(predicted_impacted=[], reasoning="")
        gap = DocumentationGap(
            gap_id="gap_001",
            area="dependency_tracking",
            description="Missed component c1",
            severity=Severity.HIGH,
            recommendation="Add dependency documentation",
            affects_team_a=True,
            affects_team_b=True,
        )

        result = ImpactResult(
            result_id="res_001",
            challenge_id="chal_001",
            component_id="m.f",
            team_a_score=0.5,
            team_b_score=0.5,
            team_a_prediction=team_pred,
            team_b_prediction=team_pred,
            team_a_true_positives=[],
            team_a_false_positives=[],
            team_a_false_negatives=["c1"],
            team_b_true_positives=[],
            team_b_false_positives=[],
            team_b_false_negatives=["c1"],
            team_a_precision=0.0,
            team_a_recall=0.0,
            team_b_precision=0.0,
            team_b_recall=0.0,
            documentation_gaps=[gap],
        )

        gaps = impact_analyzer.get_documentation_gaps(result)

        assert len(gaps) == 1
        assert gaps[0]["area"] == "dependency_tracking"
        assert gaps[0]["severity"] == "high"
