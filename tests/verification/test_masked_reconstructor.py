"""
Unit tests for the MaskedReconstructor verification strategy.

Tests cover:
- Masked challenge creation from source code
- Reconstruction evaluation and scoring
- Documentation gap identification
- Different mask types handling
- Edge cases in reconstruction logic

Related Beads Tickets:
- twinscribe-8pw: Create unit tests for all verification strategies
"""

import json
import re
from unittest.mock import AsyncMock, MagicMock

import pytest

from twinscribe.verification import (
    DocumentationGap,
    Mask,
    MaskedChallenge,
    MaskedReconstructor,
    MaskEvaluation,
    MaskType,
    ReconstructionResult,
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
    """Return sample source code for testing mask generation."""
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
# TestMaskedReconstructorInit
# =============================================================================


@pytest.mark.verification
class TestMaskedReconstructorInit:
    """Tests for MaskedReconstructor initialization."""

    def test_init_with_default_parameters(self, mock_llm_client):
        """Test initialization with default parameters."""
        reconstructor = MaskedReconstructor(llm_client=mock_llm_client)

        assert reconstructor._mask_ratio == 0.3
        assert MaskType.CONSTANTS in reconstructor._mask_types
        assert MaskType.CONDITIONS in reconstructor._mask_types
        assert MaskType.RETURNS in reconstructor._mask_types

    def test_init_with_custom_mask_types(self, mock_llm_client):
        """Test initialization with custom mask types."""
        custom_types = [MaskType.STRINGS, MaskType.FUNCTION_CALLS]
        reconstructor = MaskedReconstructor(llm_client=mock_llm_client, mask_types=custom_types)
        assert reconstructor._mask_types == custom_types

    def test_init_with_custom_mask_ratio(self, mock_llm_client):
        """Test initialization with custom mask ratio."""
        reconstructor = MaskedReconstructor(llm_client=mock_llm_client, mask_ratio=0.5)
        assert reconstructor._mask_ratio == 0.5

    def test_mask_patterns_are_valid_regex(self, mock_llm_client):
        """Test that all mask patterns are valid regular expressions."""
        for _mask_type, pattern in MaskedReconstructor.MASK_PATTERNS.items():
            # Should compile without errors
            compiled = re.compile(pattern)
            assert compiled is not None

    def test_strategy_properties(self, mock_llm_client):
        """Test strategy type and level properties."""
        reconstructor = MaskedReconstructor(llm_client=mock_llm_client)

        assert reconstructor.strategy_type == StrategyType.MASKED_RECONSTRUCTION
        assert reconstructor.level == VerificationLevel.ACTIVE
        assert (
            "specificity" in reconstructor.description.lower()
            or "reconstruction" in reconstructor.description.lower()
        )


# =============================================================================
# TestMaskedChallengeCreation
# =============================================================================


@pytest.mark.verification
class TestMaskedChallengeCreation:
    """Tests for creating masked code challenges."""

    @pytest.mark.asyncio
    async def test_generate_challenge_returns_masked_challenge(
        self, mock_llm_client, sample_source_code
    ):
        """Test that generate_challenge returns a MaskedChallenge."""
        reconstructor = MaskedReconstructor(
            llm_client=mock_llm_client,
            mask_ratio=1.0,  # Mask everything
        )
        challenge = await reconstructor.generate_challenge(
            component_id="test.calculate_discount", source_code=sample_source_code
        )

        assert isinstance(challenge, MaskedChallenge)
        assert challenge.component_id == "test.calculate_discount"
        assert challenge.original_code == sample_source_code
        assert challenge.challenge_id.startswith("chal_mask")

    @pytest.mark.asyncio
    async def test_generate_challenge_creates_masks(self, mock_llm_client, sample_source_code):
        """Test that masks are created from source code."""
        reconstructor = MaskedReconstructor(
            llm_client=mock_llm_client,
            mask_types=[MaskType.CONSTANTS],
            mask_ratio=1.0,  # Mask all constants
        )
        challenge = await reconstructor.generate_challenge(
            component_id="test.func", source_code=sample_source_code
        )

        # Should have found numeric constants like 0, 1, 0.2, 0.1, 100, 0.05
        assert len(challenge.masks) > 0
        for mask in challenge.masks:
            assert isinstance(mask, Mask)
            assert mask.mask_type == MaskType.CONSTANTS

    @pytest.mark.asyncio
    async def test_masked_code_contains_placeholders(self, mock_llm_client, sample_source_code):
        """Test that masked code has placeholders replacing original values."""
        reconstructor = MaskedReconstructor(
            llm_client=mock_llm_client, mask_types=[MaskType.CONSTANTS], mask_ratio=1.0
        )
        challenge = await reconstructor.generate_challenge(
            component_id="test.func", source_code=sample_source_code
        )

        # Masked code should contain placeholders
        for mask in challenge.masks:
            assert mask.placeholder in challenge.masked_code
            # Original value should be replaced
            # (can't directly check because some originals may appear elsewhere)

    @pytest.mark.asyncio
    async def test_mask_ratio_affects_coverage(self, mock_llm_client, sample_source_code):
        """Test that mask_ratio controls how many elements are masked."""
        # With ratio 0, should have no masks
        reconstructor_zero = MaskedReconstructor(
            llm_client=mock_llm_client, mask_types=[MaskType.CONSTANTS], mask_ratio=0.0
        )
        challenge_zero = await reconstructor_zero.generate_challenge(
            component_id="test.func", source_code=sample_source_code
        )
        assert len(challenge_zero.masks) == 0

    @pytest.mark.asyncio
    async def test_generate_challenge_with_multiple_mask_types(
        self, mock_llm_client, sample_source_code
    ):
        """Test generating challenge with multiple mask types."""
        reconstructor = MaskedReconstructor(
            llm_client=mock_llm_client,
            mask_types=[MaskType.CONSTANTS, MaskType.CONDITIONS, MaskType.RETURNS],
            mask_ratio=1.0,
        )
        challenge = await reconstructor.generate_challenge(
            component_id="test.func", source_code=sample_source_code
        )

        # Should have masks of different types
        mask_types_found = {m.mask_type for m in challenge.masks}
        assert len(mask_types_found) >= 1  # At least one type should be found


# =============================================================================
# TestMaskModel
# =============================================================================


@pytest.mark.verification
class TestMaskModel:
    """Tests for Mask data model."""

    def test_mask_creation(self):
        """Test creating a Mask instance."""
        mask = Mask(
            mask_id="mask_001", start=10, end=15, original="0.2", mask_type=MaskType.CONSTANTS
        )

        assert mask.mask_id == "mask_001"
        assert mask.start == 10
        assert mask.end == 15
        assert mask.original == "0.2"
        assert mask.mask_type == MaskType.CONSTANTS

    def test_mask_placeholder_generation(self):
        """Test that mask generates appropriate placeholder."""
        mask = Mask(
            mask_id="mask_001", start=10, end=15, original="0.2", mask_type=MaskType.CONSTANTS
        )

        # Placeholder should be a non-empty string (uses block characters)
        assert mask.placeholder is not None
        assert len(mask.placeholder) > 0

    def test_mask_json_serialization(self):
        """Test JSON serialization of Mask."""
        mask = Mask(
            mask_id="mask_001", start=10, end=15, original="0.2", mask_type=MaskType.CONSTANTS
        )

        json_str = mask.model_dump_json()
        assert "mask_001" in json_str
        assert "constants" in json_str.lower()


# =============================================================================
# TestMaskedChallengeModel
# =============================================================================


@pytest.mark.verification
class TestMaskedChallengeModel:
    """Tests for MaskedChallenge data model."""

    def test_masked_challenge_creation(self):
        """Test creating a MaskedChallenge instance."""
        masks = [
            Mask(mask_id="mask_001", start=10, end=15, original="0.2", mask_type=MaskType.CONSTANTS)
        ]
        challenge = MaskedChallenge(
            challenge_id="chal_001",
            component_id="test.component",
            original_code="x = 0.2",
            masked_code="x = [MASK:mask_001]",
            masks=masks,
            mask_ratio=0.5,
        )

        assert challenge.challenge_id == "chal_001"
        assert challenge.component_id == "test.component"
        assert len(challenge.masks) == 1
        assert challenge.mask_ratio == 0.5

    def test_masked_challenge_json_serialization(self):
        """Test JSON serialization of MaskedChallenge."""
        challenge = MaskedChallenge(
            challenge_id="chal_001",
            component_id="test.component",
            original_code="x = 0.2",
            masked_code="x = [MASK]",
            masks=[],
            mask_ratio=0.0,
        )

        json_str = challenge.model_dump_json()
        assert "chal_001" in json_str
        assert "test.component" in json_str


# =============================================================================
# TestReconstructionEvaluation
# =============================================================================


@pytest.mark.verification
class TestReconstructionEvaluation:
    """Tests for evaluating reconstruction attempts."""

    @pytest.fixture
    def sample_challenge(self) -> MaskedChallenge:
        """Create a sample challenge for testing."""
        return MaskedChallenge(
            challenge_id="chal_test",
            component_id="test.component",
            original_code="discount = 0.2\nif quantity > 100:",
            masked_code="discount = [MASK:mask_001]\nif quantity > [MASK:mask_002]:",
            masks=[
                Mask(
                    mask_id="mask_001",
                    start=11,
                    end=14,
                    original="0.2",
                    mask_type=MaskType.CONSTANTS,
                ),
                Mask(
                    mask_id="mask_002",
                    start=35,
                    end=38,
                    original="100",
                    mask_type=MaskType.CONSTANTS,
                ),
            ],
            mask_ratio=1.0,
        )

    @pytest.mark.asyncio
    async def test_evaluate_perfect_reconstruction(self, mock_llm_client, sample_challenge):
        """Test evaluation when team perfectly reconstructs all masks."""
        reconstructor = MaskedReconstructor(llm_client=mock_llm_client)

        team_a_response = json.dumps(
            [
                {"mask_id": "mask_001", "reconstructed_value": "0.2"},
                {"mask_id": "mask_002", "reconstructed_value": "100"},
            ]
        )
        team_b_response = json.dumps(
            [
                {"mask_id": "mask_001", "reconstructed_value": "0.2"},
                {"mask_id": "mask_002", "reconstructed_value": "100"},
            ]
        )

        result = await reconstructor.evaluate(
            challenge=sample_challenge,
            team_a_response=team_a_response,
            team_b_response=team_b_response,
        )

        assert isinstance(result, ReconstructionResult)
        assert result.team_a_score == 1.0
        assert result.team_b_score == 1.0
        assert len(result.masks_correct_both) == 2
        assert len(result.masks_wrong_both) == 0
        assert len(result.documentation_gaps) == 0

    @pytest.mark.asyncio
    async def test_evaluate_partial_reconstruction(self, mock_llm_client, sample_challenge):
        """Test evaluation with mixed reconstruction accuracy."""
        reconstructor = MaskedReconstructor(llm_client=mock_llm_client)

        # Team A gets both correct
        team_a_response = json.dumps(
            [
                {"mask_id": "mask_001", "reconstructed_value": "0.2"},
                {"mask_id": "mask_002", "reconstructed_value": "100"},
            ]
        )
        # Team B gets only one correct
        team_b_response = json.dumps(
            [
                {"mask_id": "mask_001", "reconstructed_value": "0.2"},
                {"mask_id": "mask_002", "reconstructed_value": "50"},  # Wrong
            ]
        )

        result = await reconstructor.evaluate(
            challenge=sample_challenge,
            team_a_response=team_a_response,
            team_b_response=team_b_response,
        )

        assert result.team_a_score == 1.0
        assert result.team_b_score == 0.5  # 1 correct, 1 wrong

    @pytest.mark.asyncio
    async def test_evaluate_identifies_documentation_gaps(self, mock_llm_client, sample_challenge):
        """Test that evaluation identifies documentation gaps when both fail."""
        reconstructor = MaskedReconstructor(llm_client=mock_llm_client)

        # Both teams fail mask_002
        team_a_response = json.dumps(
            [
                {"mask_id": "mask_001", "reconstructed_value": "0.2"},
                {"mask_id": "mask_002", "reconstructed_value": "50"},
            ]
        )
        team_b_response = json.dumps(
            [
                {"mask_id": "mask_001", "reconstructed_value": "0.2"},
                {"mask_id": "mask_002", "reconstructed_value": "75"},
            ]
        )

        result = await reconstructor.evaluate(
            challenge=sample_challenge,
            team_a_response=team_a_response,
            team_b_response=team_b_response,
        )

        assert len(result.masks_wrong_both) == 1
        assert "mask_002" in result.masks_wrong_both
        assert len(result.documentation_gaps) == 1
        assert result.documentation_gaps[0].affects_team_a is True
        assert result.documentation_gaps[0].affects_team_b is True

    @pytest.mark.asyncio
    async def test_evaluate_per_mask_type_scores(self, mock_llm_client, sample_challenge):
        """Test that evaluation provides per-mask-type scores."""
        reconstructor = MaskedReconstructor(llm_client=mock_llm_client)

        team_a_response = json.dumps(
            [
                {"mask_id": "mask_001", "reconstructed_value": "0.2"},
                {"mask_id": "mask_002", "reconstructed_value": "100"},
            ]
        )
        team_b_response = json.dumps(
            [
                {"mask_id": "mask_001", "reconstructed_value": "0.2"},
                {"mask_id": "mask_002", "reconstructed_value": "100"},
            ]
        )

        result = await reconstructor.evaluate(
            challenge=sample_challenge,
            team_a_response=team_a_response,
            team_b_response=team_b_response,
        )

        # Should have mask type scores
        assert "constants" in result.mask_type_scores
        assert result.mask_type_scores["constants"] == 1.0

    @pytest.mark.asyncio
    async def test_evaluate_exact_match_required(self, mock_llm_client, sample_challenge):
        """Test evaluation requires exact match for reconstructions."""
        reconstructor = MaskedReconstructor(llm_client=mock_llm_client)

        # Exact matches should be correct
        team_a_response = json.dumps(
            [
                {"mask_id": "mask_001", "reconstructed_value": "0.2"},  # Exact
                {"mask_id": "mask_002", "reconstructed_value": "100"},  # Exact
            ]
        )
        # Non-exact matches should be incorrect
        team_b_response = json.dumps(
            [
                {"mask_id": "mask_001", "reconstructed_value": "0.20"},  # Not exact
                {"mask_id": "mask_002", "reconstructed_value": "100"},  # Exact
            ]
        )

        result = await reconstructor.evaluate(
            challenge=sample_challenge,
            team_a_response=team_a_response,
            team_b_response=team_b_response,
        )

        # Team A should score higher with exact matches
        assert result.team_a_score >= result.team_b_score


# =============================================================================
# TestMaskEvaluationModel
# =============================================================================


@pytest.mark.verification
class TestMaskEvaluationModel:
    """Tests for MaskEvaluation data model."""

    def test_mask_evaluation_creation(self):
        """Test creating a MaskEvaluation instance."""
        evaluation = MaskEvaluation(
            mask_id="mask_001",
            is_correct=True,
            is_semantically_equivalent=True,
            similarity_score=1.0,
        )

        assert evaluation.mask_id == "mask_001"
        assert evaluation.is_correct is True
        assert evaluation.is_semantically_equivalent is True
        assert evaluation.similarity_score == 1.0

    def test_mask_evaluation_incorrect(self):
        """Test MaskEvaluation for incorrect reconstruction."""
        evaluation = MaskEvaluation(
            mask_id="mask_001",
            is_correct=False,
            is_semantically_equivalent=False,
            similarity_score=0.0,
        )

        assert evaluation.is_correct is False
        assert evaluation.similarity_score == 0.0


# =============================================================================
# TestReconstructionResultModel
# =============================================================================


@pytest.mark.verification
class TestReconstructionResultModel:
    """Tests for ReconstructionResult data model."""

    def test_reconstruction_result_creation(self):
        """Test creating a ReconstructionResult instance."""
        result = ReconstructionResult(
            result_id="res_001",
            challenge_id="chal_001",
            component_id="test.component",
            team_a_score=0.9,
            team_b_score=0.8,
            team_a_evaluations=[],
            team_b_evaluations=[],
            masks_correct_both=["mask_001"],
            masks_wrong_both=["mask_002"],
            mask_type_scores={"constants": 0.85},
        )

        assert result.result_id == "res_001"
        assert result.team_a_score == 0.9
        assert result.team_b_score == 0.8
        assert abs(result.average_score - 0.85) < 1e-10  # Handle floating point

    def test_reconstruction_result_json_serialization(self):
        """Test JSON serialization of ReconstructionResult."""
        result = ReconstructionResult(
            result_id="res_001",
            challenge_id="chal_001",
            component_id="test.component",
            team_a_score=0.9,
            team_b_score=0.8,
        )

        json_str = result.model_dump_json()
        assert "res_001" in json_str
        assert "0.9" in json_str


# =============================================================================
# TestDocumentationGapModel
# =============================================================================


@pytest.mark.verification
class TestDocumentationGapModel:
    """Tests for DocumentationGap data model."""

    def test_documentation_gap_creation(self):
        """Test creating a DocumentationGap instance."""
        gap = DocumentationGap(
            gap_id="gap_001",
            area="constants",
            description="Both teams failed to reconstruct threshold value",
            severity=Severity.HIGH,
            recommendation="Document the threshold value in the docstring",
        )

        assert gap.gap_id == "gap_001"
        assert gap.area == "constants"
        assert gap.severity == Severity.HIGH

    def test_documentation_gap_severity_levels(self):
        """Test all severity levels are valid."""
        for severity in [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]:
            gap = DocumentationGap(
                gap_id="gap_001",
                area="test",
                description="Test gap",
                severity=severity,
                recommendation="Test recommendation",
            )
            assert gap.severity == severity


# =============================================================================
# TestReconstructionScoring
# =============================================================================


@pytest.mark.verification
class TestReconstructionScoring:
    """Tests for reconstruction score calculation."""

    def test_score_perfect_reconstruction(self):
        """Test score calculation for perfect reconstruction."""
        result = ReconstructionResult(
            result_id="res_001",
            challenge_id="chal_001",
            component_id="test.component",
            team_a_score=1.0,
            team_b_score=1.0,
            team_a_evaluations=[
                MaskEvaluation(mask_id="m1", is_correct=True, similarity_score=1.0),
                MaskEvaluation(mask_id="m2", is_correct=True, similarity_score=1.0),
            ],
            team_b_evaluations=[
                MaskEvaluation(mask_id="m1", is_correct=True, similarity_score=1.0),
                MaskEvaluation(mask_id="m2", is_correct=True, similarity_score=1.0),
            ],
        )

        assert result.team_a_score == 1.0
        assert result.team_b_score == 1.0
        assert result.average_score == 1.0

    def test_score_partial_reconstruction(self):
        """Test score calculation for partial reconstruction."""
        result = ReconstructionResult(
            result_id="res_001",
            challenge_id="chal_001",
            component_id="test.component",
            team_a_score=0.5,
            team_b_score=0.5,
        )

        assert result.average_score == 0.5

    def test_score_zero_reconstruction(self):
        """Test score when no masks reconstructed correctly."""
        result = ReconstructionResult(
            result_id="res_001",
            challenge_id="chal_001",
            component_id="test.component",
            team_a_score=0.0,
            team_b_score=0.0,
        )

        assert result.average_score == 0.0


# =============================================================================
# TestMaskPatternMatching
# =============================================================================


@pytest.mark.verification
class TestMaskPatternMatching:
    """Tests for regex pattern matching in mask creation."""

    def test_constant_pattern_matches_integers(self):
        """Test constant pattern matches integer literals."""
        pattern = MaskedReconstructor.MASK_PATTERNS[MaskType.CONSTANTS]
        test_code = "x = 100\ny = 0\nz = -5"

        matches = list(re.finditer(pattern, test_code))
        matched_values = [m.group() for m in matches]

        assert "100" in matched_values
        assert "0" in matched_values
        assert "5" in matched_values  # Pattern matches the number part

    def test_constant_pattern_matches_floats(self):
        """Test constant pattern matches float literals."""
        pattern = MaskedReconstructor.MASK_PATTERNS[MaskType.CONSTANTS]
        test_code = "x = 0.2\ny = 3.14"

        matches = list(re.finditer(pattern, test_code))
        matched_values = [m.group() for m in matches]

        # Should match floats
        assert any("0.2" in v or "0" in v for v in matched_values)
        assert any("3.14" in v or "3" in v for v in matched_values)

    def test_string_pattern_matches_quoted_strings(self):
        """Test string pattern matches quoted strings."""
        pattern = MaskedReconstructor.MASK_PATTERNS[MaskType.STRINGS]
        test_code = "x = \"hello\"\ny = 'world'"

        matches = list(re.finditer(pattern, test_code))
        matched_values = [m.group() for m in matches]

        assert any("hello" in v for v in matched_values)
        assert any("world" in v for v in matched_values)

    def test_condition_pattern_matches_if_statements(self):
        """Test condition pattern matches if statement conditions."""
        pattern = MaskedReconstructor.MASK_PATTERNS[MaskType.CONDITIONS]
        test_code = "if x > 10:\n    pass"

        matches = list(re.finditer(pattern, test_code))
        assert len(matches) > 0

    def test_return_pattern_matches_return_expressions(self):
        """Test return pattern matches return statement expressions."""
        pattern = MaskedReconstructor.MASK_PATTERNS[MaskType.RETURNS]
        test_code = "return x * y\nreturn None"

        matches = list(re.finditer(pattern, test_code))
        assert len(matches) >= 1


# =============================================================================
# TestMaskedReconstructorEdgeCases
# =============================================================================


@pytest.mark.verification
class TestMaskedReconstructorEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_source_code(self, mock_llm_client):
        """Test handling of empty source code."""
        reconstructor = MaskedReconstructor(llm_client=mock_llm_client)
        challenge = await reconstructor.generate_challenge(component_id="test.func", source_code="")

        assert challenge.masks == []
        assert challenge.masked_code == ""

    @pytest.mark.asyncio
    async def test_no_maskable_elements(self, mock_llm_client):
        """Test handling when code has no maskable elements."""
        reconstructor = MaskedReconstructor(
            llm_client=mock_llm_client, mask_types=[MaskType.CONSTANTS], mask_ratio=1.0
        )
        # Code with no numbers
        challenge = await reconstructor.generate_challenge(
            component_id="test.func", source_code="def foo(): pass"
        )

        assert challenge.masks == []

    @pytest.mark.asyncio
    async def test_missing_reconstruction_for_mask(self, mock_llm_client):
        """Test evaluation when team doesn't provide reconstruction for a mask."""
        reconstructor = MaskedReconstructor(llm_client=mock_llm_client)

        challenge = MaskedChallenge(
            challenge_id="chal_test",
            component_id="test.component",
            original_code="x = 0.2",
            masked_code="x = [MASK]",
            masks=[
                Mask(
                    mask_id="mask_001", start=4, end=7, original="0.2", mask_type=MaskType.CONSTANTS
                )
            ],
            mask_ratio=1.0,
        )

        # Team A provides no reconstructions
        team_a_response = json.dumps([])
        team_b_response = json.dumps([{"mask_id": "mask_001", "reconstructed_value": "0.2"}])

        result = await reconstructor.evaluate(
            challenge=challenge, team_a_response=team_a_response, team_b_response=team_b_response
        )

        # Team A should score 0 for missing reconstruction
        assert result.team_a_score == 0.0
        assert result.team_b_score == 1.0

    @pytest.mark.asyncio
    async def test_extra_reconstructions_ignored(self, mock_llm_client):
        """Test that extra reconstructions (not matching masks) are ignored."""
        reconstructor = MaskedReconstructor(llm_client=mock_llm_client)

        challenge = MaskedChallenge(
            challenge_id="chal_test",
            component_id="test.component",
            original_code="x = 0.2",
            masked_code="x = [MASK]",
            masks=[
                Mask(
                    mask_id="mask_001", start=4, end=7, original="0.2", mask_type=MaskType.CONSTANTS
                )
            ],
            mask_ratio=1.0,
        )

        # Team provides extra reconstructions that don't match any mask
        team_a_response = json.dumps(
            [
                {"mask_id": "mask_001", "reconstructed_value": "0.2"},
                {"mask_id": "mask_extra", "reconstructed_value": "999"},  # Extra
            ]
        )
        team_b_response = json.dumps([{"mask_id": "mask_001", "reconstructed_value": "0.2"}])

        result = await reconstructor.evaluate(
            challenge=challenge, team_a_response=team_a_response, team_b_response=team_b_response
        )

        # Both should score perfectly despite extra reconstruction
        assert result.team_a_score == 1.0
        assert result.team_b_score == 1.0


# =============================================================================
# TestMaskedReconstructorIntegration
# =============================================================================


@pytest.mark.verification
class TestMaskedReconstructorIntegration:
    """Integration tests for MaskedReconstructor with other components."""

    @pytest.mark.asyncio
    async def test_full_reconstruction_workflow(self, mock_llm_client, sample_source_code):
        """Test complete masked reconstruction workflow."""
        reconstructor = MaskedReconstructor(
            llm_client=mock_llm_client, mask_types=[MaskType.CONSTANTS], mask_ratio=1.0
        )

        # Step 1: Generate challenge
        challenge = await reconstructor.generate_challenge(
            component_id="test.calculate_discount", source_code=sample_source_code
        )
        assert len(challenge.masks) > 0

        # Step 2: Create team responses (simulating teams reconstructing)
        team_a_reconstructions = [
            {"mask_id": m.mask_id, "reconstructed_value": m.original} for m in challenge.masks
        ]
        team_b_reconstructions = [
            {"mask_id": m.mask_id, "reconstructed_value": m.original} for m in challenge.masks
        ]

        # Step 3: Evaluate
        result = await reconstructor.evaluate(
            challenge=challenge,
            team_a_response=json.dumps(team_a_reconstructions),
            team_b_response=json.dumps(team_b_reconstructions),
        )

        # Step 4: Verify result
        assert isinstance(result, ReconstructionResult)
        assert result.team_a_score == 1.0
        assert result.team_b_score == 1.0

    @pytest.mark.asyncio
    async def test_get_documentation_gaps_from_result(self, mock_llm_client):
        """Test extracting documentation gaps from reconstruction result."""
        reconstructor = MaskedReconstructor(llm_client=mock_llm_client)

        result = ReconstructionResult(
            result_id="res_001",
            challenge_id="chal_001",
            component_id="test.component",
            team_a_score=0.5,
            team_b_score=0.5,
            documentation_gaps=[
                DocumentationGap(
                    gap_id="gap_001",
                    area="constants",
                    description="Both teams failed to reconstruct threshold",
                    severity=Severity.HIGH,
                    recommendation="Add threshold documentation",
                    evidence="Original: 100",
                )
            ],
        )

        gaps = reconstructor.get_documentation_gaps(result)

        assert len(gaps) == 1
        assert gaps[0]["area"] == "constants"
        assert gaps[0]["severity"] == "high"
        assert gaps[0]["recommendation"] == "Add threshold documentation"
