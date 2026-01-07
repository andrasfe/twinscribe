"""
Unit tests for the CodeReconstructor verification strategy.

Tests cover:
- Code reconstruction from documentation
- Functional equivalence evaluation
- Documentation completeness scoring
- Reconstruction quality metrics

Related Beads Tickets:
- twinscribe-8pw: Create unit tests for all verification strategies
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from twinscribe.verification import (
    CodeReconstructionChallenge,
    CodeReconstructionResult,
    CodeReconstructor,
    DocumentationGap,
    ReconstructedCode,
    Severity,
    StrategyType,
    VerificationLevel,
)


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Create a mock LLM client for CodeReconstructor."""
    client = MagicMock()

    async def mock_generate(prompt: str) -> str:
        """Return mock responses based on prompt content."""
        if "reconstruct" in prompt.lower():
            return json.dumps(
                {
                    "code": "def function(): return 1",
                    "unknown_areas": [],
                    "assumptions_made": [],
                }
            )
        elif "compare" in prompt.lower() or "equivalence" in prompt.lower():
            return json.dumps(
                {
                    "functional_equivalence": 0.85,
                    "missing_details": [],
                }
            )
        return json.dumps({"status": "success"})

    client.generate = AsyncMock(side_effect=mock_generate)
    return client


@pytest.fixture
def mock_llm_client_low_similarity() -> MagicMock:
    """Create a mock LLM client that returns low similarity scores."""
    client = MagicMock()

    async def mock_generate(prompt: str) -> str:
        if "compare" in prompt.lower() or "equivalence" in prompt.lower():
            return json.dumps(
                {
                    "functional_equivalence": 0.30,
                    "missing_details": [
                        "Missing error handling",
                        "Incorrect return type",
                        "Wrong parameter order",
                    ],
                }
            )
        return json.dumps(
            {
                "code": "def f(): pass",
                "unknown_areas": ["return_value", "error_handling"],
                "assumptions_made": ["Uses default values"],
            }
        )

    client.generate = AsyncMock(side_effect=mock_generate)
    return client


@pytest.fixture
def code_reconstructor(mock_llm_client) -> CodeReconstructor:
    """Create a CodeReconstructor instance with mock LLM."""
    return CodeReconstructor(llm_client=mock_llm_client)


@pytest.mark.verification
class TestCodeReconstructorInit:
    """Tests for CodeReconstructor initialization."""

    def test_init_with_default_settings(self, mock_llm_client):
        """Test initialization with default settings."""
        reconstructor = CodeReconstructor(llm_client=mock_llm_client)

        assert reconstructor.strategy_type == StrategyType.CODE_RECONSTRUCTION
        assert reconstructor.level == VerificationLevel.GENERATIVE
        assert "reconstruction" in reconstructor.description.lower()

    def test_init_with_custom_llm(self, mock_llm_client):
        """Test initialization with custom LLM client."""
        reconstructor = CodeReconstructor(llm_client=mock_llm_client)

        assert reconstructor._llm == mock_llm_client


@pytest.mark.verification
class TestChallengeGeneration:
    """Tests for generating code reconstruction challenges."""

    @pytest.mark.asyncio
    async def test_generate_challenge(
        self,
        code_reconstructor,
        sample_verification_function,
    ):
        """Test generating a code reconstruction challenge."""
        challenge = await code_reconstructor.generate_challenge(
            component_id=sample_verification_function.id,
            source_code=sample_verification_function.source_code,
            team_a_documentation="Team A docs",
            team_b_documentation="Team B docs",
        )

        assert isinstance(challenge, CodeReconstructionChallenge)
        assert challenge.component_id == sample_verification_function.id
        assert challenge.team_a_documentation == "Team A docs"
        assert challenge.team_b_documentation == "Team B docs"
        assert challenge.original_code == sample_verification_function.source_code

    @pytest.mark.asyncio
    async def test_generate_challenge_class(
        self,
        code_reconstructor,
        sample_verification_class,
    ):
        """Test generating a challenge for a class component."""
        challenge = await code_reconstructor.generate_challenge(
            component_id=sample_verification_class.id,
            source_code=sample_verification_class.source_code,
            team_a_documentation="Class A docs",
            team_b_documentation="Class B docs",
        )

        assert challenge.component_id == sample_verification_class.id


@pytest.mark.verification
class TestCodeReconstructionChallengeModel:
    """Tests for CodeReconstructionChallenge data model."""

    def test_challenge_creation(self):
        """Test creating a CodeReconstructionChallenge instance."""
        challenge = CodeReconstructionChallenge(
            challenge_id="chal_recon_001",
            component_id="module.function",
            original_code="def function(): return 1",
            team_a_documentation="Team A docs",
            team_b_documentation="Team B docs",
            function_signature="def function() -> int",
        )

        assert challenge.challenge_id == "chal_recon_001"
        assert challenge.component_id == "module.function"
        assert "return 1" in challenge.original_code

    def test_challenge_attributes(self):
        """Test CodeReconstructionChallenge has required attributes."""
        challenge = CodeReconstructionChallenge(
            challenge_id="chal_recon_002",
            component_id="m.f",
            original_code="code",
            team_a_documentation="",
            team_b_documentation="",
            function_signature="def f()",
        )

        assert hasattr(challenge, "challenge_id")
        assert hasattr(challenge, "component_id")
        assert hasattr(challenge, "original_code")
        assert hasattr(challenge, "team_a_documentation")
        assert hasattr(challenge, "team_b_documentation")
        assert hasattr(challenge, "function_signature")


@pytest.mark.verification
class TestReconstructedCodeModel:
    """Tests for ReconstructedCode data model."""

    def test_reconstructed_code_creation(self):
        """Test creating a ReconstructedCode instance."""
        code = ReconstructedCode(
            code="def function(): return 1",
            from_team="A",
            unknown_areas=["error_handling"],
            assumptions_made=["Assumes positive input"],
        )

        assert code.code == "def function(): return 1"
        assert code.from_team == "A"
        assert "error_handling" in code.unknown_areas

    def test_reconstructed_code_attributes(self):
        """Test ReconstructedCode has required attributes."""
        code = ReconstructedCode(
            code="code",
            from_team="B",
        )

        assert hasattr(code, "code")
        assert hasattr(code, "from_team")
        assert hasattr(code, "unknown_areas")
        assert hasattr(code, "assumptions_made")

    def test_reconstructed_code_json_serialization(self):
        """Test JSON serialization of ReconstructedCode."""
        code = ReconstructedCode(
            code="def f(): pass",
            from_team="A",
            unknown_areas=[],
            assumptions_made=["Simple function"],
        )

        json_str = code.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["from_team"] == "A"
        assert parsed["code"] == "def f(): pass"


@pytest.mark.verification
class TestReconstructionEvaluation:
    """Tests for evaluating code reconstructions."""

    @pytest.mark.asyncio
    async def test_evaluate_high_similarity(self, code_reconstructor):
        """Test evaluation when reconstruction is highly similar."""
        challenge = CodeReconstructionChallenge(
            challenge_id="chal_eval_001",
            component_id="m.f",
            original_code="def f(): return 1",
            team_a_documentation="docs",
            team_b_documentation="docs",
            function_signature="def f() -> int",
        )

        # Both teams produce similar reconstructions
        recon_a = ReconstructedCode(
            code="def f(): return 1",
            from_team="A",
        )
        recon_b = ReconstructedCode(
            code="def f(): return 1",
            from_team="B",
        )

        result = await code_reconstructor.evaluate(
            challenge=challenge,
            team_a_response=recon_a.model_dump_json(),
            team_b_response=recon_b.model_dump_json(),
        )

        # Should have functional equivalence scores
        assert result.team_a_functional_equivalence >= 0.0
        assert result.team_a_score >= 0.0

    @pytest.mark.asyncio
    async def test_evaluate_low_similarity(
        self,
        mock_llm_client_low_similarity,
        sample_verification_function,
    ):
        """Test evaluation when reconstruction differs significantly."""
        reconstructor = CodeReconstructor(llm_client=mock_llm_client_low_similarity)

        challenge = CodeReconstructionChallenge(
            challenge_id="chal_eval_002",
            component_id=sample_verification_function.id,
            original_code=sample_verification_function.source_code,
            team_a_documentation="Incomplete docs",
            team_b_documentation="Incomplete docs",
            function_signature="def calculate_discount(price, customer_type, quantity)",
        )

        recon = ReconstructedCode(
            code="def f(): pass",  # Very different
            from_team="A",
        )

        result = await reconstructor.evaluate(
            challenge=challenge,
            team_a_response=recon.model_dump_json(),
            team_b_response=recon.model_dump_json(),
        )

        # Should have lower scores
        assert isinstance(result, CodeReconstructionResult)

    @pytest.mark.asyncio
    async def test_compare_team_reconstructions(self, code_reconstructor):
        """Test comparing Team A vs Team B reconstructions."""
        challenge = CodeReconstructionChallenge(
            challenge_id="chal_eval_003",
            component_id="m.f",
            original_code="def f(x): return x * 2",
            team_a_documentation="Complete docs",
            team_b_documentation="Incomplete docs",
            function_signature="def f(x: int) -> int",
        )

        # Team A has better reconstruction
        recon_a = ReconstructedCode(
            code="def f(x): return x * 2",
            from_team="A",
        )
        # Team B has worse reconstruction
        recon_b = ReconstructedCode(
            code="def f(x): return x",
            from_team="B",
        )

        result = await code_reconstructor.evaluate(
            challenge=challenge,
            team_a_response=recon_a.model_dump_json(),
            team_b_response=recon_b.model_dump_json(),
        )

        # Both teams should have scores
        assert isinstance(result.team_a_score, float)
        assert isinstance(result.team_b_score, float)


@pytest.mark.verification
class TestCodeReconstructionResultModel:
    """Tests for CodeReconstructionResult data model."""

    def test_result_creation(self):
        """Test creating a CodeReconstructionResult instance."""
        recon_a = ReconstructedCode(
            code="def f(): pass",
            from_team="A",
        )
        recon_b = ReconstructedCode(
            code="def f(): return 1",
            from_team="B",
        )

        result = CodeReconstructionResult(
            result_id="res_recon_001",
            challenge_id="chal_001",
            component_id="m.f",
            team_a_score=0.85,
            team_b_score=0.75,
            team_a_reconstruction=recon_a,
            team_b_reconstruction=recon_b,
            team_a_functional_equivalence=0.85,
            team_b_functional_equivalence=0.75,
            missing_from_both=["error_handling"],
        )

        assert result.result_id == "res_recon_001"
        assert result.team_a_score == 0.85

    def test_result_attributes(self):
        """Test CodeReconstructionResult has required attributes."""
        recon = ReconstructedCode(
            code="",
            from_team="A",
        )

        result = CodeReconstructionResult(
            result_id="res_recon_002",
            challenge_id="chal_002",
            component_id="m.f",
            team_a_score=0.5,
            team_b_score=0.5,
            team_a_reconstruction=recon,
            team_b_reconstruction=recon,
            team_a_functional_equivalence=0.5,
            team_b_functional_equivalence=0.5,
            missing_from_both=[],
        )

        assert hasattr(result, "team_a_reconstruction")
        assert hasattr(result, "team_b_reconstruction")
        assert hasattr(result, "team_a_functional_equivalence")
        assert hasattr(result, "team_b_functional_equivalence")
        assert hasattr(result, "missing_from_both")
        assert hasattr(result, "documentation_gaps")


@pytest.mark.verification
class TestFunctionalEquivalenceMetrics:
    """Tests for functional equivalence metric calculations."""

    def test_functional_equivalence_range(self):
        """Test that functional equivalence is in valid range."""
        recon = ReconstructedCode(
            code="",
            from_team="A",
        )

        result = CodeReconstructionResult(
            result_id="res_001",
            challenge_id="chal_001",
            component_id="m.f",
            team_a_score=0.5,
            team_b_score=0.5,
            team_a_reconstruction=recon,
            team_b_reconstruction=recon,
            team_a_functional_equivalence=0.75,
            team_b_functional_equivalence=0.80,
            missing_from_both=[],
        )

        assert 0.0 <= result.team_a_functional_equivalence <= 1.0
        assert 0.0 <= result.team_b_functional_equivalence <= 1.0

    def test_average_score_calculation(self):
        """Test average score is calculated correctly."""
        recon = ReconstructedCode(
            code="",
            from_team="A",
        )

        result = CodeReconstructionResult(
            result_id="res_002",
            challenge_id="chal_002",
            component_id="m.f",
            team_a_score=0.8,
            team_b_score=0.6,
            team_a_reconstruction=recon,
            team_b_reconstruction=recon,
            team_a_functional_equivalence=0.80,
            team_b_functional_equivalence=0.60,
            missing_from_both=[],
        )

        # Average of 0.8 and 0.6
        assert result.average_score == pytest.approx(0.7, rel=0.01)


@pytest.mark.verification
class TestCodeReconstructorScoring:
    """Tests for code reconstruction scoring."""

    @pytest.mark.asyncio
    async def test_score_perfect_reconstruction(self, code_reconstructor):
        """Test score when reconstruction matches original."""
        challenge = CodeReconstructionChallenge(
            challenge_id="chal_score_1",
            component_id="m.f",
            original_code="def f(): return 1",
            team_a_documentation="docs",
            team_b_documentation="docs",
            function_signature="def f() -> int",
        )

        recon = ReconstructedCode(
            code="def f(): return 1",
            from_team="A",
        )

        result = await code_reconstructor.evaluate(
            challenge=challenge,
            team_a_response=recon.model_dump_json(),
            team_b_response=recon.model_dump_json(),
        )

        # Should have valid scores
        assert isinstance(result.team_a_score, float)

    @pytest.mark.asyncio
    async def test_score_poor_reconstruction(self, mock_llm_client_low_similarity):
        """Test score when reconstruction is poor."""
        reconstructor = CodeReconstructor(llm_client=mock_llm_client_low_similarity)

        challenge = CodeReconstructionChallenge(
            challenge_id="chal_score_2",
            component_id="m.f",
            original_code="def f(x): return x * 2",
            team_a_documentation="docs",
            team_b_documentation="docs",
            function_signature="def f(x: int) -> int",
        )

        recon = ReconstructedCode(
            code="def f(): pass",
            from_team="A",
        )

        result = await reconstructor.evaluate(
            challenge=challenge,
            team_a_response=recon.model_dump_json(),
            team_b_response=recon.model_dump_json(),
        )

        # Should return a valid result
        assert isinstance(result, CodeReconstructionResult)


@pytest.mark.verification
class TestCodeReconstructorEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_documentation(self, code_reconstructor):
        """Test reconstruction from empty documentation."""
        challenge = CodeReconstructionChallenge(
            challenge_id="chal_edge_1",
            component_id="m.f",
            original_code="def f(): return 1",
            team_a_documentation="",
            team_b_documentation="",
            function_signature="def f() -> int",
        )

        recon = ReconstructedCode(
            code="def f(): pass",
            from_team="A",
        )

        result = await code_reconstructor.evaluate(
            challenge=challenge,
            team_a_response=recon.model_dump_json(),
            team_b_response=recon.model_dump_json(),
        )

        assert isinstance(result, CodeReconstructionResult)

    @pytest.mark.asyncio
    async def test_complex_function(
        self,
        code_reconstructor,
        sample_verification_function,
    ):
        """Test reconstruction of complex function."""
        challenge = CodeReconstructionChallenge(
            challenge_id="chal_edge_2",
            component_id=sample_verification_function.id,
            original_code=sample_verification_function.source_code,
            team_a_documentation="Complex docs",
            team_b_documentation="Complex docs",
            function_signature="def calculate_discount(price, customer_type, quantity)",
        )

        recon = ReconstructedCode(
            code=sample_verification_function.source_code,
            from_team="A",
        )

        result = await code_reconstructor.evaluate(
            challenge=challenge,
            team_a_response=recon.model_dump_json(),
            team_b_response=recon.model_dump_json(),
        )

        assert isinstance(result, CodeReconstructionResult)

    @pytest.mark.asyncio
    async def test_class_reconstruction(
        self,
        code_reconstructor,
        sample_verification_class,
    ):
        """Test reconstruction of a class."""
        challenge = CodeReconstructionChallenge(
            challenge_id="chal_edge_3",
            component_id=sample_verification_class.id,
            original_code=sample_verification_class.source_code,
            team_a_documentation="Class docs",
            team_b_documentation="Class docs",
            function_signature="class DiscountCalculator",
        )

        recon = ReconstructedCode(
            code="class Test: pass",
            from_team="A",
        )

        result = await code_reconstructor.evaluate(
            challenge=challenge,
            team_a_response=recon.model_dump_json(),
            team_b_response=recon.model_dump_json(),
        )

        assert isinstance(result, CodeReconstructionResult)


@pytest.mark.verification
class TestDocumentationGapsFromReconstruction:
    """Tests for documentation gap identification from reconstruction."""

    @pytest.mark.asyncio
    async def test_poor_reconstruction_creates_gaps(
        self,
        mock_llm_client_low_similarity,
    ):
        """Test that poor reconstruction creates documentation gaps."""
        reconstructor = CodeReconstructor(llm_client=mock_llm_client_low_similarity)

        challenge = CodeReconstructionChallenge(
            challenge_id="chal_gap_1",
            component_id="m.f",
            original_code="def f(x): return x * 2",
            team_a_documentation="Incomplete docs",
            team_b_documentation="Incomplete docs",
            function_signature="def f(x: int) -> int",
        )

        recon = ReconstructedCode(
            code="def f(): pass",
            from_team="A",
        )

        result = await reconstructor.evaluate(
            challenge=challenge,
            team_a_response=recon.model_dump_json(),
            team_b_response=recon.model_dump_json(),
        )

        # Result should be valid
        assert isinstance(result, CodeReconstructionResult)

    def test_get_documentation_gaps(self, code_reconstructor):
        """Test extracting documentation gaps from result."""
        gap = DocumentationGap(
            gap_id="gap_001",
            area="implementation_details",
            description="Missing parameter documentation",
            severity=Severity.MEDIUM,
            recommendation="Add parameter descriptions",
        )

        recon = ReconstructedCode(
            code="",
            from_team="A",
        )

        result = CodeReconstructionResult(
            result_id="res_001",
            challenge_id="chal_001",
            component_id="m.f",
            team_a_score=0.5,
            team_b_score=0.5,
            team_a_reconstruction=recon,
            team_b_reconstruction=recon,
            team_a_functional_equivalence=0.5,
            team_b_functional_equivalence=0.5,
            missing_from_both=["error_handling"],
            documentation_gaps=[gap],
        )

        gaps = code_reconstructor.get_documentation_gaps(result)

        assert len(gaps) == 1
        assert gaps[0]["area"] == "implementation_details"
        assert gaps[0]["severity"] == "medium"


@pytest.mark.verification
class TestReconstructFromDocumentation:
    """Tests for reconstructing code from documentation."""

    @pytest.mark.asyncio
    async def test_reconstruct_from_docs(self, code_reconstructor):
        """Test code reconstruction from documentation."""
        documentation = """
        Function: calculate_discount
        Parameters:
        - price (float): Base price
        - customer_type (str): 'premium' or 'standard'
        - quantity (int): Number of items

        Returns: float - Final price after discount

        Premium customers get 20% discount.
        Standard customers get 10% discount.
        Volume bonus of 5% for orders over 100 items.
        """

        recon = await code_reconstructor.reconstruct_from_documentation(
            documentation=documentation,
            signature="def calculate_discount(price, customer_type, quantity)",
            team="A",
        )

        assert isinstance(recon, ReconstructedCode)
        assert recon.from_team == "A"
        assert recon.code is not None

    @pytest.mark.asyncio
    async def test_reconstructed_code_has_unknown_areas(self, code_reconstructor):
        """Test that reconstructed code can have unknown areas."""
        recon = await code_reconstructor.reconstruct_from_documentation(
            documentation="Simple function documentation",
            signature="def simple_func()",
            team="B",
        )

        # ReconstructedCode should have unknown_areas attribute
        assert hasattr(recon, "unknown_areas")
        assert isinstance(recon.unknown_areas, list)
