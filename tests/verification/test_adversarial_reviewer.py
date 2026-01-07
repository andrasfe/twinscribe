"""
Unit tests for the AdversarialReviewer verification strategy.

Tests cover:
- Cross-team review process
- Error detection in opposing team's documentation
- Finding validation against source code
- Blind spot identification
- Review result aggregation

Related Beads Tickets:
- twinscribe-8pw: Create unit tests for all verification strategies
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from twinscribe.verification import (
    AdversarialChallenge,
    AdversarialFinding,
    AdversarialResult,
    AdversarialReviewer,
    DocumentationGap,
    Severity,
    StrategyType,
    VerificationLevel,
)


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Create a mock LLM client for AdversarialReviewer."""
    client = MagicMock()

    async def mock_generate(prompt: str) -> str:
        """Return mock responses based on prompt content."""
        if "validate" in prompt.lower():
            return json.dumps({"is_valid": True, "reason": "Finding confirmed by code"})
        return json.dumps({"status": "success"})

    client.generate = AsyncMock(side_effect=mock_generate)
    return client


@pytest.fixture
def mock_llm_client_invalid_findings() -> MagicMock:
    """Create a mock LLM client that returns invalid findings."""
    client = MagicMock()

    async def mock_generate(prompt: str) -> str:
        if "validate" in prompt.lower():
            return json.dumps({"is_valid": False, "reason": "Finding not supported by code"})
        return json.dumps({"status": "success"})

    client.generate = AsyncMock(side_effect=mock_generate)
    return client


@pytest.fixture
def adversarial_reviewer(mock_llm_client) -> AdversarialReviewer:
    """Create an AdversarialReviewer instance with mock LLM."""
    return AdversarialReviewer(llm_client=mock_llm_client)


@pytest.fixture
def adversarial_reviewer_max_findings(mock_llm_client) -> AdversarialReviewer:
    """Create an AdversarialReviewer with custom max findings."""
    return AdversarialReviewer(llm_client=mock_llm_client, max_findings_per_component=5)


@pytest.mark.verification
class TestAdversarialReviewerInit:
    """Tests for AdversarialReviewer initialization."""

    def test_init_with_default_settings(self, mock_llm_client):
        """Test initialization with default settings."""
        reviewer = AdversarialReviewer(llm_client=mock_llm_client)

        assert reviewer.strategy_type == StrategyType.ADVERSARIAL_REVIEW
        assert reviewer.level == VerificationLevel.GENERATIVE
        assert "adversarial" in reviewer.description.lower()
        assert reviewer._max_findings == 10  # Default

    def test_init_with_custom_max_findings(self, mock_llm_client):
        """Test initialization with custom max findings."""
        reviewer = AdversarialReviewer(
            llm_client=mock_llm_client,
            max_findings_per_component=5,
        )

        assert reviewer._max_findings == 5

    def test_finding_types_available(self):
        """Test that all finding types are available as strings."""
        # Common finding issue types used in AdversarialFinding
        expected_types = [
            "incorrect_value",
            "missing_param",
            "missing_exception",
            "wrong_return_type",
            "vague_description",
            "outdated_example",
        ]
        # These are string values, not an enum - just verify they can be used
        for issue_type in expected_types:
            finding = AdversarialFinding(
                finding_id="test",
                reviewed_team="A",
                issue_type=issue_type,
                location="line 1",
                description="Test finding",
                severity=Severity.MEDIUM,
            )
            assert finding.issue_type == issue_type


@pytest.mark.verification
class TestChallengeGeneration:
    """Tests for generating adversarial review challenges."""

    @pytest.mark.asyncio
    async def test_generate_challenge(
        self,
        adversarial_reviewer,
        sample_verification_function,
    ):
        """Test generating an adversarial review challenge."""
        challenge = await adversarial_reviewer.generate_challenge(
            component_id=sample_verification_function.id,
            source_code=sample_verification_function.source_code,
            team_a_documentation="Team A docs",
            team_b_documentation="Team B docs",
        )

        assert isinstance(challenge, AdversarialChallenge)
        assert challenge.component_id == sample_verification_function.id
        assert challenge.team_a_documentation == "Team A docs"
        assert challenge.team_b_documentation == "Team B docs"
        assert challenge.source_code == sample_verification_function.source_code

    @pytest.mark.asyncio
    async def test_generate_challenge_with_max_findings(
        self,
        adversarial_reviewer_max_findings,
        sample_verification_function,
    ):
        """Test challenge includes max findings setting."""
        challenge = await adversarial_reviewer_max_findings.generate_challenge(
            component_id=sample_verification_function.id,
            source_code=sample_verification_function.source_code,
        )

        assert challenge.max_findings == 5


@pytest.mark.verification
class TestAdversarialChallengeModel:
    """Tests for AdversarialChallenge data model."""

    def test_adversarial_challenge_creation(self):
        """Test creating an AdversarialChallenge instance."""
        challenge = AdversarialChallenge(
            challenge_id="chal_adv_001",
            component_id="module.function",
            team_a_documentation="Team A documentation content",
            team_b_documentation="Team B documentation content",
            source_code="def function(): pass",
            max_findings=10,
        )

        assert challenge.challenge_id == "chal_adv_001"
        assert challenge.component_id == "module.function"
        assert challenge.max_findings == 10

    def test_adversarial_challenge_attributes(self):
        """Test AdversarialChallenge has required attributes."""
        challenge = AdversarialChallenge(
            challenge_id="chal_adv_002",
            component_id="m.f",
            team_a_documentation="",
            team_b_documentation="",
            source_code="",
            max_findings=5,
        )

        assert hasattr(challenge, "challenge_id")
        assert hasattr(challenge, "component_id")
        assert hasattr(challenge, "team_a_documentation")
        assert hasattr(challenge, "team_b_documentation")
        assert hasattr(challenge, "source_code")
        assert hasattr(challenge, "max_findings")


@pytest.mark.verification
class TestReviewFindingModel:
    """Tests for AdversarialFinding data model."""

    def test_review_finding_creation(self):
        """Test creating an AdversarialFinding instance."""
        finding = AdversarialFinding(
            finding_id="find_001",
            reviewed_team="B",
            issue_type="incorrect_value",
            location="line 15",
            description="Documentation states 15% discount, code shows 20%",
            severity=Severity.HIGH,
            code_evidence="base_discount = 0.2",
        )

        assert finding.finding_id == "find_001"
        assert finding.reviewed_team == "B"
        assert finding.issue_type == "incorrect_value"
        assert finding.severity == Severity.HIGH

    def test_review_finding_attributes(self):
        """Test AdversarialFinding has required attributes."""
        finding = AdversarialFinding(
            finding_id="find_002",
            reviewed_team="A",
            issue_type="missing_exception",
            location="method docs",
            description="Missing exception documentation",
            severity=Severity.MEDIUM,
        )

        assert hasattr(finding, "finding_id")
        assert hasattr(finding, "reviewed_team")
        assert hasattr(finding, "issue_type")
        assert hasattr(finding, "location")
        assert hasattr(finding, "description")
        assert hasattr(finding, "severity")

    def test_review_finding_json_serialization(self):
        """Test JSON serialization of AdversarialFinding."""
        finding = AdversarialFinding(
            finding_id="find_003",
            reviewed_team="B",
            issue_type="wrong_return_type",
            location="return statement",
            description="Wrong return type documented",
            severity=Severity.LOW,
        )

        json_str = finding.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["finding_id"] == "find_003"
        assert parsed["severity"] == "low"


@pytest.mark.verification
class TestFindingValidation:
    """Tests for validating review findings against source code."""

    @pytest.mark.asyncio
    async def test_validate_correct_finding(
        self,
        adversarial_reviewer,
        sample_verification_function,
    ):
        """Test validation of correct finding."""
        challenge = AdversarialChallenge(
            challenge_id="chal_val_001",
            component_id=sample_verification_function.id,
            team_a_documentation="Team A docs",
            team_b_documentation="Team B docs",
            source_code=sample_verification_function.source_code,
            max_findings=10,
        )

        # Findings from Team A about Team B's docs
        finding_a = AdversarialFinding(
            finding_id="find_a1",
            reviewed_team="B",
            issue_type="incorrect_value",
            location="line 10",
            description="Wrong discount value",
            severity=Severity.HIGH,
            code_evidence="base_discount = 0.2",
        )

        findings_by_a = [finding_a.model_dump()]
        findings_by_b = []  # Team B found nothing

        result = await adversarial_reviewer.evaluate(
            challenge=challenge,
            team_a_response=json.dumps(findings_by_a),
            team_b_response=json.dumps(findings_by_b),
        )

        # With mock returning is_valid=True, the finding should be validated
        assert "find_a1" in result.validated_findings

    @pytest.mark.asyncio
    async def test_validate_incorrect_finding(
        self,
        mock_llm_client_invalid_findings,
        sample_verification_function,
    ):
        """Test validation of incorrect/false finding."""
        reviewer = AdversarialReviewer(llm_client=mock_llm_client_invalid_findings)

        challenge = AdversarialChallenge(
            challenge_id="chal_val_002",
            component_id=sample_verification_function.id,
            team_a_documentation="Team A docs",
            team_b_documentation="Team B docs",
            source_code=sample_verification_function.source_code,
            max_findings=10,
        )

        finding = AdversarialFinding(
            finding_id="find_false",
            reviewed_team="A",
            issue_type="missing_side_effect",
            location="method",
            description="False claim about side effects",
            severity=Severity.LOW,
        )

        result = await reviewer.evaluate(
            challenge=challenge,
            team_a_response=json.dumps([]),
            team_b_response=json.dumps([finding.model_dump()]),
        )

        # With mock returning is_valid=False, finding should be in false_findings
        assert "find_false" in result.false_findings


@pytest.mark.verification
class TestAdversarialResultModel:
    """Tests for AdversarialResult data model."""

    def test_review_result_creation(self):
        """Test creating an AdversarialResult instance."""
        finding_a = AdversarialFinding(
            finding_id="f1",
            reviewed_team="B",
            issue_type="error",
            location="line 1",
            description="Error found",
            severity=Severity.MEDIUM,
        )
        finding_b = AdversarialFinding(
            finding_id="f2",
            reviewed_team="A",
            issue_type="error",
            location="line 2",
            description="Another error",
            severity=Severity.HIGH,
        )

        result = AdversarialResult(
            result_id="res_adv_001",
            challenge_id="chal_001",
            component_id="m.f",
            team_a_score=0.8,
            team_b_score=0.6,
            findings_by_a=[finding_a],
            findings_by_b=[finding_b],
            validated_findings=["f1", "f2"],
            false_findings=[],
        )

        assert result.result_id == "res_adv_001"
        assert len(result.findings_by_a) == 1
        assert len(result.validated_findings) == 2

    def test_review_result_attributes(self):
        """Test AdversarialResult has required attributes."""
        result = AdversarialResult(
            result_id="res_adv_002",
            challenge_id="chal_002",
            component_id="m.f",
            team_a_score=0.5,
            team_b_score=0.5,
            findings_by_a=[],
            findings_by_b=[],
            validated_findings=[],
            false_findings=[],
        )

        assert hasattr(result, "findings_by_a")
        assert hasattr(result, "findings_by_b")
        assert hasattr(result, "validated_findings")
        assert hasattr(result, "false_findings")
        assert hasattr(result, "documentation_gaps")


@pytest.mark.verification
class TestAdversarialScoring:
    """Tests for adversarial review scoring."""

    @pytest.mark.asyncio
    async def test_count_verified_findings(self, adversarial_reviewer):
        """Test counting verified findings."""
        challenge = AdversarialChallenge(
            challenge_id="chal_score_1",
            component_id="m.f",
            team_a_documentation="docs",
            team_b_documentation="docs",
            source_code="code",
            max_findings=10,
        )

        # Team A finds 2 valid findings
        findings_a = [
            AdversarialFinding(
                finding_id="f1",
                reviewed_team="B",
                issue_type="error1",
                location="l1",
                description="d1",
                severity=Severity.HIGH,
            ).model_dump(),
            AdversarialFinding(
                finding_id="f2",
                reviewed_team="B",
                issue_type="error2",
                location="l2",
                description="d2",
                severity=Severity.MEDIUM,
            ).model_dump(),
        ]

        # Team B finds 1 valid finding
        findings_b = [
            AdversarialFinding(
                finding_id="f3",
                reviewed_team="A",
                issue_type="error3",
                location="l3",
                description="d3",
                severity=Severity.LOW,
            ).model_dump(),
        ]

        result = await adversarial_reviewer.evaluate(
            challenge=challenge,
            team_a_response=json.dumps(findings_a),
            team_b_response=json.dumps(findings_b),
        )

        # All findings validated (mock returns is_valid=True)
        assert len(result.validated_findings) == 3

    @pytest.mark.asyncio
    async def test_score_calculation_precision(self, adversarial_reviewer):
        """Test score is based on precision of findings."""
        challenge = AdversarialChallenge(
            challenge_id="chal_score_2",
            component_id="m.f",
            team_a_documentation="docs",
            team_b_documentation="docs",
            source_code="code",
            max_findings=10,
        )

        # Team A: 2 findings, all valid
        findings_a = [
            AdversarialFinding(
                finding_id="f1",
                reviewed_team="B",
                issue_type="e",
                location="l",
                description="d",
                severity=Severity.MEDIUM,
            ).model_dump(),
            AdversarialFinding(
                finding_id="f2",
                reviewed_team="B",
                issue_type="e",
                location="l",
                description="d",
                severity=Severity.MEDIUM,
            ).model_dump(),
        ]

        result = await adversarial_reviewer.evaluate(
            challenge=challenge,
            team_a_response=json.dumps(findings_a),
            team_b_response=json.dumps([]),
        )

        # Team A: 2/2 valid = 100% precision
        assert result.team_a_score == 1.0
        # Team B: no findings = 0
        assert result.team_b_score == 0.0


@pytest.mark.verification
class TestAdversarialReviewerEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_identical_documentation(self, adversarial_reviewer):
        """Test review when both teams have identical documentation."""
        challenge = AdversarialChallenge(
            challenge_id="chal_edge_1",
            component_id="m.f",
            team_a_documentation="Same documentation",
            team_b_documentation="Same documentation",
            source_code="def f(): pass",
            max_findings=10,
        )

        result = await adversarial_reviewer.evaluate(
            challenge=challenge,
            team_a_response=json.dumps([]),
            team_b_response=json.dumps([]),
        )

        # No findings from either team
        assert len(result.findings_by_a) == 0
        assert len(result.findings_by_b) == 0

    @pytest.mark.asyncio
    async def test_empty_documentation(self, adversarial_reviewer):
        """Test review when documentation is empty."""
        challenge = AdversarialChallenge(
            challenge_id="chal_edge_2",
            component_id="m.f",
            team_a_documentation="",
            team_b_documentation="",
            source_code="def f(): pass",
            max_findings=10,
        )

        result = await adversarial_reviewer.evaluate(
            challenge=challenge,
            team_a_response=json.dumps([]),
            team_b_response=json.dumps([]),
        )

        assert isinstance(result, AdversarialResult)

    @pytest.mark.asyncio
    async def test_no_findings_from_teams(self, adversarial_reviewer):
        """Test when neither team finds any issues."""
        challenge = AdversarialChallenge(
            challenge_id="chal_edge_3",
            component_id="m.f",
            team_a_documentation="Good docs",
            team_b_documentation="Good docs",
            source_code="def f(): pass",
            max_findings=10,
        )

        result = await adversarial_reviewer.evaluate(
            challenge=challenge,
            team_a_response=json.dumps([]),
            team_b_response=json.dumps([]),
        )

        assert result.team_a_score == 0.0
        assert result.team_b_score == 0.0
        assert len(result.validated_findings) == 0


@pytest.mark.verification
class TestDocumentationGapsFromFindings:
    """Tests for creating documentation gaps from validated findings."""

    @pytest.mark.asyncio
    async def test_validated_findings_create_gaps(self, adversarial_reviewer):
        """Test that validated findings create documentation gaps."""
        challenge = AdversarialChallenge(
            challenge_id="chal_gap_1",
            component_id="m.f",
            team_a_documentation="docs",
            team_b_documentation="docs",
            source_code="code",
            max_findings=10,
        )

        finding = AdversarialFinding(
            finding_id="f1",
            reviewed_team="B",
            issue_type="incorrect_value",
            location="line 5",
            description="Wrong value documented",
            severity=Severity.HIGH,
            code_evidence="actual_value = 0.2",
        )

        result = await adversarial_reviewer.evaluate(
            challenge=challenge,
            team_a_response=json.dumps([finding.model_dump()]),
            team_b_response=json.dumps([]),
        )

        # Validated finding should create a documentation gap
        assert len(result.documentation_gaps) > 0
        gap = result.documentation_gaps[0]
        assert gap.area == "incorrect_value"
        assert gap.affects_team_b is True  # Finding was about Team B's docs

    def test_get_documentation_gaps(self, adversarial_reviewer):
        """Test extracting documentation gaps from result."""
        gap = DocumentationGap(
            gap_id="gap_001",
            area="missing_exception",
            description="PaymentError not documented",
            severity=Severity.MEDIUM,
            recommendation="Fix missing_exception at method docs",
            evidence="raise PaymentError(...)",
            affects_team_a=True,
            affects_team_b=False,
        )

        result = AdversarialResult(
            result_id="res_001",
            challenge_id="chal_001",
            component_id="m.f",
            team_a_score=0.8,
            team_b_score=1.0,
            findings_by_a=[],
            findings_by_b=[],
            validated_findings=[],
            false_findings=[],
            documentation_gaps=[gap],
        )

        gaps = adversarial_reviewer.get_documentation_gaps(result)

        assert len(gaps) == 1
        assert gaps[0]["area"] == "missing_exception"
        assert gaps[0]["severity"] == "medium"
        assert gaps[0]["evidence"] == "raise PaymentError(...)"


@pytest.mark.verification
class TestCrossTeamReview:
    """Tests for cross-team review process."""

    @pytest.mark.asyncio
    async def test_team_a_reviews_team_b(
        self,
        adversarial_reviewer,
        sample_verification_function,
    ):
        """Test Team A reviewing Team B's documentation."""
        challenge = AdversarialChallenge(
            challenge_id="chal_cross_1",
            component_id=sample_verification_function.id,
            team_a_documentation="Complete accurate docs",
            team_b_documentation="Docs with errors",
            source_code=sample_verification_function.source_code,
            max_findings=10,
        )

        # Team A finds issues in Team B's docs
        finding = AdversarialFinding(
            finding_id="fa1",
            reviewed_team="B",  # A is reviewing B
            issue_type="incorrect_value",
            location="discount section",
            description="Wrong discount percentage",
            severity=Severity.HIGH,
        )

        result = await adversarial_reviewer.evaluate(
            challenge=challenge,
            team_a_response=json.dumps([finding.model_dump()]),
            team_b_response=json.dumps([]),
        )

        assert len(result.findings_by_a) == 1
        assert result.findings_by_a[0].reviewed_team == "B"

    @pytest.mark.asyncio
    async def test_team_b_reviews_team_a(
        self,
        adversarial_reviewer,
        sample_verification_function,
    ):
        """Test Team B reviewing Team A's documentation."""
        challenge = AdversarialChallenge(
            challenge_id="chal_cross_2",
            component_id=sample_verification_function.id,
            team_a_documentation="Docs with errors",
            team_b_documentation="Complete accurate docs",
            source_code=sample_verification_function.source_code,
            max_findings=10,
        )

        # Team B finds issues in Team A's docs
        finding = AdversarialFinding(
            finding_id="fb1",
            reviewed_team="A",  # B is reviewing A
            issue_type="missing_exception",
            location="exceptions section",
            description="Missing ValueError documentation",
            severity=Severity.MEDIUM,
        )

        result = await adversarial_reviewer.evaluate(
            challenge=challenge,
            team_a_response=json.dumps([]),
            team_b_response=json.dumps([finding.model_dump()]),
        )

        assert len(result.findings_by_b) == 1
        assert result.findings_by_b[0].reviewed_team == "A"

    @pytest.mark.asyncio
    async def test_bidirectional_review(
        self,
        adversarial_reviewer,
        sample_verification_function,
    ):
        """Test simultaneous bidirectional review."""
        challenge = AdversarialChallenge(
            challenge_id="chal_cross_3",
            component_id=sample_verification_function.id,
            team_a_documentation="Team A docs",
            team_b_documentation="Team B docs",
            source_code=sample_verification_function.source_code,
            max_findings=10,
        )

        # Both teams find issues in each other's docs
        finding_a = AdversarialFinding(
            finding_id="fa1",
            reviewed_team="B",
            issue_type="error1",
            location="l1",
            description="d1",
            severity=Severity.HIGH,
        )
        finding_b = AdversarialFinding(
            finding_id="fb1",
            reviewed_team="A",
            issue_type="error2",
            location="l2",
            description="d2",
            severity=Severity.MEDIUM,
        )

        result = await adversarial_reviewer.evaluate(
            challenge=challenge,
            team_a_response=json.dumps([finding_a.model_dump()]),
            team_b_response=json.dumps([finding_b.model_dump()]),
        )

        assert len(result.findings_by_a) == 1
        assert len(result.findings_by_b) == 1
        # Both findings validated
        assert "fa1" in result.validated_findings
        assert "fb1" in result.validated_findings
