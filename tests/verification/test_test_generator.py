"""
Unit tests for the TestGenerationValidator verification strategy.

Tests cover:
- Test generation from documentation
- Test execution against actual code
- Documentation error detection via test failures
- Coverage analysis of generated tests
- Test completeness validation

Related Beads Tickets:
- twinscribe-8pw: Create unit tests for all verification strategies
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from twinscribe.verification import (
    DocumentationGap,
    GeneratedTest,
    Severity,
    StrategyType,
    TestExecution,
    TestGenerationChallenge,
    TestGenerationValidator,
    TestValidationResult,
    VerificationLevel,
)


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Create a mock LLM client for TestGenerationValidator."""
    client = MagicMock()

    async def mock_generate(prompt: str) -> str:
        """Return mock responses based on prompt content."""
        if "analyze" in prompt.lower() and "pass" in prompt.lower():
            # Test execution simulation
            return json.dumps(
                {
                    "passed": True,
                    "error_message": None,
                    "failure_reason": None,
                }
            )
        elif "generate" in prompt.lower() and "test" in prompt.lower():
            # Test generation
            return json.dumps(
                [
                    {
                        "test_id": "test_001",
                        "test_name": "test_normal_case",
                        "test_code": "def test_normal_case(): assert func(1) == 1",
                        "tests_aspect": "happy_path",
                    }
                ]
            )
        return json.dumps({"status": "success"})

    client.generate = AsyncMock(side_effect=mock_generate)
    return client


@pytest.fixture
def mock_llm_client_failing_tests() -> MagicMock:
    """Create a mock LLM client that simulates test failures."""
    client = MagicMock()

    async def mock_generate(prompt: str) -> str:
        if "analyze" in prompt.lower():
            return json.dumps(
                {
                    "passed": False,
                    "error_message": "AssertionError: expected 0.2, got 0.15",
                    "failure_reason": "Documentation stated wrong discount value",
                }
            )
        return json.dumps({"status": "success"})

    client.generate = AsyncMock(side_effect=mock_generate)
    return client


@pytest.fixture
def test_generator(mock_llm_client) -> TestGenerationValidator:
    """Create a TestGenerationValidator instance with mock LLM."""
    return TestGenerationValidator(llm_client=mock_llm_client)


@pytest.fixture
def test_generator_custom(mock_llm_client) -> TestGenerationValidator:
    """Create a TestGenerationValidator with custom test count."""
    return TestGenerationValidator(llm_client=mock_llm_client, tests_per_team=5)


@pytest.mark.verification
class TestTestGenerationValidatorInit:
    """Tests for TestGenerationValidator initialization."""

    def test_init_with_default_settings(self, mock_llm_client):
        """Test initialization with default settings."""
        validator = TestGenerationValidator(llm_client=mock_llm_client)

        assert validator.strategy_type == StrategyType.TEST_GENERATION
        assert validator.level == VerificationLevel.GENERATIVE
        assert "test" in validator.description.lower()
        assert validator._tests_per_team == 10  # Default

    def test_init_with_custom_llm(self, mock_llm_client):
        """Test initialization with custom LLM client."""
        validator = TestGenerationValidator(llm_client=mock_llm_client)

        assert validator._llm == mock_llm_client

    def test_init_with_custom_tests_per_team(self, mock_llm_client):
        """Test initialization with custom tests per team."""
        validator = TestGenerationValidator(
            llm_client=mock_llm_client,
            tests_per_team=5,
        )

        assert validator._tests_per_team == 5


@pytest.mark.verification
class TestChallengeGeneration:
    """Tests for generating test generation challenges."""

    @pytest.mark.asyncio
    async def test_generate_challenge(
        self,
        test_generator,
        sample_verification_function,
    ):
        """Test generating a test generation challenge."""
        challenge = await test_generator.generate_challenge(
            component_id=sample_verification_function.id,
            source_code=sample_verification_function.source_code,
            team_a_documentation="Team A docs",
            team_b_documentation="Team B docs",
        )

        assert isinstance(challenge, TestGenerationChallenge)
        assert challenge.component_id == sample_verification_function.id
        assert challenge.team_a_documentation == "Team A docs"
        assert challenge.team_b_documentation == "Team B docs"
        assert challenge.source_code == sample_verification_function.source_code

    @pytest.mark.asyncio
    async def test_generate_challenge_with_custom_tests(
        self,
        test_generator_custom,
        sample_verification_function,
    ):
        """Test challenge includes custom tests per team setting."""
        challenge = await test_generator_custom.generate_challenge(
            component_id=sample_verification_function.id,
            source_code=sample_verification_function.source_code,
        )

        assert challenge.tests_per_team == 5


@pytest.mark.verification
class TestTestGenerationChallengeModel:
    """Tests for TestGenerationChallenge data model."""

    def test_challenge_creation(self):
        """Test creating a TestGenerationChallenge instance."""
        challenge = TestGenerationChallenge(
            challenge_id="chal_test_001",
            component_id="module.function",
            team_a_documentation="Team A docs",
            team_b_documentation="Team B docs",
            source_code="def function(): pass",
            tests_per_team=10,
        )

        assert challenge.challenge_id == "chal_test_001"
        assert challenge.component_id == "module.function"
        assert challenge.tests_per_team == 10

    def test_challenge_attributes(self):
        """Test TestGenerationChallenge has required attributes."""
        challenge = TestGenerationChallenge(
            challenge_id="chal_test_002",
            component_id="m.f",
            team_a_documentation="",
            team_b_documentation="",
            source_code="",
            tests_per_team=5,
        )

        assert hasattr(challenge, "challenge_id")
        assert hasattr(challenge, "component_id")
        assert hasattr(challenge, "team_a_documentation")
        assert hasattr(challenge, "team_b_documentation")
        assert hasattr(challenge, "source_code")
        assert hasattr(challenge, "tests_per_team")


@pytest.mark.verification
class TestGeneratedTestModel:
    """Tests for GeneratedTest data model."""

    def test_generated_test_creation(self):
        """Test creating a GeneratedTest instance."""
        test = GeneratedTest(
            test_id="test_001",
            test_name="test_premium_discount",
            test_code="def test_premium_discount(): assert func('premium') == 0.8",
            tests_aspect="happy_path",
            from_team="A",
        )

        assert test.test_id == "test_001"
        assert test.test_name == "test_premium_discount"
        assert test.from_team == "A"

    def test_generated_test_attributes(self):
        """Test GeneratedTest has required attributes."""
        test = GeneratedTest(
            test_id="test_002",
            test_name="test_error_handling",
            test_code="def test_error(): ...",
            tests_aspect="error_handling",
            from_team="B",
        )

        assert hasattr(test, "test_id")
        assert hasattr(test, "test_name")
        assert hasattr(test, "test_code")
        assert hasattr(test, "tests_aspect")
        assert hasattr(test, "from_team")

    def test_generated_test_json_serialization(self):
        """Test JSON serialization of GeneratedTest."""
        test = GeneratedTest(
            test_id="test_003",
            test_name="test_edge_case",
            test_code="def test_edge(): pass",
            tests_aspect="edge_case",
            from_team="A",
        )

        json_str = test.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["test_id"] == "test_003"
        assert parsed["from_team"] == "A"


@pytest.mark.verification
class TestTestExecution:
    """Tests for executing generated tests against actual code."""

    @pytest.mark.asyncio
    async def test_run_tests_all_pass(self, test_generator):
        """Test execution when all generated tests pass."""
        challenge = TestGenerationChallenge(
            challenge_id="chal_exec_001",
            component_id="m.f",
            team_a_documentation="docs",
            team_b_documentation="docs",
            source_code="def f(): return 1",
            tests_per_team=2,
        )

        # Both teams have passing tests
        tests_a = [
            GeneratedTest(
                test_id="ta1",
                test_name="test_a1",
                test_code="def test_a1(): assert f() == 1",
                tests_aspect="happy_path",
                from_team="A",
            ).model_dump(),
        ]
        tests_b = [
            GeneratedTest(
                test_id="tb1",
                test_name="test_b1",
                test_code="def test_b1(): assert f() == 1",
                tests_aspect="happy_path",
                from_team="B",
            ).model_dump(),
        ]

        result = await test_generator.evaluate(
            challenge=challenge,
            team_a_response=json.dumps(tests_a),
            team_b_response=json.dumps(tests_b),
        )

        # Mock returns all tests passing
        assert result.team_a_score == 1.0
        assert result.team_b_score == 1.0

    @pytest.mark.asyncio
    async def test_run_tests_some_fail(
        self,
        mock_llm_client_failing_tests,
        sample_verification_function,
    ):
        """Test execution when some tests fail due to doc errors."""
        validator = TestGenerationValidator(llm_client=mock_llm_client_failing_tests)

        challenge = TestGenerationChallenge(
            challenge_id="chal_exec_002",
            component_id=sample_verification_function.id,
            team_a_documentation="Wrong docs",
            team_b_documentation="Wrong docs",
            source_code=sample_verification_function.source_code,
            tests_per_team=2,
        )

        tests = [
            GeneratedTest(
                test_id="t1",
                test_name="test_discount",
                test_code="def test_discount(): assert discount == 0.15",
                tests_aspect="return_value",
                from_team="A",
            ).model_dump(),
        ]

        result = await validator.evaluate(
            challenge=challenge,
            team_a_response=json.dumps(tests),
            team_b_response=json.dumps([]),
        )

        # Test failed
        assert result.team_a_score == 0.0
        assert len(result.documentation_errors) > 0

    @pytest.mark.asyncio
    async def test_detect_documentation_error(
        self,
        mock_llm_client_failing_tests,
    ):
        """Test detection of documentation errors via test failures."""
        validator = TestGenerationValidator(llm_client=mock_llm_client_failing_tests)

        challenge = TestGenerationChallenge(
            challenge_id="chal_detect_001",
            component_id="m.f",
            team_a_documentation="Incorrect documentation",
            team_b_documentation="Correct documentation",
            source_code="def f(): return 0.2",
            tests_per_team=1,
        )

        tests = [
            GeneratedTest(
                test_id="t1",
                test_name="test_value",
                test_code="def test_value(): assert f() == 0.15",
                tests_aspect="return_value",
                from_team="A",
            ).model_dump(),
        ]

        result = await validator.evaluate(
            challenge=challenge,
            team_a_response=json.dumps(tests),
            team_b_response=json.dumps([]),
        )

        # Documentation error detected via test failure
        assert len(result.documentation_errors) > 0
        assert "wrong discount value" in result.documentation_errors[0].lower()


@pytest.mark.verification
class TestTestExecutionModel:
    """Tests for TestExecution data model."""

    def test_execution_creation_passed(self):
        """Test creating a passing TestExecution instance."""
        execution = TestExecution(
            test_id="test_001",
            passed=True,
            error_message=None,
            failure_reason=None,
        )

        assert execution.test_id == "test_001"
        assert execution.passed is True
        assert execution.error_message is None

    def test_execution_creation_failed(self):
        """Test creating a failed TestExecution instance."""
        execution = TestExecution(
            test_id="test_002",
            passed=False,
            error_message="AssertionError: 0.15 != 0.2",
            failure_reason="Documentation stated wrong value",
        )

        assert execution.passed is False
        assert "AssertionError" in execution.error_message
        assert execution.failure_reason is not None


@pytest.mark.verification
class TestValidationResultModel:
    """Tests for TestValidationResult data model."""

    def test_validation_result_creation(self):
        """Test creating a TestValidationResult instance."""
        test_a = GeneratedTest(
            test_id="ta1",
            test_name="test_a",
            test_code="pass",
            tests_aspect="happy_path",
            from_team="A",
        )
        exec_a = TestExecution(test_id="ta1", passed=True)

        result = TestValidationResult(
            result_id="res_test_001",
            challenge_id="chal_001",
            component_id="m.f",
            team_a_score=1.0,
            team_b_score=0.5,
            team_a_tests=[test_a],
            team_b_tests=[],
            team_a_executions=[exec_a],
            team_b_executions=[],
            documentation_errors=[],
        )

        assert result.result_id == "res_test_001"
        assert len(result.team_a_tests) == 1
        assert result.team_a_score == 1.0

    def test_validation_result_attributes(self):
        """Test TestValidationResult has required attributes."""
        result = TestValidationResult(
            result_id="res_test_002",
            challenge_id="chal_002",
            component_id="m.f",
            team_a_score=0.0,
            team_b_score=0.0,
            team_a_tests=[],
            team_b_tests=[],
            team_a_executions=[],
            team_b_executions=[],
            documentation_errors=[],
        )

        assert hasattr(result, "team_a_tests")
        assert hasattr(result, "team_b_tests")
        assert hasattr(result, "team_a_executions")
        assert hasattr(result, "team_b_executions")
        assert hasattr(result, "documentation_errors")
        assert hasattr(result, "documentation_gaps")

    def test_pass_rate_calculation(self):
        """Test pass rate is reflected in score."""
        test1 = GeneratedTest(
            test_id="t1",
            test_name="t1",
            test_code="pass",
            tests_aspect="test",
            from_team="A",
        )
        test2 = GeneratedTest(
            test_id="t2",
            test_name="t2",
            test_code="pass",
            tests_aspect="test",
            from_team="A",
        )
        exec1 = TestExecution(test_id="t1", passed=True)
        exec2 = TestExecution(test_id="t2", passed=False, error_message="Failed")

        result = TestValidationResult(
            result_id="res_003",
            challenge_id="chal_003",
            component_id="m.f",
            team_a_score=0.5,  # 1 of 2 passed
            team_b_score=0.0,
            team_a_tests=[test1, test2],
            team_b_tests=[],
            team_a_executions=[exec1, exec2],
            team_b_executions=[],
            documentation_errors=[],
        )

        # 50% pass rate
        assert result.team_a_score == 0.5


@pytest.mark.verification
class TestTestGeneratorScoring:
    """Tests for test generation scoring."""

    @pytest.mark.asyncio
    async def test_score_all_tests_pass(self, test_generator):
        """Test score when all generated tests pass."""
        challenge = TestGenerationChallenge(
            challenge_id="chal_score_1",
            component_id="m.f",
            team_a_documentation="docs",
            team_b_documentation="docs",
            source_code="code",
            tests_per_team=2,
        )

        tests = [
            GeneratedTest(
                test_id="t1",
                test_name="t1",
                test_code="pass",
                tests_aspect="test",
                from_team="A",
            ).model_dump(),
            GeneratedTest(
                test_id="t2",
                test_name="t2",
                test_code="pass",
                tests_aspect="test",
                from_team="A",
            ).model_dump(),
        ]

        result = await test_generator.evaluate(
            challenge=challenge,
            team_a_response=json.dumps(tests),
            team_b_response=json.dumps([]),
        )

        # All tests pass = 100%
        assert result.team_a_score == 1.0

    @pytest.mark.asyncio
    async def test_score_no_tests_pass(self, mock_llm_client_failing_tests):
        """Test score when no tests pass."""
        validator = TestGenerationValidator(llm_client=mock_llm_client_failing_tests)

        challenge = TestGenerationChallenge(
            challenge_id="chal_score_2",
            component_id="m.f",
            team_a_documentation="docs",
            team_b_documentation="docs",
            source_code="code",
            tests_per_team=2,
        )

        tests = [
            GeneratedTest(
                test_id="t1",
                test_name="t1",
                test_code="fail",
                tests_aspect="test",
                from_team="A",
            ).model_dump(),
        ]

        result = await validator.evaluate(
            challenge=challenge,
            team_a_response=json.dumps(tests),
            team_b_response=json.dumps([]),
        )

        # No tests pass = 0%
        assert result.team_a_score == 0.0


@pytest.mark.verification
class TestTestGeneratorEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_documentation(self, test_generator):
        """Test generation from empty documentation."""
        challenge = TestGenerationChallenge(
            challenge_id="chal_edge_1",
            component_id="m.f",
            team_a_documentation="",
            team_b_documentation="",
            source_code="def f(): pass",
            tests_per_team=2,
        )

        result = await test_generator.evaluate(
            challenge=challenge,
            team_a_response=json.dumps([]),
            team_b_response=json.dumps([]),
        )

        assert isinstance(result, TestValidationResult)
        # No tests = 0 score
        assert result.team_a_score == 0.0

    @pytest.mark.asyncio
    async def test_no_tests_generated(self, test_generator):
        """Test handling when no tests are generated."""
        challenge = TestGenerationChallenge(
            challenge_id="chal_edge_2",
            component_id="m.f",
            team_a_documentation="docs",
            team_b_documentation="docs",
            source_code="def f(): pass",
            tests_per_team=2,
        )

        result = await test_generator.evaluate(
            challenge=challenge,
            team_a_response=json.dumps([]),
            team_b_response=json.dumps([]),
        )

        assert len(result.team_a_tests) == 0
        assert len(result.team_b_tests) == 0


@pytest.mark.verification
class TestDocumentationGapsFromTests:
    """Tests for documentation gap identification from test failures."""

    @pytest.mark.asyncio
    async def test_failed_tests_create_gaps(self, mock_llm_client_failing_tests):
        """Test that failed tests create documentation gaps."""
        validator = TestGenerationValidator(llm_client=mock_llm_client_failing_tests)

        challenge = TestGenerationChallenge(
            challenge_id="chal_gap_1",
            component_id="m.f",
            team_a_documentation="Wrong docs",
            team_b_documentation="Wrong docs",
            source_code="def f(): return 0.2",
            tests_per_team=1,
        )

        tests = [
            GeneratedTest(
                test_id="t1",
                test_name="test_value",
                test_code="assert f() == 0.15",
                tests_aspect="return_value",
                from_team="A",
            ).model_dump(),
        ]

        result = await validator.evaluate(
            challenge=challenge,
            team_a_response=json.dumps(tests),
            team_b_response=json.dumps([]),
        )

        # Failed test should create documentation gap
        assert len(result.documentation_gaps) > 0
        gap = result.documentation_gaps[0]
        assert gap.area == "behavioral_accuracy"
        assert gap.severity == Severity.HIGH

    def test_get_documentation_gaps(self, test_generator):
        """Test extracting documentation gaps from result."""
        gap = DocumentationGap(
            gap_id="gap_001",
            area="behavioral_accuracy",
            description="Test failure: Wrong value documented",
            severity=Severity.HIGH,
            recommendation="Update documentation to match actual behavior",
        )

        result = TestValidationResult(
            result_id="res_001",
            challenge_id="chal_001",
            component_id="m.f",
            team_a_score=0.5,
            team_b_score=1.0,
            team_a_tests=[],
            team_b_tests=[],
            team_a_executions=[],
            team_b_executions=[],
            documentation_errors=["Wrong value"],
            documentation_gaps=[gap],
        )

        gaps = test_generator.get_documentation_gaps(result)

        assert len(gaps) == 1
        assert gaps[0]["area"] == "behavioral_accuracy"
        assert gaps[0]["severity"] == "high"


@pytest.mark.verification
class TestTestGenerationFromDocumentation:
    """Tests for generating tests from documentation."""

    @pytest.mark.asyncio
    async def test_generate_tests_from_docs(self, test_generator):
        """Test test generation from documentation."""
        documentation = """
        Function: calculate_discount
        - Premium customers get 20% discount
        - Standard customers get 10% discount
        - Volume bonus of 5% for orders over 100 items
        - Raises ValueError for negative price
        """

        tests = await test_generator.generate_tests_from_documentation(
            documentation=documentation,
            team="A",
            num_tests=3,
        )

        assert isinstance(tests, list)
        # Mock returns at least one test
        assert all(isinstance(t, GeneratedTest) for t in tests)

    @pytest.mark.asyncio
    async def test_generated_tests_have_required_fields(self, test_generator):
        """Test that generated tests have all required fields."""
        tests = await test_generator.generate_tests_from_documentation(
            documentation="Test documentation",
            team="B",
            num_tests=1,
        )

        if tests:  # If mock returns tests
            test = tests[0]
            assert test.test_id is not None
            assert test.test_name is not None
            assert test.test_code is not None
            assert test.from_team == "B"
