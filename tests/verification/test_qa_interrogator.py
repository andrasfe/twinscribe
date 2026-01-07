"""
Unit tests for the QAInterrogator verification strategy.

Tests cover:
- Question generation from source code
- Answer evaluation against ground truth
- Q&A result scoring and gap identification
- Multiple question category handling
- Edge cases in evaluation logic

Related Beads Tickets:
- twinscribe-8pw: Create unit tests for all verification strategies
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from twinscribe.verification import (
    QAChallenge,
    QAEvaluation,
    QAInterrogator,
    QAResult,
    Question,
    QuestionCategory,
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
    """Return sample source code for testing question generation."""
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
# TestQAInterrogatorInit
# =============================================================================


@pytest.mark.verification
class TestQAInterrogatorInit:
    """Tests for QAInterrogator initialization."""

    def test_init_with_default_parameters(self, mock_llm_client):
        """Test initialization with default parameters."""
        interrogator = QAInterrogator(llm_client=mock_llm_client)

        assert interrogator._questions_per_component == 5
        assert interrogator._edge_case_focus is False
        assert interrogator._categories == list(QuestionCategory)

    def test_init_with_custom_questions_per_component(self, mock_llm_client):
        """Test initialization with custom questions per component."""
        interrogator = QAInterrogator(llm_client=mock_llm_client, questions_per_component=10)
        assert interrogator._questions_per_component == 10

    def test_init_with_custom_categories(self, mock_llm_client):
        """Test initialization with custom question categories."""
        custom_categories = [QuestionCategory.RETURN_VALUE, QuestionCategory.ERROR_HANDLING]
        interrogator = QAInterrogator(llm_client=mock_llm_client, categories=custom_categories)
        assert interrogator._categories == custom_categories

    def test_init_with_edge_case_focus(self, mock_llm_client):
        """Test initialization with edge case focus mode."""
        interrogator = QAInterrogator(llm_client=mock_llm_client, edge_case_focus=True)
        assert interrogator._edge_case_focus is True
        # Edge case focus should set specific categories
        assert QuestionCategory.EDGE_CASE in interrogator._categories
        assert QuestionCategory.ERROR_HANDLING in interrogator._categories

    def test_question_templates_defined(self, mock_llm_client):
        """Test that all question category templates are defined."""
        # All question categories should have templates
        for category in QuestionCategory:
            if category in [QuestionCategory.PRECONDITION, QuestionCategory.POSTCONDITION]:
                # These might not have templates
                continue
            assert category in QAInterrogator.QUESTION_TEMPLATES


# =============================================================================
# TestQuestionGeneration
# =============================================================================


@pytest.mark.verification
class TestQuestionGeneration:
    """Tests for question generation from source code."""

    @pytest.mark.asyncio
    async def test_generate_challenge_returns_qa_challenge(
        self, mock_llm_client, sample_source_code
    ):
        """Test that generate_challenge returns a QAChallenge."""
        mock_llm_client.generate.return_value = json.dumps(
            [
                {
                    "question_id": "q_001",
                    "text": "What happens if price is negative?",
                    "category": "error_handling",
                    "correct_answer": "Raises ValueError",
                    "gap_indicator": "Error handling not documented",
                    "difficulty": 3,
                }
            ]
        )

        interrogator = QAInterrogator(llm_client=mock_llm_client)
        challenge = await interrogator.generate_challenge(
            component_id="test.calculate_discount", source_code=sample_source_code
        )

        assert isinstance(challenge, QAChallenge)
        assert challenge.component_id == "test.calculate_discount"
        assert len(challenge.questions) == 1

    @pytest.mark.asyncio
    async def test_generate_challenge_creates_questions_with_correct_fields(
        self, mock_llm_client, sample_source_code
    ):
        """Test that generated questions have all required fields."""
        mock_llm_client.generate.return_value = json.dumps(
            [
                {
                    "question_id": "q_001",
                    "text": "What discount does a premium customer receive?",
                    "category": "return_value",
                    "correct_answer": "20% base discount",
                    "gap_indicator": "Discount value not documented",
                    "difficulty": 2,
                }
            ]
        )

        interrogator = QAInterrogator(llm_client=mock_llm_client)
        challenge = await interrogator.generate_challenge(
            component_id="test.func", source_code=sample_source_code
        )

        question = challenge.questions[0]
        assert isinstance(question, Question)
        assert question.question_id == "q_001"
        assert question.text == "What discount does a premium customer receive?"
        assert question.category == QuestionCategory.RETURN_VALUE
        assert question.correct_answer == "20% base discount"
        assert question.gap_indicator == "Discount value not documented"
        assert question.difficulty == 2

    @pytest.mark.asyncio
    async def test_generate_challenge_respects_num_questions_kwarg(
        self, mock_llm_client, sample_source_code
    ):
        """Test that num_questions kwarg is passed to the prompt."""
        mock_llm_client.generate.return_value = json.dumps(
            [
                {
                    "question_id": f"q_{i:03d}",
                    "text": f"Question {i}?",
                    "category": "return_value",
                    "correct_answer": f"Answer {i}",
                    "difficulty": 3,
                }
                for i in range(7)
            ]
        )

        interrogator = QAInterrogator(llm_client=mock_llm_client)
        challenge = await interrogator.generate_challenge(
            component_id="test.func", source_code=sample_source_code, num_questions=7
        )

        assert len(challenge.questions) == 7
        # Verify the prompt was called with num_questions
        call_args = mock_llm_client.generate.call_args
        assert "7 questions" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_generate_challenge_includes_source_code_metadata(
        self, mock_llm_client, sample_source_code
    ):
        """Test that challenge metadata includes source code length."""
        mock_llm_client.generate.return_value = json.dumps([])

        interrogator = QAInterrogator(llm_client=mock_llm_client)
        challenge = await interrogator.generate_challenge(
            component_id="test.func", source_code=sample_source_code
        )

        assert "source_code_length" in challenge.metadata
        assert challenge.metadata["source_code_length"] == len(sample_source_code)

    @pytest.mark.asyncio
    async def test_generate_challenge_handles_missing_optional_fields(
        self, mock_llm_client, sample_source_code
    ):
        """Test that missing optional fields are handled with defaults."""
        mock_llm_client.generate.return_value = json.dumps(
            [
                {
                    "text": "Simple question?",
                    "category": "edge_case",
                    "correct_answer": "Simple answer",
                    # Missing: question_id, gap_indicator, difficulty
                }
            ]
        )

        interrogator = QAInterrogator(llm_client=mock_llm_client)
        challenge = await interrogator.generate_challenge(
            component_id="test.func", source_code=sample_source_code
        )

        question = challenge.questions[0]
        # Should have auto-generated ID
        assert question.question_id.startswith("q_")
        # Default gap indicator
        assert question.gap_indicator == ""
        # Default difficulty
        assert question.difficulty == 3


# =============================================================================
# TestAnswerEvaluation
# =============================================================================


@pytest.mark.verification
class TestAnswerEvaluation:
    """Tests for evaluating team answers against ground truth."""

    @pytest.fixture
    def sample_challenge(self) -> QAChallenge:
        """Create a sample QAChallenge for testing."""
        return QAChallenge(
            challenge_id="chal_test",
            component_id="test.component",
            questions=[
                Question(
                    question_id="q_001",
                    text="What exception is raised for negative price?",
                    category=QuestionCategory.ERROR_HANDLING,
                    correct_answer="ValueError with message 'Price cannot be negative'",
                    gap_indicator="Error handling not documented",
                    difficulty=3,
                ),
                Question(
                    question_id="q_002",
                    text="What is the premium discount rate?",
                    category=QuestionCategory.RETURN_VALUE,
                    correct_answer="20% (0.2)",
                    gap_indicator="Discount values not specified",
                    difficulty=2,
                ),
            ],
        )

    @pytest.mark.asyncio
    async def test_evaluate_both_teams_correct(self, mock_llm_client, sample_challenge):
        """Test evaluation when both teams answer correctly."""
        # Mock LLM to say both answers are correct
        mock_llm_client.generate.return_value = json.dumps(
            {"is_correct": True, "is_complete": True, "semantic_similarity": 0.95}
        )

        team_a_response = json.dumps(
            [
                {"question_id": "q_001", "answer": "Raises ValueError"},
                {"question_id": "q_002", "answer": "20%"},
            ]
        )
        team_b_response = json.dumps(
            [
                {"question_id": "q_001", "answer": "ValueError is raised"},
                {"question_id": "q_002", "answer": "0.2 (20%)"},
            ]
        )

        interrogator = QAInterrogator(llm_client=mock_llm_client)
        result = await interrogator.evaluate(
            challenge=sample_challenge,
            team_a_response=team_a_response,
            team_b_response=team_b_response,
        )

        assert isinstance(result, QAResult)
        assert len(result.questions_correct_both) == 2
        assert len(result.questions_wrong_both) == 0
        assert len(result.documentation_gaps) == 0

    @pytest.mark.asyncio
    async def test_evaluate_team_a_correct_team_b_wrong(self, mock_llm_client, sample_challenge):
        """Test evaluation when only Team A is correct."""
        call_count = [0]

        async def mock_generate_side_effect(prompt):
            call_count[0] += 1
            # First two calls are for Team A (correct)
            # Next two calls are for Team B (wrong on q_002)
            if call_count[0] <= 2 or call_count[0] == 3:
                return json.dumps(
                    {"is_correct": True, "is_complete": True, "semantic_similarity": 0.9}
                )
            else:
                return json.dumps(
                    {"is_correct": False, "is_complete": False, "semantic_similarity": 0.3}
                )

        mock_llm_client.generate.side_effect = mock_generate_side_effect

        team_a_response = json.dumps(
            [
                {"question_id": "q_001", "answer": "Raises ValueError"},
                {"question_id": "q_002", "answer": "20%"},
            ]
        )
        team_b_response = json.dumps(
            [
                {"question_id": "q_001", "answer": "ValueError is raised"},
                {"question_id": "q_002", "answer": "10%"},  # Wrong
            ]
        )

        interrogator = QAInterrogator(llm_client=mock_llm_client)
        result = await interrogator.evaluate(
            challenge=sample_challenge,
            team_a_response=team_a_response,
            team_b_response=team_b_response,
        )

        # q_001 correct by both, q_002 only by Team A
        assert "q_001" in result.questions_correct_both
        assert "q_002" not in result.questions_correct_both
        assert "q_002" not in result.questions_wrong_both

    @pytest.mark.asyncio
    async def test_evaluate_both_teams_wrong(self, mock_llm_client, sample_challenge):
        """Test evaluation when both teams answer incorrectly."""
        mock_llm_client.generate.return_value = json.dumps(
            {"is_correct": False, "is_complete": False, "semantic_similarity": 0.2}
        )

        team_a_response = json.dumps(
            [
                {"question_id": "q_001", "answer": "Returns None"},
                {"question_id": "q_002", "answer": "5%"},
            ]
        )
        team_b_response = json.dumps(
            [
                {"question_id": "q_001", "answer": "Nothing happens"},
                {"question_id": "q_002", "answer": "15%"},
            ]
        )

        interrogator = QAInterrogator(llm_client=mock_llm_client)
        result = await interrogator.evaluate(
            challenge=sample_challenge,
            team_a_response=team_a_response,
            team_b_response=team_b_response,
        )

        assert len(result.questions_wrong_both) == 2
        assert len(result.documentation_gaps) == 2
        # Check that gaps have the right properties
        for gap in result.documentation_gaps:
            assert gap.affects_team_a is True
            assert gap.affects_team_b is True

    @pytest.mark.asyncio
    async def test_evaluate_handles_missing_answer(self, mock_llm_client, sample_challenge):
        """Test evaluation when team provides no answer for a question."""
        mock_llm_client.generate.return_value = json.dumps(
            {"is_correct": True, "is_complete": True, "semantic_similarity": 0.9}
        )

        # Team A only answers q_001
        team_a_response = json.dumps([{"question_id": "q_001", "answer": "Raises ValueError"}])
        # Team B answers both
        team_b_response = json.dumps(
            [
                {"question_id": "q_001", "answer": "ValueError"},
                {"question_id": "q_002", "answer": "20%"},
            ]
        )

        interrogator = QAInterrogator(llm_client=mock_llm_client)
        result = await interrogator.evaluate(
            challenge=sample_challenge,
            team_a_response=team_a_response,
            team_b_response=team_b_response,
        )

        # Team A's q_002 should be marked as incorrect due to missing answer
        assert len(result.team_a_evaluations) == 2
        q002_eval = next(e for e in result.team_a_evaluations if e.question_id == "q_002")
        assert q002_eval.is_correct is False
        assert q002_eval.semantic_similarity == 0.0

    @pytest.mark.asyncio
    async def test_evaluate_calculates_scores_correctly(self, mock_llm_client, sample_challenge):
        """Test that scores are calculated as average of semantic similarities."""
        mock_llm_client.generate.return_value = json.dumps(
            {"is_correct": True, "is_complete": True, "semantic_similarity": 0.8}
        )

        team_a_response = json.dumps(
            [
                {"question_id": "q_001", "answer": "Raises ValueError"},
                {"question_id": "q_002", "answer": "20%"},
            ]
        )
        team_b_response = json.dumps(
            [
                {"question_id": "q_001", "answer": "ValueError"},
                {"question_id": "q_002", "answer": "20%"},
            ]
        )

        interrogator = QAInterrogator(llm_client=mock_llm_client)
        result = await interrogator.evaluate(
            challenge=sample_challenge,
            team_a_response=team_a_response,
            team_b_response=team_b_response,
        )

        assert result.team_a_score == 0.8
        assert result.team_b_score == 0.8

    @pytest.mark.asyncio
    async def test_evaluate_tracks_category_scores(self, mock_llm_client, sample_challenge):
        """Test that category-level scores are tracked."""
        mock_llm_client.generate.return_value = json.dumps(
            {"is_correct": True, "is_complete": True, "semantic_similarity": 0.85}
        )

        team_a_response = json.dumps(
            [
                {"question_id": "q_001", "answer": "Raises ValueError"},
                {"question_id": "q_002", "answer": "20%"},
            ]
        )
        team_b_response = json.dumps(
            [
                {"question_id": "q_001", "answer": "ValueError"},
                {"question_id": "q_002", "answer": "20%"},
            ]
        )

        interrogator = QAInterrogator(llm_client=mock_llm_client)
        result = await interrogator.evaluate(
            challenge=sample_challenge,
            team_a_response=team_a_response,
            team_b_response=team_b_response,
        )

        assert "error_handling" in result.category_scores
        assert "return_value" in result.category_scores


# =============================================================================
# TestQAResultModel
# =============================================================================


@pytest.mark.verification
class TestQAResultModel:
    """Tests for QAResult data model."""

    def test_qa_result_creation(self):
        """Test creating QAResult from evaluation data."""
        result = QAResult(
            result_id="res_001",
            challenge_id="chal_001",
            component_id="test.component",
            team_a_score=0.85,
            team_b_score=0.75,
            team_a_evaluations=[],
            team_b_evaluations=[],
            questions_correct_both=["q_001"],
            questions_wrong_both=["q_002"],
            category_scores={"return_value": 0.8},
        )

        assert result.result_id == "res_001"
        assert result.team_a_score == 0.85
        assert result.team_b_score == 0.75
        assert result.average_score == 0.8

    def test_qa_result_json_serialization(self):
        """Test JSON serialization of QAResult."""
        result = QAResult(
            result_id="res_001",
            challenge_id="chal_001",
            component_id="test.component",
            team_a_score=0.9,
            team_b_score=0.8,
        )

        # Should be serializable to JSON via model_dump
        json_str = result.model_dump_json()
        assert "res_001" in json_str
        assert "0.9" in json_str

    def test_qa_result_shared_knowledge_gaps_computed(self):
        """Test that shared_knowledge_gaps is computed correctly."""
        result = QAResult(
            result_id="res_001",
            challenge_id="chal_001",
            component_id="test.component",
            questions_wrong_both=["q_001", "q_002", "q_003"],
        )

        assert result.shared_knowledge_gaps == 3

    def test_qa_result_both_teams_failed_computed(self):
        """Test that both_teams_failed is computed correctly."""
        result_failed = QAResult(
            result_id="res_001",
            challenge_id="chal_001",
            component_id="test.component",
            team_a_score=0.4,
            team_b_score=0.3,
        )
        assert result_failed.both_teams_failed is True

        result_passed = QAResult(
            result_id="res_002",
            challenge_id="chal_001",
            component_id="test.component",
            team_a_score=0.6,
            team_b_score=0.7,
        )
        assert result_passed.both_teams_failed is False


# =============================================================================
# TestQuestionModel
# =============================================================================


@pytest.mark.verification
class TestQuestionModel:
    """Tests for Question data model."""

    def test_question_creation(self):
        """Test creating a Question instance."""
        question = Question(
            question_id="q_001",
            text="What happens when input is empty?",
            category=QuestionCategory.EDGE_CASE,
            correct_answer="Returns empty list",
            gap_indicator="Edge case not documented",
            difficulty=3,
        )

        assert question.question_id == "q_001"
        assert question.text == "What happens when input is empty?"
        assert question.category == QuestionCategory.EDGE_CASE
        assert question.correct_answer == "Returns empty list"
        assert question.gap_indicator == "Edge case not documented"
        assert question.difficulty == 3

    def test_question_categories_all_valid(self):
        """Test all question categories are valid."""
        valid_categories = [
            QuestionCategory.RETURN_VALUE,
            QuestionCategory.SIDE_EFFECT,
            QuestionCategory.ERROR_HANDLING,
            QuestionCategory.EDGE_CASE,
            QuestionCategory.DEPENDENCY,
            QuestionCategory.CALL_FLOW,
            QuestionCategory.PRECONDITION,
            QuestionCategory.POSTCONDITION,
        ]

        for cat in valid_categories:
            question = Question(
                question_id="q_test",
                text="Test question?",
                category=cat,
                correct_answer="Test answer",
            )
            assert question.category == cat

    def test_question_json_serialization(self):
        """Test JSON serialization of Question."""
        question = Question(
            question_id="q_001",
            text="Test question?",
            category=QuestionCategory.RETURN_VALUE,
            correct_answer="Test answer",
        )

        json_str = question.model_dump_json()
        assert "q_001" in json_str
        assert "return_value" in json_str

    def test_question_text_minimum_length_validation(self):
        """Test that question text has minimum length validation."""
        with pytest.raises(ValueError):
            Question(
                question_id="q_001",
                text="Short?",  # Less than 10 characters
                category=QuestionCategory.RETURN_VALUE,
                correct_answer="Answer",
            )

    def test_question_difficulty_range_validation(self):
        """Test that difficulty is validated within range 1-5."""
        # Valid difficulties
        for diff in [1, 2, 3, 4, 5]:
            q = Question(
                question_id="q_001",
                text="A valid question text here?",
                category=QuestionCategory.RETURN_VALUE,
                correct_answer="Answer",
                difficulty=diff,
            )
            assert q.difficulty == diff

        # Invalid difficulties
        with pytest.raises(ValueError):
            Question(
                question_id="q_001",
                text="A valid question text here?",
                category=QuestionCategory.RETURN_VALUE,
                correct_answer="Answer",
                difficulty=0,
            )

        with pytest.raises(ValueError):
            Question(
                question_id="q_001",
                text="A valid question text here?",
                category=QuestionCategory.RETURN_VALUE,
                correct_answer="Answer",
                difficulty=6,
            )


# =============================================================================
# TestQAScoring
# =============================================================================


@pytest.mark.verification
class TestQAScoring:
    """Tests for Q&A score aggregation."""

    def test_calculate_qa_score_perfect(self):
        """Test QA score calculation with all correct answers."""
        result = QAResult(
            result_id="res_001",
            challenge_id="chal_001",
            component_id="test.component",
            team_a_score=1.0,
            team_b_score=1.0,
            team_a_evaluations=[
                QAEvaluation(question_id="q_001", is_correct=True, semantic_similarity=1.0),
                QAEvaluation(question_id="q_002", is_correct=True, semantic_similarity=1.0),
            ],
            team_b_evaluations=[
                QAEvaluation(question_id="q_001", is_correct=True, semantic_similarity=1.0),
                QAEvaluation(question_id="q_002", is_correct=True, semantic_similarity=1.0),
            ],
        )

        assert result.team_a_score == 1.0
        assert result.team_b_score == 1.0
        assert result.average_score == 1.0

    def test_calculate_qa_score_partial(self):
        """Test QA score calculation with mixed results."""
        result = QAResult(
            result_id="res_001",
            challenge_id="chal_001",
            component_id="test.component",
            team_a_score=0.7,
            team_b_score=0.5,
            team_a_evaluations=[
                QAEvaluation(question_id="q_001", is_correct=True, semantic_similarity=0.9),
                QAEvaluation(question_id="q_002", is_correct=False, semantic_similarity=0.5),
            ],
            team_b_evaluations=[
                QAEvaluation(question_id="q_001", is_correct=True, semantic_similarity=0.8),
                QAEvaluation(question_id="q_002", is_correct=False, semantic_similarity=0.2),
            ],
        )

        assert result.team_a_score == 0.7
        assert result.team_b_score == 0.5
        assert result.average_score == 0.6

    def test_calculate_qa_score_zero(self):
        """Test QA score when no questions answered correctly."""
        result = QAResult(
            result_id="res_001",
            challenge_id="chal_001",
            component_id="test.component",
            team_a_score=0.0,
            team_b_score=0.0,
            team_a_evaluations=[
                QAEvaluation(question_id="q_001", is_correct=False, semantic_similarity=0.0),
                QAEvaluation(question_id="q_002", is_correct=False, semantic_similarity=0.0),
            ],
            team_b_evaluations=[
                QAEvaluation(question_id="q_001", is_correct=False, semantic_similarity=0.0),
                QAEvaluation(question_id="q_002", is_correct=False, semantic_similarity=0.0),
            ],
        )

        assert result.team_a_score == 0.0
        assert result.team_b_score == 0.0
        assert result.average_score == 0.0


# =============================================================================
# TestQAInterrogatorEdgeCases
# =============================================================================


@pytest.mark.verification
class TestQAInterrogatorEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_handle_llm_error_during_generation(self, mock_llm_client, sample_source_code):
        """Test graceful handling of LLM errors during question generation."""
        mock_llm_client.generate.side_effect = Exception("LLM API error")

        interrogator = QAInterrogator(llm_client=mock_llm_client)

        with pytest.raises(Exception, match="LLM API error"):
            await interrogator.generate_challenge(
                component_id="test.func", source_code=sample_source_code
            )

    @pytest.mark.asyncio
    async def test_handle_malformed_llm_response(self, mock_llm_client, sample_source_code):
        """Test handling of malformed JSON responses from LLM."""
        mock_llm_client.generate.return_value = "not valid json"

        interrogator = QAInterrogator(llm_client=mock_llm_client)

        with pytest.raises(json.JSONDecodeError):
            await interrogator.generate_challenge(
                component_id="test.func", source_code=sample_source_code
            )

    @pytest.mark.asyncio
    async def test_handle_empty_questions_list(self, mock_llm_client, sample_source_code):
        """Test handling when LLM returns empty questions list."""
        mock_llm_client.generate.return_value = json.dumps([])

        interrogator = QAInterrogator(llm_client=mock_llm_client)
        challenge = await interrogator.generate_challenge(
            component_id="test.func", source_code=sample_source_code
        )

        assert challenge.questions == []

    @pytest.mark.asyncio
    async def test_get_documentation_gaps_from_result(self, mock_llm_client):
        """Test extracting documentation gaps from Q&A result."""
        from twinscribe.verification.models import DocumentationGap, Severity

        result = QAResult(
            result_id="res_001",
            challenge_id="chal_001",
            component_id="test.component",
            documentation_gaps=[
                DocumentationGap(
                    gap_id="gap_001",
                    area="error_handling",
                    description="Exception handling not documented",
                    severity=Severity.HIGH,
                    recommendation="Add exception documentation",
                )
            ],
        )

        interrogator = QAInterrogator(llm_client=mock_llm_client)
        gaps = interrogator.get_documentation_gaps(result)

        assert len(gaps) == 1
        assert gaps[0]["area"] == "error_handling"
        assert gaps[0]["severity"] == "high"


# =============================================================================
# TestQAInterrogatorIntegration
# =============================================================================


@pytest.mark.verification
class TestQAInterrogatorIntegration:
    """Integration tests for QAInterrogator with other components."""

    @pytest.mark.asyncio
    async def test_full_qa_workflow(self, mock_llm_client, sample_source_code):
        """Test complete Q&A interrogation workflow."""
        # Mock for question generation
        mock_llm_client.generate.return_value = json.dumps(
            [
                {
                    "question_id": "q_001",
                    "text": "What happens if price is negative?",
                    "category": "error_handling",
                    "correct_answer": "Raises ValueError",
                    "gap_indicator": "Error handling gap",
                    "difficulty": 3,
                }
            ]
        )

        interrogator = QAInterrogator(llm_client=mock_llm_client)

        # Step 1: Generate challenge
        challenge = await interrogator.generate_challenge(
            component_id="test.calculate_discount", source_code=sample_source_code
        )
        assert len(challenge.questions) == 1

        # Step 2: Mock evaluation response
        mock_llm_client.generate.return_value = json.dumps(
            {"is_correct": True, "is_complete": True, "semantic_similarity": 0.9}
        )

        # Step 3: Evaluate responses
        team_a_response = json.dumps([{"question_id": "q_001", "answer": "Raises ValueError"}])
        team_b_response = json.dumps(
            [{"question_id": "q_001", "answer": "Raises ValueError exception"}]
        )

        result = await interrogator.evaluate(
            challenge=challenge, team_a_response=team_a_response, team_b_response=team_b_response
        )

        # Step 4: Verify result
        assert isinstance(result, QAResult)
        assert result.team_a_score > 0
        assert result.team_b_score > 0

    @pytest.mark.asyncio
    async def test_strategy_properties(self, mock_llm_client):
        """Test strategy type and level properties."""
        from twinscribe.verification import StrategyType, VerificationLevel

        interrogator = QAInterrogator(llm_client=mock_llm_client)

        assert interrogator.strategy_type == StrategyType.QA_INTERROGATION
        assert interrogator.level == VerificationLevel.ACTIVE
        assert (
            "Q&A" in interrogator.description or "documentation" in interrogator.description.lower()
        )

    @pytest.mark.asyncio
    async def test_edge_case_focus_changes_strategy_type(self, mock_llm_client):
        """Test that edge_case_focus changes the strategy type."""
        from twinscribe.verification import StrategyType

        interrogator = QAInterrogator(llm_client=mock_llm_client, edge_case_focus=True)

        assert interrogator.strategy_type == StrategyType.EDGE_CASE_EXTRACTION
