"""
CrossCheck Verification Framework - Strategy Implementations.

This module provides concrete implementations of all verification strategies:

1. QAInterrogator - Q&A interrogation strategy
2. MaskedReconstructor - Masked code reconstruction strategy
3. ScenarioWalker - Scenario walkthrough strategy
4. MutationDetector - Mutation detection strategy
5. ImpactAnalyzer - Impact analysis strategy
6. AdversarialReviewer - Adversarial review strategy
7. TestGenerator - Test generation validation strategy
8. CodeReconstructor - Code reconstruction strategy

Each strategy follows the VerificationStrategy protocol and can be used
independently or orchestrated through the VerificationPipeline.
"""

import json
import random
import re
import uuid
from typing import Protocol

from twinscribe.verification.base import (
    ChangeType,
    MaskType,
    MutationType,
    QuestionCategory,
    ScenarioType,
    Severity,
    StrategyType,
    VerificationLevel,
    VerificationStrategy,
)
from twinscribe.verification.models import (
    AdversarialChallenge,
    AdversarialFinding,
    AdversarialResult,
    CodeReconstructionChallenge,
    CodeReconstructionResult,
    DocumentationGap,
    ExecutionTrace,
    GeneratedTest,
    ImpactChallenge,
    ImpactPrediction,
    ImpactResult,
    Mask,
    MaskedChallenge,
    MaskEvaluation,
    MaskReconstruction,
    Mutation,
    MutationAssessment,
    MutationChallenge,
    MutationEvaluation,
    MutationResult,
    QAChallenge,
    QAEvaluation,
    QAResult,
    Question,
    ReconstructedCode,
    ReconstructionResult,
    Scenario,
    ScenarioChallenge,
    ScenarioEvaluation,
    ScenarioResult,
    TeamAnswer,
    TestExecution,
    TestGenerationChallenge,
    TestValidationResult,
)


class LLMClient(Protocol):
    """Protocol for LLM client interface.

    Any LLM client that implements this protocol can be used
    with the verification strategies.
    """

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: The prompt to send
            **kwargs: Additional generation parameters

        Returns:
            The generated response text
        """
        ...


def _generate_id(prefix: str) -> str:
    """Generate a unique ID with given prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# =============================================================================
# Q&A Interrogation Strategy
# =============================================================================


class QAInterrogator(VerificationStrategy[QAChallenge, QAResult]):
    """Q&A interrogation strategy.

    Generates questions about code behavior that teams must answer
    using only their documentation. Answers are compared against
    ground truth from code analysis.

    This strategy tests:
        - Knowledge completeness
        - Understanding of edge cases
        - Error handling documentation
        - Call flow accuracy
    """

    # Mask patterns for different element types
    QUESTION_TEMPLATES = {
        QuestionCategory.RETURN_VALUE: [
            "What does {func} return when {condition}?",
            "What is the return type of {func} when {condition}?",
            "Under what conditions does {func} return None?",
        ],
        QuestionCategory.ERROR_HANDLING: [
            "What exceptions can {func} raise?",
            "What happens if {func} receives {invalid_input}?",
            "How does {func} handle {error_condition}?",
        ],
        QuestionCategory.EDGE_CASE: [
            "What happens if {func} receives an empty {input_type}?",
            "How does {func} behave at the boundary of {boundary}?",
            "What is the behavior when {edge_condition}?",
        ],
        QuestionCategory.CALL_FLOW: [
            "Which functions are called when {func} processes {input}?",
            "Trace the call path from {func} to {target}.",
            "What is the order of function calls when {condition}?",
        ],
        QuestionCategory.SIDE_EFFECT: [
            "Does {func} modify any global state?",
            "What side effects does {func} have?",
            "Does {func} write to any files or databases?",
        ],
        QuestionCategory.DEPENDENCY: [
            "What must be initialized before calling {func}?",
            "What are the preconditions for {func}?",
            "Which services or resources does {func} depend on?",
        ],
    }

    def __init__(
        self,
        llm_client: LLMClient,
        questions_per_component: int = 5,
        categories: list[QuestionCategory] | None = None,
        edge_case_focus: bool = False,
    ) -> None:
        """Initialize the Q&A interrogator.

        Args:
            llm_client: LLM client for generating questions and evaluating answers
            questions_per_component: Number of questions to generate per component
            categories: Question categories to use (default: all)
            edge_case_focus: Focus primarily on edge case questions
        """
        # Determine strategy type based on focus mode
        strategy_type = (
            StrategyType.EDGE_CASE_EXTRACTION if edge_case_focus else StrategyType.QA_INTERROGATION
        )

        super().__init__(
            strategy_type=strategy_type,
            level=VerificationLevel.ACTIVE,
            description="Tests documentation completeness through Q&A examination",
        )
        self._llm = llm_client
        self._questions_per_component = questions_per_component
        self._edge_case_focus = edge_case_focus

        # If edge case focus, prioritize edge case categories
        if edge_case_focus:
            self._categories = [
                QuestionCategory.EDGE_CASE,
                QuestionCategory.ERROR_HANDLING,
                QuestionCategory.PRECONDITION,
                QuestionCategory.POSTCONDITION,
            ]
        else:
            self._categories = categories or list(QuestionCategory)

    async def generate_challenge(
        self,
        component_id: str,
        source_code: str,
        **kwargs,
    ) -> QAChallenge:
        """Generate Q&A questions from source code.

        Args:
            component_id: Component identifier
            source_code: Component source code
            **kwargs: Additional parameters (num_questions, categories)

        Returns:
            QAChallenge with generated questions
        """
        num_questions = kwargs.get("num_questions", self._questions_per_component)
        categories = kwargs.get("categories", self._categories)

        prompt = f"""
You are examining this code to generate verification questions.
The questions will test whether documentation accurately describes behavior.

SOURCE CODE:
```python
{source_code}
```

Generate {num_questions} questions that:
1. Have definitive answers derivable from the code
2. Test understanding of behavior, not just syntax
3. Cover different aspects: {", ".join(c.value for c in categories)}
4. Would expose documentation gaps if answered incorrectly

For each question, provide:
- question_id: unique identifier (q_001, q_002, etc.)
- text: the question
- category: one of {", ".join(c.value for c in categories)}
- correct_answer: the correct answer from code analysis
- gap_indicator: what documentation gap a wrong answer would indicate
- difficulty: 1-5

Output as JSON array.
"""

        response = await self._llm.generate(prompt)
        questions_data = json.loads(response)

        questions = [
            Question(
                question_id=q.get("question_id", _generate_id("q")),
                text=q["text"],
                category=QuestionCategory(q["category"]),
                correct_answer=q["correct_answer"],
                gap_indicator=q.get("gap_indicator", ""),
                difficulty=q.get("difficulty", 3),
            )
            for q in questions_data
        ]

        return QAChallenge(
            challenge_id=_generate_id("chal_qa"),
            component_id=component_id,
            questions=questions,
            metadata={"source_code_length": len(source_code)},
        )

    async def evaluate(
        self,
        challenge: QAChallenge,
        team_a_response: str,
        team_b_response: str,
        ground_truth: str | None = None,
    ) -> QAResult:
        """Evaluate team answers against ground truth.

        Args:
            challenge: The Q&A challenge
            team_a_response: JSON string of TeamAnswer list from Team A
            team_b_response: JSON string of TeamAnswer list from Team B
            ground_truth: Optional additional ground truth

        Returns:
            QAResult with evaluation details
        """
        team_a_answers = [TeamAnswer(**a) for a in json.loads(team_a_response)]
        team_b_answers = [TeamAnswer(**a) for a in json.loads(team_b_response)]

        # Build lookup for answers
        a_lookup = {a.question_id: a for a in team_a_answers}
        b_lookup = {a.question_id: a for a in team_b_answers}

        team_a_evaluations = []
        team_b_evaluations = []
        questions_correct_both = []
        questions_wrong_both = []
        category_scores: dict[str, list[float]] = {}
        documentation_gaps = []

        for question in challenge.questions:
            a_answer = a_lookup.get(question.question_id)
            b_answer = b_lookup.get(question.question_id)

            # Evaluate each answer
            a_eval = await self._evaluate_single_answer(question, a_answer, ground_truth)
            b_eval = await self._evaluate_single_answer(question, b_answer, ground_truth)

            team_a_evaluations.append(a_eval)
            team_b_evaluations.append(b_eval)

            # Track shared correct/wrong
            if a_eval.is_correct and b_eval.is_correct:
                questions_correct_both.append(question.question_id)
            elif not a_eval.is_correct and not b_eval.is_correct:
                questions_wrong_both.append(question.question_id)
                # Both teams wrong = documentation gap
                documentation_gaps.append(
                    DocumentationGap(
                        gap_id=_generate_id("gap"),
                        area=question.category.value,
                        description=question.gap_indicator or f"Both teams failed: {question.text}",
                        severity=Severity.HIGH if question.difficulty >= 3 else Severity.MEDIUM,
                        recommendation=f"Add documentation for: {question.text}",
                        evidence=f"Correct answer: {question.correct_answer}",
                        affects_team_a=True,
                        affects_team_b=True,
                    )
                )

            # Track category scores
            cat = question.category.value
            if cat not in category_scores:
                category_scores[cat] = []
            category_scores[cat].append(
                (a_eval.semantic_similarity + b_eval.semantic_similarity) / 2
            )

        # Calculate overall scores
        team_a_score = (
            sum(e.semantic_similarity for e in team_a_evaluations) / len(team_a_evaluations)
            if team_a_evaluations
            else 0.0
        )
        team_b_score = (
            sum(e.semantic_similarity for e in team_b_evaluations) / len(team_b_evaluations)
            if team_b_evaluations
            else 0.0
        )

        return QAResult(
            result_id=_generate_id("res_qa"),
            challenge_id=challenge.challenge_id,
            component_id=challenge.component_id,
            team_a_score=team_a_score,
            team_b_score=team_b_score,
            team_a_evaluations=team_a_evaluations,
            team_b_evaluations=team_b_evaluations,
            questions_correct_both=questions_correct_both,
            questions_wrong_both=questions_wrong_both,
            category_scores={k: sum(v) / len(v) if v else 0.0 for k, v in category_scores.items()},
            documentation_gaps=documentation_gaps,
        )

    async def _evaluate_single_answer(
        self,
        question: Question,
        answer: TeamAnswer | None,
        ground_truth: str | None,
    ) -> QAEvaluation:
        """Evaluate a single answer against the question."""
        if not answer:
            return QAEvaluation(
                question_id=question.question_id,
                is_correct=False,
                is_complete=False,
                semantic_similarity=0.0,
                identified_gap=question.gap_indicator,
            )

        prompt = f"""
Evaluate this answer against the correct answer.

QUESTION: {question.text}
CORRECT ANSWER: {question.correct_answer}
PROVIDED ANSWER: {answer.answer}

Evaluate:
1. is_correct: Does the answer match the correct answer semantically?
2. is_complete: Does it cover all aspects of the correct answer?
3. semantic_similarity: Score from 0.0 to 1.0

Output as JSON with fields: is_correct, is_complete, semantic_similarity
"""
        response = await self._llm.generate(prompt)
        eval_data = json.loads(response)

        return QAEvaluation(
            question_id=question.question_id,
            is_correct=eval_data.get("is_correct", False),
            is_complete=eval_data.get("is_complete", False),
            semantic_similarity=eval_data.get("semantic_similarity", 0.0),
            identified_gap=question.gap_indicator if not eval_data.get("is_correct") else None,
        )

    def get_documentation_gaps(self, result: QAResult) -> list[dict]:
        """Extract documentation gaps from Q&A result."""
        return [
            {
                "area": gap.area,
                "severity": gap.severity.value,
                "recommendation": gap.recommendation,
                "description": gap.description,
                "evidence": gap.evidence,
            }
            for gap in result.documentation_gaps
        ]


# =============================================================================
# Masked Reconstruction Strategy
# =============================================================================


class MaskedReconstructor(VerificationStrategy[MaskedChallenge, ReconstructionResult]):
    """Masked code reconstruction strategy.

    Masks portions of code and asks teams to reconstruct what's hidden
    using only their documentation. Tests documentation specificity.

    This strategy tests:
        - Implementation detail documentation
        - Specific value documentation (constants, thresholds)
        - Business rule documentation
        - Algorithm documentation
    """

    MASK_PATTERNS = {
        MaskType.CONSTANTS: r"\b\d+\.?\d*\b",  # Numbers
        MaskType.STRINGS: r'["\'][^"\']*["\']',  # String literals
        MaskType.CONDITIONS: r"if\s+(.+?):",  # Condition expressions
        MaskType.RETURNS: r"return\s+(.+?)(?:\n|$)",  # Return expressions
        MaskType.FUNCTION_CALLS: r"(\w+)\s*\(",  # Function calls
        MaskType.LOOP_BOUNDS: r"(?:range|while)\s*\(([^)]+)\)",  # Loop bounds
    }

    def __init__(
        self,
        llm_client: LLMClient,
        mask_types: list[MaskType] | None = None,
        mask_ratio: float = 0.3,
    ) -> None:
        """Initialize the masked reconstructor.

        Args:
            llm_client: LLM client for evaluation
            mask_types: Types of elements to mask (default: all)
            mask_ratio: Ratio of matching elements to mask (0.0-1.0)
        """
        super().__init__(
            strategy_type=StrategyType.MASKED_RECONSTRUCTION,
            level=VerificationLevel.ACTIVE,
            description="Tests documentation specificity through code reconstruction",
        )
        self._llm = llm_client
        self._mask_types = mask_types or [
            MaskType.CONSTANTS,
            MaskType.CONDITIONS,
            MaskType.RETURNS,
        ]
        self._mask_ratio = mask_ratio

    async def generate_challenge(
        self,
        component_id: str,
        source_code: str,
        **kwargs,
    ) -> MaskedChallenge:
        """Generate a masked code challenge.

        Args:
            component_id: Component identifier
            source_code: Component source code
            **kwargs: Additional parameters (mask_types, mask_ratio)

        Returns:
            MaskedChallenge with masked code
        """
        mask_types = kwargs.get("mask_types", self._mask_types)
        mask_ratio = kwargs.get("mask_ratio", self._mask_ratio)

        masks = []
        for mask_type in mask_types:
            pattern = self.MASK_PATTERNS.get(mask_type)
            if pattern:
                for match in re.finditer(pattern, source_code):
                    if random.random() < mask_ratio:
                        masks.append(
                            Mask(
                                mask_id=_generate_id("mask"),
                                start=match.start(),
                                end=match.end(),
                                original=match.group(),
                                mask_type=mask_type,
                            )
                        )

        # Sort masks by position (descending) for replacement
        masks.sort(key=lambda m: m.start, reverse=True)

        # Apply masks
        masked_code = source_code
        for mask in masks:
            masked_code = masked_code[: mask.start] + mask.placeholder + masked_code[mask.end :]

        return MaskedChallenge(
            challenge_id=_generate_id("chal_mask"),
            component_id=component_id,
            original_code=source_code,
            masked_code=masked_code,
            masks=list(reversed(masks)),  # Restore original order
            mask_ratio=len(masks) / max(len(re.findall(r"\S+", source_code)), 1),
        )

    async def evaluate(
        self,
        challenge: MaskedChallenge,
        team_a_response: str,
        team_b_response: str,
        ground_truth: str | None = None,
    ) -> ReconstructionResult:
        """Evaluate team reconstructions.

        Args:
            challenge: The masked challenge
            team_a_response: JSON string of MaskReconstruction list from Team A
            team_b_response: JSON string of MaskReconstruction list from Team B
            ground_truth: Original code (already in challenge)

        Returns:
            ReconstructionResult with evaluation details
        """
        team_a_recons = [MaskReconstruction(**r) for r in json.loads(team_a_response)]
        team_b_recons = [MaskReconstruction(**r) for r in json.loads(team_b_response)]

        # Build lookups
        a_lookup = {r.mask_id: r for r in team_a_recons}
        b_lookup = {r.mask_id: r for r in team_b_recons}

        team_a_evaluations = []
        team_b_evaluations = []
        masks_correct_both = []
        masks_wrong_both = []
        mask_type_scores: dict[str, list[float]] = {}
        documentation_gaps = []

        for mask in challenge.masks:
            a_recon = a_lookup.get(mask.mask_id)
            b_recon = b_lookup.get(mask.mask_id)

            a_eval = self._evaluate_reconstruction(mask, a_recon)
            b_eval = self._evaluate_reconstruction(mask, b_recon)

            team_a_evaluations.append(a_eval)
            team_b_evaluations.append(b_eval)

            # Track shared results
            if a_eval.is_correct and b_eval.is_correct:
                masks_correct_both.append(mask.mask_id)
            elif not a_eval.is_correct and not b_eval.is_correct:
                masks_wrong_both.append(mask.mask_id)
                documentation_gaps.append(
                    DocumentationGap(
                        gap_id=_generate_id("gap"),
                        area=mask.mask_type.value,
                        description=f"Both teams failed to reconstruct {mask.mask_type.value}",
                        severity=Severity.HIGH,
                        recommendation=f"Add specific {mask.mask_type.value} documentation: {mask.original}",
                        evidence=f"Original value: {mask.original}",
                        affects_team_a=True,
                        affects_team_b=True,
                    )
                )

            # Track by mask type
            mt = mask.mask_type.value
            if mt not in mask_type_scores:
                mask_type_scores[mt] = []
            mask_type_scores[mt].append((a_eval.similarity_score + b_eval.similarity_score) / 2)

        # Calculate scores
        team_a_score = (
            sum(e.similarity_score for e in team_a_evaluations) / len(team_a_evaluations)
            if team_a_evaluations
            else 0.0
        )
        team_b_score = (
            sum(e.similarity_score for e in team_b_evaluations) / len(team_b_evaluations)
            if team_b_evaluations
            else 0.0
        )

        return ReconstructionResult(
            result_id=_generate_id("res_mask"),
            challenge_id=challenge.challenge_id,
            component_id=challenge.component_id,
            team_a_score=team_a_score,
            team_b_score=team_b_score,
            team_a_evaluations=team_a_evaluations,
            team_b_evaluations=team_b_evaluations,
            masks_correct_both=masks_correct_both,
            masks_wrong_both=masks_wrong_both,
            mask_type_scores={
                k: sum(v) / len(v) if v else 0.0 for k, v in mask_type_scores.items()
            },
            documentation_gaps=documentation_gaps,
        )

    def _evaluate_reconstruction(
        self,
        mask: Mask,
        reconstruction: MaskReconstruction | None,
    ) -> MaskEvaluation:
        """Evaluate a single reconstruction."""
        if not reconstruction:
            return MaskEvaluation(
                mask_id=mask.mask_id,
                is_correct=False,
                is_semantically_equivalent=False,
                similarity_score=0.0,
            )

        # Exact match
        is_correct = reconstruction.reconstructed_value.strip() == mask.original.strip()

        # Semantic equivalence (simplified - would use LLM in production)
        is_equivalent = is_correct or self._is_semantically_equivalent(
            mask.original, reconstruction.reconstructed_value
        )

        # Similarity score
        similarity = 1.0 if is_correct else (0.7 if is_equivalent else 0.0)

        return MaskEvaluation(
            mask_id=mask.mask_id,
            is_correct=is_correct,
            is_semantically_equivalent=is_equivalent,
            similarity_score=similarity,
        )

    def _is_semantically_equivalent(self, original: str, reconstructed: str) -> bool:
        """Check if two code fragments are semantically equivalent."""
        # Simplified check - in production, use AST comparison or LLM
        # Normalize whitespace and compare
        norm_orig = " ".join(original.split())
        norm_recon = " ".join(reconstructed.split())
        return norm_orig.lower() == norm_recon.lower()

    def get_documentation_gaps(self, result: ReconstructionResult) -> list[dict]:
        """Extract documentation gaps from reconstruction result."""
        return [
            {
                "area": gap.area,
                "severity": gap.severity.value,
                "recommendation": gap.recommendation,
                "description": gap.description,
                "evidence": gap.evidence,
            }
            for gap in result.documentation_gaps
        ]


# =============================================================================
# Scenario Walkthrough Strategy
# =============================================================================


class ScenarioWalker(VerificationStrategy[ScenarioChallenge, ScenarioResult]):
    """Scenario walkthrough strategy.

    Provides input scenarios and asks teams to trace execution
    using their documentation. Validates behavioral accuracy.

    This strategy tests:
        - Call graph accuracy
        - Error path documentation
        - Side effect documentation
        - State transition documentation
    """

    def __init__(
        self,
        llm_client: LLMClient,
        scenarios_per_component: int = 3,
        scenario_types: list[ScenarioType] | None = None,
    ) -> None:
        """Initialize the scenario walker.

        Args:
            llm_client: LLM client for scenario generation and evaluation
            scenarios_per_component: Number of scenarios per component
            scenario_types: Types of scenarios to generate
        """
        super().__init__(
            strategy_type=StrategyType.SCENARIO_WALKTHROUGH,
            level=VerificationLevel.ACTIVE,
            description="Tests behavioral accuracy through execution tracing",
        )
        self._llm = llm_client
        self._scenarios_per_component = scenarios_per_component
        self._scenario_types = scenario_types or [
            ScenarioType.HAPPY_PATH,
            ScenarioType.ERROR_PATH,
            ScenarioType.EDGE_CASE,
        ]

    async def generate_challenge(
        self,
        component_id: str,
        source_code: str,
        **kwargs,
    ) -> ScenarioChallenge:
        """Generate scenarios for walkthrough testing.

        Args:
            component_id: Component identifier
            source_code: Component source code
            **kwargs: Additional parameters

        Returns:
            ScenarioChallenge with generated scenarios
        """
        num_scenarios = kwargs.get("num_scenarios", self._scenarios_per_component)
        scenario_types = kwargs.get("scenario_types", self._scenario_types)

        prompt = f"""
Analyze this code and generate {num_scenarios} execution scenarios for testing documentation accuracy.

SOURCE CODE:
```python
{source_code}
```

Generate scenarios of types: {", ".join(t.value for t in scenario_types)}

For each scenario, provide:
- scenario_id: unique identifier (scen_001, etc.)
- scenario_type: one of {", ".join(t.value for t in scenario_types)}
- description: what the scenario tests (e.g., "User calls X with empty list")
- inputs: dictionary of input values/state
- expected_calls: list of function calls in order
- expected_output: what should be returned or exception raised
- expected_side_effects: list of side effects (logging, state changes, etc.)

Output as JSON array.
"""
        response = await self._llm.generate(prompt)
        scenarios_data = json.loads(response)

        scenarios = [
            Scenario(
                scenario_id=s.get("scenario_id", _generate_id("scen")),
                scenario_type=ScenarioType(s["scenario_type"]),
                description=s["description"],
                inputs=s.get("inputs", {}),
                expected_calls=s.get("expected_calls", []),
                expected_output=s.get("expected_output"),
                expected_side_effects=s.get("expected_side_effects", []),
            )
            for s in scenarios_data
        ]

        return ScenarioChallenge(
            challenge_id=_generate_id("chal_scen"),
            component_id=component_id,
            scenarios=scenarios,
        )

    async def evaluate(
        self,
        challenge: ScenarioChallenge,
        team_a_response: str,
        team_b_response: str,
        ground_truth: str | None = None,
    ) -> ScenarioResult:
        """Evaluate team execution traces.

        Args:
            challenge: The scenario challenge
            team_a_response: JSON string of ExecutionTrace list from Team A
            team_b_response: JSON string of ExecutionTrace list from Team B
            ground_truth: Optional additional ground truth

        Returns:
            ScenarioResult with evaluation details
        """
        team_a_traces = [ExecutionTrace(**t) for t in json.loads(team_a_response)]
        team_b_traces = [ExecutionTrace(**t) for t in json.loads(team_b_response)]

        a_lookup = {t.scenario_id: t for t in team_a_traces}
        b_lookup = {t.scenario_id: t for t in team_b_traces}

        team_a_evaluations = []
        team_b_evaluations = []
        scenarios_correct_both = []
        scenarios_wrong_both = []
        commonly_missed_calls = []
        commonly_missed_side_effects = []
        documentation_gaps = []

        for scenario in challenge.scenarios:
            a_trace = a_lookup.get(scenario.scenario_id)
            b_trace = b_lookup.get(scenario.scenario_id)

            a_eval = self._evaluate_trace(scenario, a_trace)
            b_eval = self._evaluate_trace(scenario, b_trace)

            team_a_evaluations.append(a_eval)
            team_b_evaluations.append(b_eval)

            # Track shared results
            if a_eval.overall_score >= 0.8 and b_eval.overall_score >= 0.8:
                scenarios_correct_both.append(scenario.scenario_id)
            elif a_eval.overall_score < 0.5 and b_eval.overall_score < 0.5:
                scenarios_wrong_both.append(scenario.scenario_id)

            # Track commonly missed elements
            shared_missed_calls = set(a_eval.missed_calls) & set(b_eval.missed_calls)
            commonly_missed_calls.extend(shared_missed_calls)

            shared_missed_effects = set(a_eval.missed_side_effects) & set(
                b_eval.missed_side_effects
            )
            commonly_missed_side_effects.extend(shared_missed_effects)

            # Create gaps for shared failures
            if shared_missed_calls or shared_missed_effects:
                documentation_gaps.append(
                    DocumentationGap(
                        gap_id=_generate_id("gap"),
                        area="execution_behavior",
                        description=f"Both teams missed elements in scenario: {scenario.description}",
                        severity=Severity.HIGH,
                        recommendation=f"Document missed calls: {shared_missed_calls}, effects: {shared_missed_effects}",
                        affects_team_a=True,
                        affects_team_b=True,
                    )
                )

        # Calculate scores
        team_a_score = (
            sum(e.overall_score for e in team_a_evaluations) / len(team_a_evaluations)
            if team_a_evaluations
            else 0.0
        )
        team_b_score = (
            sum(e.overall_score for e in team_b_evaluations) / len(team_b_evaluations)
            if team_b_evaluations
            else 0.0
        )

        return ScenarioResult(
            result_id=_generate_id("res_scen"),
            challenge_id=challenge.challenge_id,
            component_id=challenge.component_id,
            team_a_score=team_a_score,
            team_b_score=team_b_score,
            team_a_evaluations=team_a_evaluations,
            team_b_evaluations=team_b_evaluations,
            scenarios_correct_both=scenarios_correct_both,
            scenarios_wrong_both=scenarios_wrong_both,
            commonly_missed_calls=list(set(commonly_missed_calls)),
            commonly_missed_side_effects=list(set(commonly_missed_side_effects)),
            documentation_gaps=documentation_gaps,
        )

    def _evaluate_trace(
        self,
        scenario: Scenario,
        trace: ExecutionTrace | None,
    ) -> ScenarioEvaluation:
        """Evaluate a single execution trace."""
        if not trace:
            return ScenarioEvaluation(
                scenario_id=scenario.scenario_id,
                call_sequence_score=0.0,
                output_correct=False,
                side_effects_score=0.0,
                overall_score=0.0,
                missed_calls=scenario.expected_calls,
                missed_side_effects=scenario.expected_side_effects,
            )

        # Evaluate call sequence (order matters)
        call_score = self._sequence_similarity(scenario.expected_calls, trace.predicted_calls)

        # Evaluate output
        output_correct = self._outputs_match(scenario.expected_output, trace.predicted_output)

        # Evaluate side effects
        expected_effects = set(scenario.expected_side_effects)
        predicted_effects = set(trace.predicted_side_effects)
        effects_score = (
            len(expected_effects & predicted_effects) / len(expected_effects)
            if expected_effects
            else 1.0
        )

        # Calculate overall score
        overall = (call_score * 0.4) + (0.3 if output_correct else 0.0) + (effects_score * 0.3)

        # Identify missed elements
        missed_calls = [c for c in scenario.expected_calls if c not in trace.predicted_calls]
        missed_effects = list(expected_effects - predicted_effects)

        return ScenarioEvaluation(
            scenario_id=scenario.scenario_id,
            call_sequence_score=call_score,
            output_correct=output_correct,
            side_effects_score=effects_score,
            overall_score=overall,
            missed_calls=missed_calls,
            missed_side_effects=missed_effects,
        )

    def _sequence_similarity(self, expected: list[str], actual: list[str]) -> float:
        """Calculate similarity between two call sequences."""
        if not expected:
            return 1.0 if not actual else 0.0
        if not actual:
            return 0.0

        # Simple LCS-based similarity
        lcs_length = self._lcs_length(expected, actual)
        return lcs_length / max(len(expected), len(actual))

    def _lcs_length(self, seq1: list[str], seq2: list[str]) -> int:
        """Calculate longest common subsequence length."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    def _outputs_match(self, expected: str | None, actual: str | None) -> bool:
        """Check if outputs match."""
        if expected is None and actual is None:
            return True
        if expected is None or actual is None:
            return False
        return expected.strip().lower() == actual.strip().lower()

    def get_documentation_gaps(self, result: ScenarioResult) -> list[dict]:
        """Extract documentation gaps from scenario result."""
        return [
            {
                "area": gap.area,
                "severity": gap.severity.value,
                "recommendation": gap.recommendation,
                "description": gap.description,
            }
            for gap in result.documentation_gaps
        ]


# =============================================================================
# Mutation Detection Strategy
# =============================================================================


class MutationDetector(VerificationStrategy[MutationChallenge, MutationResult]):
    """Mutation detection strategy.

    Introduces subtle bugs and asks teams if their documentation
    would help a reviewer catch the bug. Tests documentation precision.

    This strategy tests:
        - Boundary condition documentation
        - Logic documentation precision
        - Variable purpose documentation
        - Sequence documentation
    """

    def __init__(
        self,
        llm_client: LLMClient,
        mutations_per_component: int = 5,
        mutation_types: list[MutationType] | None = None,
    ) -> None:
        """Initialize the mutation detector.

        Args:
            llm_client: LLM client for mutation generation and evaluation
            mutations_per_component: Number of mutations per component
            mutation_types: Types of mutations to generate
        """
        super().__init__(
            strategy_type=StrategyType.MUTATION_DETECTION,
            level=VerificationLevel.BEHAVIORAL,
            description="Tests documentation precision through mutation detection",
        )
        self._llm = llm_client
        self._mutations_per_component = mutations_per_component
        self._mutation_types = mutation_types or [
            MutationType.BOUNDARY,
            MutationType.OFF_BY_ONE,
            MutationType.NULL_HANDLING,
        ]

    async def generate_challenge(
        self,
        component_id: str,
        source_code: str,
        **kwargs,
    ) -> MutationChallenge:
        """Generate mutations for detection testing.

        Args:
            component_id: Component identifier
            source_code: Component source code
            **kwargs: Additional parameters

        Returns:
            MutationChallenge with generated mutations
        """
        num_mutations = kwargs.get("num_mutations", self._mutations_per_component)
        mutation_types = kwargs.get("mutation_types", self._mutation_types)

        prompt = f"""
Analyze this code and generate {num_mutations} subtle mutations that would introduce bugs.

SOURCE CODE:
```python
{source_code}
```

Generate mutations of types: {", ".join(t.value for t in mutation_types)}

For each mutation, provide:
- mutation_id: unique identifier (mut_001, etc.)
- mutation_type: one of {", ".join(t.value for t in mutation_types)}
- original_code: the original code snippet
- mutated_code: the mutated code snippet
- description: what changed
- line_number: line where mutation occurs
- detection_hint: what documentation would help catch this

Output as JSON array.
"""
        response = await self._llm.generate(prompt)
        mutations_data = json.loads(response)

        mutations = [
            Mutation(
                mutation_id=m.get("mutation_id", _generate_id("mut")),
                mutation_type=MutationType(m["mutation_type"]),
                original_code=m["original_code"],
                mutated_code=m["mutated_code"],
                description=m["description"],
                line_number=m.get("line_number", 1),
                detection_hint=m.get("detection_hint", ""),
            )
            for m in mutations_data
        ]

        return MutationChallenge(
            challenge_id=_generate_id("chal_mut"),
            component_id=component_id,
            mutations=mutations,
            full_mutated_code=source_code,  # Would apply mutations in production
        )

    async def evaluate(
        self,
        challenge: MutationChallenge,
        team_a_response: str,
        team_b_response: str,
        ground_truth: str | None = None,
    ) -> MutationResult:
        """Evaluate team mutation assessments.

        Args:
            challenge: The mutation challenge
            team_a_response: JSON string of MutationAssessment list from Team A
            team_b_response: JSON string of MutationAssessment list from Team B
            ground_truth: Optional additional ground truth

        Returns:
            MutationResult with evaluation details
        """
        team_a_assessments = [MutationAssessment(**a) for a in json.loads(team_a_response)]
        team_b_assessments = [MutationAssessment(**a) for a in json.loads(team_b_response)]

        a_lookup = {a.mutation_id: a for a in team_a_assessments}
        b_lookup = {a.mutation_id: a for a in team_b_assessments}

        team_a_evaluations = []
        team_b_evaluations = []
        detectable_mutations = []
        undetectable_mutations = []
        mutation_type_scores: dict[str, list[float]] = {}
        documentation_gaps = []

        for mutation in challenge.mutations:
            a_assessment = a_lookup.get(mutation.mutation_id)
            b_assessment = b_lookup.get(mutation.mutation_id)

            a_eval = await self._evaluate_assessment(mutation, a_assessment)
            b_eval = await self._evaluate_assessment(mutation, b_assessment)

            team_a_evaluations.append(a_eval)
            team_b_evaluations.append(b_eval)

            # Track detectability
            if a_eval.docs_actually_help and b_eval.docs_actually_help:
                detectable_mutations.append(mutation.mutation_id)
            elif not a_eval.docs_actually_help and not b_eval.docs_actually_help:
                undetectable_mutations.append(mutation.mutation_id)
                documentation_gaps.append(
                    DocumentationGap(
                        gap_id=_generate_id("gap"),
                        area=f"boundary_precision_{mutation.mutation_type.value}",
                        description=f"Neither team's docs would catch: {mutation.description}",
                        severity=Severity.HIGH,
                        recommendation=mutation.detection_hint
                        or f"Add precise {mutation.mutation_type.value} documentation",
                        affects_team_a=True,
                        affects_team_b=True,
                    )
                )

            # Track by mutation type
            mt = mutation.mutation_type.value
            if mt not in mutation_type_scores:
                mutation_type_scores[mt] = []
            score = 0.5 * (1.0 if a_eval.docs_actually_help else 0.0) + 0.5 * (
                1.0 if b_eval.docs_actually_help else 0.0
            )
            mutation_type_scores[mt].append(score)

        # Calculate scores
        team_a_score = (
            sum(1.0 if e.docs_actually_help else 0.0 for e in team_a_evaluations)
            / len(team_a_evaluations)
            if team_a_evaluations
            else 0.0
        )
        team_b_score = (
            sum(1.0 if e.docs_actually_help else 0.0 for e in team_b_evaluations)
            / len(team_b_evaluations)
            if team_b_evaluations
            else 0.0
        )

        return MutationResult(
            result_id=_generate_id("res_mut"),
            challenge_id=challenge.challenge_id,
            component_id=challenge.component_id,
            team_a_score=team_a_score,
            team_b_score=team_b_score,
            team_a_evaluations=team_a_evaluations,
            team_b_evaluations=team_b_evaluations,
            detectable_mutations=detectable_mutations,
            undetectable_mutations=undetectable_mutations,
            mutation_type_scores={
                k: sum(v) / len(v) if v else 0.0 for k, v in mutation_type_scores.items()
            },
            documentation_gaps=documentation_gaps,
        )

    async def _evaluate_assessment(
        self,
        mutation: Mutation,
        assessment: MutationAssessment | None,
    ) -> MutationEvaluation:
        """Evaluate a single mutation assessment."""
        if not assessment:
            return MutationEvaluation(
                mutation_id=mutation.mutation_id,
                assessment_accurate=False,
                docs_actually_help=False,
                precision_gap=mutation.detection_hint,
            )

        # Use LLM to evaluate if the cited documentation would actually help
        prompt = f"""
Evaluate whether this documentation would help detect the mutation.

MUTATION: {mutation.description}
ORIGINAL: {mutation.original_code}
MUTATED: {mutation.mutated_code}

TEAM'S CLAIM: Would detect = {assessment.would_detect}
CITED DOCUMENTATION: {assessment.relevant_documentation}

Would the cited documentation actually help a reviewer detect this bug?
Output JSON: {{"docs_actually_help": true/false, "explanation": "..."}}
"""
        response = await self._llm.generate(prompt)
        eval_data = json.loads(response)

        return MutationEvaluation(
            mutation_id=mutation.mutation_id,
            assessment_accurate=assessment.would_detect
            == eval_data.get("docs_actually_help", False),
            docs_actually_help=eval_data.get("docs_actually_help", False),
            precision_gap=None if eval_data.get("docs_actually_help") else mutation.detection_hint,
        )

    def get_documentation_gaps(self, result: MutationResult) -> list[dict]:
        """Extract documentation gaps from mutation result."""
        return [
            {
                "area": gap.area,
                "severity": gap.severity.value,
                "recommendation": gap.recommendation,
                "description": gap.description,
            }
            for gap in result.documentation_gaps
        ]


# =============================================================================
# Impact Analysis Strategy
# =============================================================================


class ImpactAnalyzer(VerificationStrategy[ImpactChallenge, ImpactResult]):
    """Impact analysis strategy.

    Asks teams to predict what would break if a component changed.
    Tests dependency documentation accuracy.

    This strategy tests:
        - Dependency documentation
        - Caller/callee relationships
        - Coupling documentation
        - Interface contracts
    """

    def __init__(
        self,
        llm_client: LLMClient,
        change_types: list[ChangeType] | None = None,
    ) -> None:
        """Initialize the impact analyzer.

        Args:
            llm_client: LLM client for challenge generation
            change_types: Types of changes to test
        """
        super().__init__(
            strategy_type=StrategyType.IMPACT_ANALYSIS,
            level=VerificationLevel.BEHAVIORAL,
            description="Tests dependency documentation through impact prediction",
        )
        self._llm = llm_client
        self._change_types = change_types or list(ChangeType)

    async def generate_challenge(
        self,
        component_id: str,
        source_code: str,
        **kwargs,
    ) -> ImpactChallenge:
        """Generate an impact analysis challenge.

        Args:
            component_id: Component identifier
            source_code: Component source code
            **kwargs: Additional parameters (change_type, actual_impacted)

        Returns:
            ImpactChallenge with change scenario
        """
        change_type = kwargs.get("change_type", random.choice(self._change_types))
        actual_impacted = kwargs.get("actual_impacted", [])

        # Generate change description
        component_name = component_id.split(".")[-1]
        change_descriptions = {
            ChangeType.SIGNATURE: f"Change {component_name} signature: add required parameter 'options: dict'",
            ChangeType.RETURN_TYPE: f"Change {component_name} return type from current to Optional[...]",
            ChangeType.BEHAVIOR: f"Change {component_name} to return None on error instead of raising exception",
            ChangeType.REMOVAL: f"Delete {component_name} entirely",
            ChangeType.RENAME: f"Rename {component_name} to {component_name}_v2",
        }

        return ImpactChallenge(
            challenge_id=_generate_id("chal_impact"),
            component_id=component_id,
            change_type=change_type,
            change_description=change_descriptions[change_type],
            actual_impacted=actual_impacted,
        )

    async def evaluate(
        self,
        challenge: ImpactChallenge,
        team_a_response: str,
        team_b_response: str,
        ground_truth: str | None = None,
    ) -> ImpactResult:
        """Evaluate team impact predictions.

        Args:
            challenge: The impact challenge
            team_a_response: JSON string of ImpactPrediction from Team A
            team_b_response: JSON string of ImpactPrediction from Team B
            ground_truth: Optional override for actual_impacted

        Returns:
            ImpactResult with evaluation details
        """
        team_a_pred = ImpactPrediction(**json.loads(team_a_response))
        team_b_pred = ImpactPrediction(**json.loads(team_b_response))

        actual = set(ground_truth.split(",") if ground_truth else challenge.actual_impacted)
        pred_a = set(team_a_pred.predicted_impacted)
        pred_b = set(team_b_pred.predicted_impacted)

        # Calculate precision/recall for each team
        a_tp = pred_a & actual
        a_fp = pred_a - actual
        a_fn = actual - pred_a

        b_tp = pred_b & actual
        b_fp = pred_b - actual
        b_fn = actual - pred_b

        a_precision = len(a_tp) / len(pred_a) if pred_a else 0.0
        a_recall = len(a_tp) / len(actual) if actual else 1.0
        b_precision = len(b_tp) / len(pred_b) if pred_b else 0.0
        b_recall = len(b_tp) / len(actual) if actual else 1.0

        # Create documentation gaps for missed impacts
        documentation_gaps = []
        missed_by_both = a_fn & b_fn
        if missed_by_both:
            documentation_gaps.append(
                DocumentationGap(
                    gap_id=_generate_id("gap"),
                    area="dependency_tracking",
                    description=f"Both teams missed impacted components: {missed_by_both}",
                    severity=Severity.CRITICAL,
                    recommendation=f"Add dependency documentation for: {', '.join(missed_by_both)}",
                    affects_team_a=True,
                    affects_team_b=True,
                )
            )

        # Use F1 score for team scores
        team_a_score = (
            2 * a_precision * a_recall / (a_precision + a_recall)
            if (a_precision + a_recall) > 0
            else 0.0
        )
        team_b_score = (
            2 * b_precision * b_recall / (b_precision + b_recall)
            if (b_precision + b_recall) > 0
            else 0.0
        )

        return ImpactResult(
            result_id=_generate_id("res_impact"),
            challenge_id=challenge.challenge_id,
            component_id=challenge.component_id,
            team_a_score=team_a_score,
            team_b_score=team_b_score,
            team_a_prediction=team_a_pred,
            team_b_prediction=team_b_pred,
            team_a_true_positives=list(a_tp),
            team_a_false_positives=list(a_fp),
            team_a_false_negatives=list(a_fn),
            team_b_true_positives=list(b_tp),
            team_b_false_positives=list(b_fp),
            team_b_false_negatives=list(b_fn),
            team_a_precision=a_precision,
            team_a_recall=a_recall,
            team_b_precision=b_precision,
            team_b_recall=b_recall,
            documentation_gaps=documentation_gaps,
        )

    def get_documentation_gaps(self, result: ImpactResult) -> list[dict]:
        """Extract documentation gaps from impact result."""
        return [
            {
                "area": gap.area,
                "severity": gap.severity.value,
                "recommendation": gap.recommendation,
                "description": gap.description,
            }
            for gap in result.documentation_gaps
        ]


# =============================================================================
# Adversarial Review Strategy
# =============================================================================


class AdversarialReviewer(VerificationStrategy[AdversarialChallenge, AdversarialResult]):
    """Adversarial review strategy.

    One team reviews the other team's documentation against code and
    tries to find errors. Agent C validates findings.

    This strategy tests:
        - Cross-validation accuracy
        - Documentation precision
        - Coverage completeness
        - Consistency with code
    """

    def __init__(
        self,
        llm_client: LLMClient,
        max_findings_per_component: int = 10,
    ) -> None:
        """Initialize the adversarial reviewer.

        Args:
            llm_client: LLM client for review and validation
            max_findings_per_component: Maximum findings to report per component
        """
        super().__init__(
            strategy_type=StrategyType.ADVERSARIAL_REVIEW,
            level=VerificationLevel.GENERATIVE,
            description="Tests documentation through cross-team adversarial review",
        )
        self._llm = llm_client
        self._max_findings = max_findings_per_component

    async def generate_challenge(
        self,
        component_id: str,
        source_code: str,
        **kwargs,
    ) -> AdversarialChallenge:
        """Generate an adversarial review challenge.

        Args:
            component_id: Component identifier
            source_code: Component source code
            **kwargs: Additional parameters (team_a_documentation, team_b_documentation)

        Returns:
            AdversarialChallenge for cross-team review
        """
        team_a_docs = kwargs.get("team_a_documentation", "")
        team_b_docs = kwargs.get("team_b_documentation", "")

        return AdversarialChallenge(
            challenge_id=_generate_id("chal_adv"),
            component_id=component_id,
            team_a_documentation=team_a_docs,
            team_b_documentation=team_b_docs,
            source_code=source_code,
            max_findings=self._max_findings,
        )

    async def evaluate(
        self,
        challenge: AdversarialChallenge,
        team_a_response: str,
        team_b_response: str,
        ground_truth: str | None = None,
    ) -> AdversarialResult:
        """Evaluate adversarial review findings.

        Args:
            challenge: The adversarial challenge
            team_a_response: JSON string of findings A found in B's docs
            team_b_response: JSON string of findings B found in A's docs
            ground_truth: Source code for validation

        Returns:
            AdversarialResult with validated findings
        """
        findings_by_a = [AdversarialFinding(**f) for f in json.loads(team_a_response)]
        findings_by_b = [AdversarialFinding(**f) for f in json.loads(team_b_response)]

        # Validate each finding against source code
        validated_findings = []
        false_findings = []

        all_findings = findings_by_a + findings_by_b
        for finding in all_findings:
            is_valid = await self._validate_finding(finding, challenge.source_code)
            if is_valid:
                validated_findings.append(finding.finding_id)
            else:
                false_findings.append(finding.finding_id)

        # Create documentation gaps from validated findings
        documentation_gaps = []
        for finding in all_findings:
            if finding.finding_id in validated_findings:
                documentation_gaps.append(
                    DocumentationGap(
                        gap_id=_generate_id("gap"),
                        area=finding.issue_type,
                        description=finding.description,
                        severity=finding.severity,
                        recommendation=f"Fix {finding.issue_type} at {finding.location}",
                        evidence=finding.code_evidence,
                        affects_team_a=finding.reviewed_team == "A",
                        affects_team_b=finding.reviewed_team == "B",
                    )
                )

        # Calculate scores based on findings accuracy
        total_findings_a = len(findings_by_a)
        valid_findings_a = sum(1 for f in findings_by_a if f.finding_id in validated_findings)
        total_findings_b = len(findings_by_b)
        valid_findings_b = sum(1 for f in findings_by_b if f.finding_id in validated_findings)

        # Score is precision of findings (valid / total)
        team_a_score = valid_findings_a / total_findings_a if total_findings_a > 0 else 0.0
        team_b_score = valid_findings_b / total_findings_b if total_findings_b > 0 else 0.0

        return AdversarialResult(
            result_id=_generate_id("res_adv"),
            challenge_id=challenge.challenge_id,
            component_id=challenge.component_id,
            team_a_score=team_a_score,
            team_b_score=team_b_score,
            findings_by_a=findings_by_a,
            findings_by_b=findings_by_b,
            validated_findings=validated_findings,
            false_findings=false_findings,
            documentation_gaps=documentation_gaps,
        )

    async def _validate_finding(
        self,
        finding: AdversarialFinding,
        source_code: str,
    ) -> bool:
        """Validate a finding against source code.

        Args:
            finding: The finding to validate
            source_code: Source code for verification

        Returns:
            True if finding is valid
        """
        prompt = f"""
Validate whether this finding about documentation is correct based on the source code.

FINDING:
- Issue Type: {finding.issue_type}
- Location: {finding.location}
- Description: {finding.description}
- Code Evidence: {finding.code_evidence}

SOURCE CODE:
```python
{source_code}
```

Is this finding valid? Does the documentation actually have this issue based on the code?
Output JSON: {{"is_valid": true/false, "reason": "..."}}
"""
        response = await self._llm.generate(prompt)
        result = json.loads(response)
        return result.get("is_valid", False)

    def get_documentation_gaps(self, result: AdversarialResult) -> list[dict]:
        """Extract documentation gaps from adversarial result."""
        return [
            {
                "area": gap.area,
                "severity": gap.severity.value,
                "recommendation": gap.recommendation,
                "description": gap.description,
                "evidence": gap.evidence,
            }
            for gap in result.documentation_gaps
        ]


# =============================================================================
# Test Generation Validation Strategy
# =============================================================================


class TestGenerationValidator(VerificationStrategy[TestGenerationChallenge, TestValidationResult]):
    """Test generation validation strategy.

    Generates unit tests from documentation and runs them against actual code.
    Test failures reveal documentation inaccuracies.

    This strategy tests:
        - Behavioral accuracy
        - Return value documentation
        - Exception documentation
        - Edge case coverage
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tests_per_team: int = 10,
    ) -> None:
        """Initialize the test generation validator.

        Args:
            llm_client: LLM client for test generation
            tests_per_team: Number of tests to generate per team
        """
        super().__init__(
            strategy_type=StrategyType.TEST_GENERATION,
            level=VerificationLevel.GENERATIVE,
            description="Validates documentation by generating and running tests",
        )
        self._llm = llm_client
        self._tests_per_team = tests_per_team

    async def generate_challenge(
        self,
        component_id: str,
        source_code: str,
        **kwargs,
    ) -> TestGenerationChallenge:
        """Generate a test generation challenge.

        Args:
            component_id: Component identifier
            source_code: Component source code
            **kwargs: Additional parameters (team_a_documentation, team_b_documentation)

        Returns:
            TestGenerationChallenge for test generation
        """
        team_a_docs = kwargs.get("team_a_documentation", "")
        team_b_docs = kwargs.get("team_b_documentation", "")
        tests_per_team = kwargs.get("tests_per_team", self._tests_per_team)

        return TestGenerationChallenge(
            challenge_id=_generate_id("chal_test"),
            component_id=component_id,
            team_a_documentation=team_a_docs,
            team_b_documentation=team_b_docs,
            source_code=source_code,
            tests_per_team=tests_per_team,
        )

    async def evaluate(
        self,
        challenge: TestGenerationChallenge,
        team_a_response: str,
        team_b_response: str,
        ground_truth: str | None = None,
    ) -> TestValidationResult:
        """Evaluate by generating tests and checking results.

        Args:
            challenge: The test generation challenge
            team_a_response: JSON string of tests generated from A's docs
            team_b_response: JSON string of tests generated from B's docs
            ground_truth: Source code for test execution

        Returns:
            TestValidationResult with test execution results
        """
        team_a_tests = [GeneratedTest(**t) for t in json.loads(team_a_response)]
        team_b_tests = [GeneratedTest(**t) for t in json.loads(team_b_response)]

        source_code = ground_truth or challenge.source_code

        # Execute tests against source code
        team_a_executions = await self._execute_tests(team_a_tests, source_code)
        team_b_executions = await self._execute_tests(team_b_tests, source_code)

        # Identify documentation errors from failures
        documentation_errors = []
        documentation_gaps = []

        for execution in team_a_executions + team_b_executions:
            if not execution.passed and execution.failure_reason:
                documentation_errors.append(execution.failure_reason)
                documentation_gaps.append(
                    DocumentationGap(
                        gap_id=_generate_id("gap"),
                        area="behavioral_accuracy",
                        description=f"Test failure: {execution.failure_reason}",
                        severity=Severity.HIGH,
                        recommendation="Update documentation to match actual behavior",
                        evidence=execution.error_message or "",
                    )
                )

        # Calculate pass rates
        team_a_passed = sum(1 for e in team_a_executions if e.passed)
        team_b_passed = sum(1 for e in team_b_executions if e.passed)

        team_a_score = team_a_passed / len(team_a_executions) if team_a_executions else 0.0
        team_b_score = team_b_passed / len(team_b_executions) if team_b_executions else 0.0

        return TestValidationResult(
            result_id=_generate_id("res_test"),
            challenge_id=challenge.challenge_id,
            component_id=challenge.component_id,
            team_a_score=team_a_score,
            team_b_score=team_b_score,
            team_a_tests=team_a_tests,
            team_b_tests=team_b_tests,
            team_a_executions=team_a_executions,
            team_b_executions=team_b_executions,
            documentation_errors=documentation_errors,
            documentation_gaps=documentation_gaps,
        )

    async def generate_tests_from_documentation(
        self,
        documentation: str,
        team: str,
        num_tests: int,
    ) -> list[GeneratedTest]:
        """Generate tests from documentation.

        Args:
            documentation: Team's documentation
            team: Team identifier (A or B)
            num_tests: Number of tests to generate

        Returns:
            List of generated tests
        """
        prompt = f"""
Based on this documentation, generate {num_tests} comprehensive unit tests.

DOCUMENTATION:
{documentation}

Generate tests for:
1. Normal behavior (happy path)
2. Each documented exception
3. Edge cases mentioned
4. Boundary conditions
5. Return value validation

For each test, provide:
- test_id: unique identifier
- test_name: descriptive test name (e.g., test_process_data_empty_list)
- test_code: pytest-compatible Python test code
- tests_aspect: what aspect is being tested

Output as JSON array.
"""
        response = await self._llm.generate(prompt)
        tests_data = json.loads(response)

        return [
            GeneratedTest(
                test_id=t.get("test_id", _generate_id("test")),
                test_name=t["test_name"],
                test_code=t["test_code"],
                tests_aspect=t.get("tests_aspect", "behavior"),
                from_team=team,
            )
            for t in tests_data
        ]

    async def _execute_tests(
        self,
        tests: list[GeneratedTest],
        source_code: str,
    ) -> list[TestExecution]:
        """Execute generated tests against source code.

        Note: In production, this would actually run pytest.
        For now, uses LLM to simulate test execution.

        Args:
            tests: Tests to execute
            source_code: Source code to test against

        Returns:
            List of test execution results
        """
        executions = []

        for test in tests:
            prompt = f"""
Analyze whether this test would pass or fail against the source code.

SOURCE CODE:
```python
{source_code}
```

TEST CODE:
```python
{test.test_code}
```

Determine:
1. Would this test pass or fail?
2. If fail, what's the error message?
3. If fail, is it because the documentation was wrong?

Output JSON: {{
    "passed": true/false,
    "error_message": "..." or null,
    "failure_reason": "..." or null
}}
"""
            response = await self._llm.generate(prompt)
            result = json.loads(response)

            executions.append(
                TestExecution(
                    test_id=test.test_id,
                    passed=result.get("passed", False),
                    error_message=result.get("error_message"),
                    failure_reason=result.get("failure_reason"),
                )
            )

        return executions

    def get_documentation_gaps(self, result: TestValidationResult) -> list[dict]:
        """Extract documentation gaps from test validation result."""
        return [
            {
                "area": gap.area,
                "severity": gap.severity.value,
                "recommendation": gap.recommendation,
                "description": gap.description,
            }
            for gap in result.documentation_gaps
        ]


# =============================================================================
# Code Reconstruction Strategy
# =============================================================================


class CodeReconstructor(
    VerificationStrategy[CodeReconstructionChallenge, CodeReconstructionResult]
):
    """Code reconstruction strategy.

    Given only documentation, reconstructs functionally equivalent code.
    Gaps in reconstruction reveal documentation incompleteness.

    This strategy tests:
        - Documentation completeness
        - Algorithm specification
        - Business rule documentation
        - Implementation detail coverage
    """

    def __init__(
        self,
        llm_client: LLMClient,
    ) -> None:
        """Initialize the code reconstructor.

        Args:
            llm_client: LLM client for code reconstruction
        """
        super().__init__(
            strategy_type=StrategyType.CODE_RECONSTRUCTION,
            level=VerificationLevel.GENERATIVE,
            description="Tests documentation completeness through code reconstruction",
        )
        self._llm = llm_client

    async def generate_challenge(
        self,
        component_id: str,
        source_code: str,
        **kwargs,
    ) -> CodeReconstructionChallenge:
        """Generate a code reconstruction challenge.

        Args:
            component_id: Component identifier
            source_code: Component source code (ground truth)
            **kwargs: Additional parameters (team_a_documentation, team_b_documentation)

        Returns:
            CodeReconstructionChallenge for reconstruction
        """
        team_a_docs = kwargs.get("team_a_documentation", "")
        team_b_docs = kwargs.get("team_b_documentation", "")

        # Extract function signature from source code
        signature = self._extract_signature(source_code)

        return CodeReconstructionChallenge(
            challenge_id=_generate_id("chal_recon"),
            component_id=component_id,
            team_a_documentation=team_a_docs,
            team_b_documentation=team_b_docs,
            original_code=source_code,
            function_signature=signature,
        )

    def _extract_signature(self, source_code: str) -> str:
        """Extract function/method signature from source code.

        Args:
            source_code: Source code to extract from

        Returns:
            Function signature string
        """
        # Simple regex to extract def line
        match = re.search(r"^(def\s+\w+\s*\([^)]*\).*?:)", source_code, re.MULTILINE)
        if match:
            return match.group(1)
        return "def unknown():"

    async def evaluate(
        self,
        challenge: CodeReconstructionChallenge,
        team_a_response: str,
        team_b_response: str,
        ground_truth: str | None = None,
    ) -> CodeReconstructionResult:
        """Evaluate code reconstructions.

        Args:
            challenge: The reconstruction challenge
            team_a_response: JSON string of ReconstructedCode from Team A
            team_b_response: JSON string of ReconstructedCode from Team B
            ground_truth: Original code for comparison

        Returns:
            CodeReconstructionResult with evaluation
        """
        team_a_recon = ReconstructedCode(**json.loads(team_a_response))
        team_b_recon = ReconstructedCode(**json.loads(team_b_response))

        original = ground_truth or challenge.original_code

        # Evaluate functional equivalence
        team_a_equivalence = await self._evaluate_equivalence(original, team_a_recon.code)
        team_b_equivalence = await self._evaluate_equivalence(original, team_b_recon.code)

        # Identify missing details from both
        missing_from_both = list(set(team_a_recon.unknown_areas) & set(team_b_recon.unknown_areas))

        # Create documentation gaps for shared unknowns
        documentation_gaps = []
        for unknown in missing_from_both:
            documentation_gaps.append(
                DocumentationGap(
                    gap_id=_generate_id("gap"),
                    area="implementation_completeness",
                    description=f"Neither team could reconstruct: {unknown}",
                    severity=Severity.HIGH,
                    recommendation=f"Add documentation for: {unknown}",
                    affects_team_a=True,
                    affects_team_b=True,
                )
            )

        return CodeReconstructionResult(
            result_id=_generate_id("res_recon"),
            challenge_id=challenge.challenge_id,
            component_id=challenge.component_id,
            team_a_score=team_a_equivalence,
            team_b_score=team_b_equivalence,
            team_a_reconstruction=team_a_recon,
            team_b_reconstruction=team_b_recon,
            team_a_functional_equivalence=team_a_equivalence,
            team_b_functional_equivalence=team_b_equivalence,
            missing_from_both=missing_from_both,
            documentation_gaps=documentation_gaps,
        )

    async def reconstruct_from_documentation(
        self,
        documentation: str,
        signature: str,
        team: str,
    ) -> ReconstructedCode:
        """Reconstruct code from documentation.

        Args:
            documentation: Team's documentation
            signature: Function signature to implement
            team: Team identifier (A or B)

        Returns:
            ReconstructedCode with implementation and notes
        """
        prompt = f"""
Based solely on this documentation, reconstruct the implementation.

FUNCTION SIGNATURE:
{signature}

DOCUMENTATION:
{documentation}

Reconstruct the implementation. Mark any areas where the documentation
is insufficient with comments like "# UNKNOWN: <what's missing>".

Track:
1. The reconstructed code
2. Areas where documentation was insufficient
3. Assumptions you had to make due to missing information

Output JSON: {{
    "code": "...",
    "unknown_areas": ["area1", "area2"],
    "assumptions_made": ["assumption1", "assumption2"]
}}
"""
        response = await self._llm.generate(prompt)
        result = json.loads(response)

        return ReconstructedCode(
            from_team=team,
            code=result.get("code", ""),
            unknown_areas=result.get("unknown_areas", []),
            assumptions_made=result.get("assumptions_made", []),
        )

    async def _evaluate_equivalence(
        self,
        original: str,
        reconstructed: str,
    ) -> float:
        """Evaluate functional equivalence between original and reconstructed code.

        Args:
            original: Original source code
            reconstructed: Reconstructed code

        Returns:
            Equivalence score from 0.0 to 1.0
        """
        prompt = f"""
Compare these two code implementations and evaluate functional equivalence.

ORIGINAL CODE:
```python
{original}
```

RECONSTRUCTED CODE:
```python
{reconstructed}
```

Evaluate:
1. Would they produce the same outputs for the same inputs?
2. Do they handle the same edge cases?
3. Do they have the same side effects?
4. Are the algorithms functionally equivalent?

Score from 0.0 (completely different) to 1.0 (functionally identical).

Output JSON: {{"equivalence_score": 0.0-1.0, "analysis": "..."}}
"""
        response = await self._llm.generate(prompt)
        result = json.loads(response)
        return result.get("equivalence_score", 0.0)

    def get_documentation_gaps(self, result: CodeReconstructionResult) -> list[dict]:
        """Extract documentation gaps from reconstruction result."""
        return [
            {
                "area": gap.area,
                "severity": gap.severity.value,
                "recommendation": gap.recommendation,
                "description": gap.description,
            }
            for gap in result.documentation_gaps
        ]


# =============================================================================
# Strategy Registry
# =============================================================================


class StrategyRegistry:
    """Registry for verification strategies.

    Provides factory methods for creating strategy instances
    and discovering available strategies.

    Usage:
        registry = StrategyRegistry(llm_client)
        qa_strategy = registry.get(StrategyType.QA_INTERROGATION)
        all_strategies = registry.get_all()
    """

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize the registry.

        Args:
            llm_client: LLM client to inject into strategies
        """
        self._llm = llm_client
        self._strategies: dict[StrategyType, VerificationStrategy] = {}

    def get(self, strategy_type: StrategyType) -> VerificationStrategy:
        """Get or create a strategy instance.

        Args:
            strategy_type: Type of strategy to get

        Returns:
            Strategy instance

        Raises:
            ValueError: If strategy type is not supported
        """
        if strategy_type not in self._strategies:
            self._strategies[strategy_type] = self._create_strategy(strategy_type)
        return self._strategies[strategy_type]

    def get_all(self) -> list[VerificationStrategy]:
        """Get all available strategy instances."""
        return [self.get(st) for st in StrategyType]

    def get_by_level(self, level: VerificationLevel) -> list[VerificationStrategy]:
        """Get strategies at a specific verification level.

        Args:
            level: Verification level to filter by

        Returns:
            List of strategies at that level
        """
        return [s for s in self.get_all() if s.level == level]

    def _create_strategy(self, strategy_type: StrategyType) -> VerificationStrategy:
        """Create a strategy instance."""
        factories = {
            StrategyType.QA_INTERROGATION: lambda: QAInterrogator(self._llm),
            StrategyType.MASKED_RECONSTRUCTION: lambda: MaskedReconstructor(self._llm),
            StrategyType.SCENARIO_WALKTHROUGH: lambda: ScenarioWalker(self._llm),
            StrategyType.MUTATION_DETECTION: lambda: MutationDetector(self._llm),
            StrategyType.IMPACT_ANALYSIS: lambda: ImpactAnalyzer(self._llm),
            StrategyType.ADVERSARIAL_REVIEW: lambda: AdversarialReviewer(self._llm),
            StrategyType.TEST_GENERATION: lambda: TestGenerationValidator(self._llm),
            StrategyType.CODE_RECONSTRUCTION: lambda: CodeReconstructor(self._llm),
            # EDGE_CASE_EXTRACTION is a variant, can use QAInterrogator with different config
            StrategyType.EDGE_CASE_EXTRACTION: lambda: QAInterrogator(
                self._llm, edge_case_focus=True
            ),
        }

        factory = factories.get(strategy_type)
        if not factory:
            raise ValueError(f"Unsupported strategy type: {strategy_type}")
        return factory()
