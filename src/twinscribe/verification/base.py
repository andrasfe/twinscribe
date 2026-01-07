"""
CrossCheck Verification Framework - Base Classes and Enumerations.

This module provides the foundational abstractions for the verification system:

- VerificationStrategy: Abstract base class for all 8 verification strategies
- VerificationLevel: Hierarchy of verification intensity
- StrategyType: Enumeration of available strategies
- StrategyConfig: Base configuration class for strategies
- QuestionCategory: Categories for Q&A interrogation
- MaskType: Types of code masking for reconstruction tests
- ScenarioType: Types of execution scenarios
- MutationType: Types of code mutations for detection tests

The verification system performs "active verification" - testing whether
documentation is sufficient to understand, predict, and reconstruct code behavior.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Generic, TypeVar

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from twinscribe.verification.models import (
        VerificationChallenge,
        VerificationResult,
    )

# Generic type variables for strategy inputs/outputs
ChallengeT = TypeVar("ChallengeT", bound="VerificationChallenge")
ResultT = TypeVar("ResultT", bound="VerificationResult")


class VerificationLevel(str, Enum):
    """Hierarchy of verification intensity.

    Each level builds upon the previous, with increasing thoroughness
    and computational cost.

    Attributes:
        PASSIVE: Basic comparison of A vs B outputs (existing)
        ACTIVE: Q&A, masked reconstruction, scenario walkthrough
        BEHAVIORAL: Mutation detection, impact analysis, edge cases
        GENERATIVE: Code reconstruction, test generation, inverse docs
    """

    PASSIVE = "passive"
    ACTIVE = "active"
    BEHAVIORAL = "behavioral"
    GENERATIVE = "generative"


class StrategyType(str, Enum):
    """Types of verification strategies available.

    Each strategy tests a different aspect of documentation quality.
    """

    # Level 2: Active Interrogation
    QA_INTERROGATION = "qa_interrogation"
    MASKED_RECONSTRUCTION = "masked_reconstruction"
    SCENARIO_WALKTHROUGH = "scenario_walkthrough"

    # Level 3: Behavioral Verification
    MUTATION_DETECTION = "mutation_detection"
    IMPACT_ANALYSIS = "impact_analysis"
    EDGE_CASE_EXTRACTION = "edge_case_extraction"

    # Level 4: Generative Verification
    CODE_RECONSTRUCTION = "code_reconstruction"
    TEST_GENERATION = "test_generation"
    ADVERSARIAL_REVIEW = "adversarial_review"


class QuestionCategory(str, Enum):
    """Categories for Q&A interrogation questions.

    Each category tests different documentation aspects.
    """

    RETURN_VALUE = "return_value"
    SIDE_EFFECT = "side_effect"
    ERROR_HANDLING = "error_handling"
    EDGE_CASE = "edge_case"
    DEPENDENCY = "dependency"
    CALL_FLOW = "call_flow"
    PRECONDITION = "precondition"
    POSTCONDITION = "postcondition"


class MaskType(str, Enum):
    """Types of code elements that can be masked.

    Used in masked reconstruction challenges to test
    documentation specificity.
    """

    CONSTANTS = "constants"
    STRINGS = "strings"
    CONDITIONS = "conditions"
    RETURNS = "returns"
    FUNCTION_CALLS = "function_calls"
    ERROR_HANDLING = "error_handling"
    FULL_BLOCKS = "full_blocks"
    LOOP_BOUNDS = "loop_bounds"


class ScenarioType(str, Enum):
    """Types of execution scenarios for walkthrough tests.

    Each type validates different documentation aspects.
    """

    HAPPY_PATH = "happy_path"
    ERROR_PATH = "error_path"
    EDGE_CASE = "edge_case"
    CONCURRENT = "concurrent"
    STATE_DEPENDENT = "state_dependent"
    BOUNDARY = "boundary"


class MutationType(str, Enum):
    """Types of code mutations for detection tests.

    Each mutation type tests different precision aspects
    of documentation.
    """

    BOUNDARY = "boundary"  # >= to >, < to <=
    OFF_BY_ONE = "off_by_one"  # i < n to i <= n
    WRONG_VARIABLE = "wrong_variable"  # x to y
    MISSING_CALL = "missing_call"  # Remove function call
    WRONG_ORDER = "wrong_order"  # Swap statement order
    NULL_HANDLING = "null_handling"  # Remove null check
    TYPE_COERCION = "type_coercion"  # int to float, etc.
    LOGIC_INVERSION = "logic_inversion"  # and to or, True to False


class ChangeType(str, Enum):
    """Types of changes for impact analysis challenges."""

    SIGNATURE = "signature"
    RETURN_TYPE = "return_type"
    BEHAVIOR = "behavior"
    REMOVAL = "removal"
    RENAME = "rename"


class Severity(str, Enum):
    """Severity levels for documentation gaps and issues."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class VerificationStrategy(ABC, Generic[ChallengeT, ResultT]):
    """Abstract base class for all verification strategies.

    Each strategy implements a different approach to testing
    documentation quality through active verification.

    Type Parameters:
        ChallengeT: The challenge type this strategy generates
        ResultT: The result type this strategy produces

    Subclasses must implement:
        - generate_challenge: Create a verification challenge
        - evaluate: Evaluate team responses against ground truth
        - get_documentation_gaps: Extract identified gaps

    Attributes:
        strategy_type: The type of this strategy
        level: The verification level this strategy belongs to
        description: Human-readable description of what this tests
    """

    def __init__(
        self,
        strategy_type: StrategyType,
        level: VerificationLevel,
        description: str,
    ) -> None:
        """Initialize the verification strategy.

        Args:
            strategy_type: Type identifier for this strategy
            level: Verification hierarchy level
            description: Human-readable description
        """
        self._strategy_type = strategy_type
        self._level = level
        self._description = description

    @property
    def strategy_type(self) -> StrategyType:
        """Get the strategy type identifier."""
        return self._strategy_type

    @property
    def level(self) -> VerificationLevel:
        """Get the verification level."""
        return self._level

    @property
    def description(self) -> str:
        """Get the human-readable description."""
        return self._description

    @abstractmethod
    async def generate_challenge(
        self,
        component_id: str,
        source_code: str,
        **kwargs,
    ) -> ChallengeT:
        """Generate a verification challenge for a component.

        Args:
            component_id: Unique identifier of the component
            source_code: Source code of the component
            **kwargs: Strategy-specific parameters

        Returns:
            A challenge object for this strategy
        """
        ...

    @abstractmethod
    async def evaluate(
        self,
        challenge: ChallengeT,
        team_a_response: str,
        team_b_response: str,
        ground_truth: str | None = None,
    ) -> ResultT:
        """Evaluate team responses against the challenge.

        Args:
            challenge: The challenge that was presented
            team_a_response: Response from Team A (using their docs)
            team_b_response: Response from Team B (using their docs)
            ground_truth: Ground truth from code analysis (if available)

        Returns:
            Evaluation result with scores and identified gaps
        """
        ...

    @abstractmethod
    def get_documentation_gaps(self, result: ResultT) -> list[dict]:
        """Extract documentation gaps from evaluation result.

        Args:
            result: The evaluation result

        Returns:
            List of documentation gap dictionaries with:
                - area: Which aspect is lacking
                - severity: How critical the gap is
                - recommendation: Suggested fix
        """
        ...

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(type={self.strategy_type.value}, level={self.level.value})"
        )


# =============================================================================
# Strategy Configuration Models
# =============================================================================


class StrategyConfig(BaseModel):
    """Base configuration for verification strategies.

    Provides common configuration options that apply to all strategies.
    Strategy-specific configs extend this class.

    Attributes:
        enabled: Whether this strategy is enabled
        weight: Weight in overall score calculation (0.0-1.0)
        max_items: Maximum number of challenges/questions to generate
        timeout_seconds: Maximum time for strategy execution
        retry_on_failure: Whether to retry on transient failures
        max_retries: Maximum retry attempts
    """

    enabled: bool = Field(
        default=True,
        description="Whether strategy is enabled",
    )
    weight: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Weight in score calculation",
    )
    max_items: int = Field(
        default=10,
        ge=1,
        description="Max challenges to generate",
    )
    timeout_seconds: int = Field(
        default=120,
        ge=10,
        description="Execution timeout",
    )
    retry_on_failure: bool = Field(
        default=True,
        description="Retry on transient failures",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Max retry attempts",
    )


class QAInterrogatorConfig(StrategyConfig):
    """Configuration for Q&A Interrogation strategy.

    Attributes:
        questions_per_component: Number of questions to generate
        include_edge_cases: Include edge case questions
        include_error_handling: Include error handling questions
        difficulty_range: Allowed difficulty levels (1-5)
        categories: Question categories to include
    """

    questions_per_component: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Questions per component",
    )
    include_edge_cases: bool = Field(
        default=True,
        description="Include edge case questions",
    )
    include_error_handling: bool = Field(
        default=True,
        description="Include error handling questions",
    )
    difficulty_range: tuple[int, int] = Field(
        default=(1, 5),
        description="Difficulty range (min, max)",
    )
    categories: list[QuestionCategory] | None = Field(
        default=None,
        description="Categories to include (None = all)",
    )
    edge_case_focus: bool = Field(
        default=False,
        description="Focus primarily on edge case questions",
    )


class MaskedReconstructorConfig(StrategyConfig):
    """Configuration for Masked Reconstruction strategy.

    Attributes:
        mask_ratio: Ratio of code to mask (0.0-1.0)
        mask_types: Types of elements to mask
        min_masks: Minimum masks to apply
        max_masks: Maximum masks to apply
    """

    mask_ratio: float = Field(
        default=0.3,
        ge=0.1,
        le=0.8,
        description="Ratio of code to mask",
    )
    mask_types: list[MaskType] = Field(
        default_factory=lambda: [
            MaskType.CONSTANTS,
            MaskType.STRINGS,
            MaskType.CONDITIONS,
            MaskType.RETURNS,
        ],
        description="Types of elements to mask",
    )
    min_masks: int = Field(
        default=3,
        ge=1,
        description="Minimum masks to apply",
    )
    max_masks: int = Field(
        default=15,
        ge=1,
        description="Maximum masks to apply",
    )


class ScenarioWalkerConfig(StrategyConfig):
    """Configuration for Scenario Walkthrough strategy.

    Attributes:
        scenarios_per_component: Number of scenarios to generate
        scenario_types: Types of scenarios to include
        include_side_effects: Test side effect prediction
        max_call_depth: Maximum call chain depth to test
    """

    scenarios_per_component: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Scenarios per component",
    )
    scenario_types: list[ScenarioType] = Field(
        default_factory=lambda: [
            ScenarioType.HAPPY_PATH,
            ScenarioType.ERROR_PATH,
            ScenarioType.EDGE_CASE,
        ],
        description="Scenario types to include",
    )
    include_side_effects: bool = Field(
        default=True,
        description="Test side effect prediction",
    )
    max_call_depth: int = Field(
        default=5,
        ge=1,
        description="Max call chain depth",
    )


class MutationDetectorConfig(StrategyConfig):
    """Configuration for Mutation Detection strategy.

    Attributes:
        mutations_per_component: Number of mutations to generate
        mutation_types: Types of mutations to apply
        subtle_only: Only generate subtle, hard-to-detect mutations
    """

    mutations_per_component: int = Field(
        default=8,
        ge=1,
        le=30,
        description="Mutations per component",
    )
    mutation_types: list[MutationType] = Field(
        default_factory=lambda: [
            MutationType.BOUNDARY,
            MutationType.OFF_BY_ONE,
            MutationType.NULL_HANDLING,
            MutationType.LOGIC_INVERSION,
        ],
        description="Mutation types to apply",
    )
    subtle_only: bool = Field(
        default=False,
        description="Only subtle mutations",
    )


class ImpactAnalyzerConfig(StrategyConfig):
    """Configuration for Impact Analysis strategy.

    Attributes:
        changes_per_component: Number of hypothetical changes
        change_types: Types of changes to propose
        include_transitive: Include transitive dependencies
        max_impact_depth: Maximum dependency depth
    """

    changes_per_component: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Changes per component",
    )
    change_types: list[ChangeType] = Field(
        default_factory=lambda: [
            ChangeType.SIGNATURE,
            ChangeType.RETURN_TYPE,
            ChangeType.BEHAVIOR,
        ],
        description="Change types to propose",
    )
    include_transitive: bool = Field(
        default=True,
        description="Include transitive dependencies",
    )
    max_impact_depth: int = Field(
        default=3,
        ge=1,
        description="Max dependency depth",
    )


class AdversarialReviewerConfig(StrategyConfig):
    """Configuration for Adversarial Review strategy.

    Attributes:
        max_findings_per_team: Maximum findings per reviewer
        validate_against_code: Validate findings against source
        severity_threshold: Minimum severity to report
    """

    max_findings_per_team: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Max findings per team",
    )
    validate_against_code: bool = Field(
        default=True,
        description="Validate against source code",
    )
    severity_threshold: Severity = Field(
        default=Severity.LOW,
        description="Min severity to report",
    )


class TestGeneratorConfig(StrategyConfig):
    """Configuration for Test Generation Validation strategy.

    Attributes:
        tests_per_team: Number of tests to generate per team
        test_framework: Target test framework
        include_edge_cases: Generate edge case tests
        include_negative_tests: Generate negative/error tests
        execute_tests: Actually execute generated tests
    """

    tests_per_team: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Tests per team",
    )
    test_framework: str = Field(
        default="pytest",
        description="Target test framework",
    )
    include_edge_cases: bool = Field(
        default=True,
        description="Generate edge case tests",
    )
    include_negative_tests: bool = Field(
        default=True,
        description="Generate negative tests",
    )
    execute_tests: bool = Field(
        default=False,
        description="Actually execute tests (requires sandbox)",
    )


class CodeReconstructorConfig(StrategyConfig):
    """Configuration for Code Reconstruction strategy.

    Attributes:
        provide_signature: Provide function signature
        provide_imports: Provide required imports
        functional_equivalence_threshold: Min equivalence score
    """

    provide_signature: bool = Field(
        default=True,
        description="Provide function signature",
    )
    provide_imports: bool = Field(
        default=True,
        description="Provide required imports",
    )
    functional_equivalence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Min equivalence score to pass",
    )


# Type mapping for configs
STRATEGY_CONFIG_TYPES: dict[StrategyType, type[StrategyConfig]] = {
    StrategyType.QA_INTERROGATION: QAInterrogatorConfig,
    StrategyType.MASKED_RECONSTRUCTION: MaskedReconstructorConfig,
    StrategyType.SCENARIO_WALKTHROUGH: ScenarioWalkerConfig,
    StrategyType.MUTATION_DETECTION: MutationDetectorConfig,
    StrategyType.IMPACT_ANALYSIS: ImpactAnalyzerConfig,
    StrategyType.ADVERSARIAL_REVIEW: AdversarialReviewerConfig,
    StrategyType.TEST_GENERATION: TestGeneratorConfig,
    StrategyType.CODE_RECONSTRUCTION: CodeReconstructorConfig,
    StrategyType.EDGE_CASE_EXTRACTION: QAInterrogatorConfig,  # Uses QA config with edge_case_focus=True
}
