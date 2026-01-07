"""
CrossCheck Verification Framework.

This module provides active verification capabilities for testing documentation
quality through multiple verification strategies. It extends the existing
dual-stream documentation system with comprehensive quality assessment.

Verification Levels:
    - PASSIVE: Basic comparison of A vs B outputs (existing)
    - ACTIVE: Q&A interrogation, masked reconstruction, scenario walkthrough
    - BEHAVIORAL: Mutation detection, impact analysis, edge case extraction
    - GENERATIVE: Code reconstruction, test generation, adversarial review

Key Components:
    - VerificationStrategy: Base class for implementing verification strategies
    - VerificationPipeline: Orchestrates verification across the documentation pipeline
    - VerificationScores: Aggregated scores with quality grading (A/B/C/F)
    - Strategy implementations: QAInterrogator, MaskedReconstructor, etc.

Quick Start:
    ```python
    from twinscribe.verification import (
        VerificationPipeline,
        PipelineBuilder,
        StrategyType,
        VerificationScores,
    )

    # Build pipeline with minimum recommended strategies
    pipeline = (
        PipelineBuilder()
        .with_minimum_strategies()
        .with_doc_provider(my_doc_provider)
        .with_source_provider(my_source_provider)
        .with_llm_client(my_llm)
        .build()
    )

    # Verify components
    run = await pipeline.verify_components(["mypackage.module.MyClass.method"])

    # Check results
    scores = run.get_aggregate_scores()
    print(f"Grade: {scores.quality_grade}")  # A, B, C, or F
    print(f"Quality: {scores.overall_quality:.1%}")
    print(f"Weakest areas: {scores.get_weakest_areas()}")
    ```

Configuration:
    See verification_config.yaml for configurable thresholds and strategy settings.

For detailed documentation, see /home/andras/twinscribe/crosscheck_verification_framework.md
"""

# Base classes and enums
from twinscribe.verification.base import (
    STRATEGY_CONFIG_TYPES,
    AdversarialReviewerConfig,
    ChangeType,
    CodeReconstructorConfig,
    ImpactAnalyzerConfig,
    MaskedReconstructorConfig,
    MaskType,
    MutationDetectorConfig,
    MutationType,
    QAInterrogatorConfig,
    QuestionCategory,
    ScenarioType,
    ScenarioWalkerConfig,
    Severity,
    # Configuration classes
    StrategyConfig,
    StrategyType,
    TestGeneratorConfig,
    VerificationLevel,
    VerificationStrategy,
)

# Data models
from twinscribe.verification.models import (
    # Adversarial review models
    AdversarialChallenge,
    AdversarialFinding,
    AdversarialResult,
    # Code reconstruction models
    CodeReconstructionChallenge,
    CodeReconstructionResult,
    # Base models
    DocumentationGap,
    # Scenario walkthrough models
    ExecutionTrace,
    # Test generation models
    GeneratedTest,
    # Impact analysis models
    ImpactChallenge,
    ImpactPrediction,
    ImpactResult,
    # Masked reconstruction models
    Mask,
    MaskedChallenge,
    MaskEvaluation,
    MaskReconstruction,
    # Mutation detection models
    Mutation,
    MutationAssessment,
    MutationChallenge,
    MutationEvaluation,
    MutationResult,
    # Q&A models
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
    VerificationChallenge,
    VerificationResult,
)

# Pipeline integration
from twinscribe.verification.pipeline import (
    ComponentVerification,
    DocumentationProvider,
    PipelineBuilder,
    PipelineConfig,
    PipelinePhase,
    PipelineRun,
    PipelineStatus,
    SourceCodeProvider,
    StrategyExecution,
    TicketCreator,
    VerificationPipeline,
)

# Score aggregation
from twinscribe.verification.scores import (
    QualityGrade,
    QualityReport,
    ScoreAggregator,
    ScoreAnalyzer,
    ScoreComparison,
    VerificationScores,
    VerificationThresholds,
    WeaknessArea,
)

# Strategy implementations
from twinscribe.verification.strategies import (
    AdversarialReviewer,
    CodeReconstructor,
    ImpactAnalyzer,
    MaskedReconstructor,
    MutationDetector,
    QAInterrogator,
    ScenarioWalker,
    StrategyRegistry,
    TestGenerationValidator,
)

__all__ = [
    # Base classes and enums
    "ChangeType",
    "MaskType",
    "MutationType",
    "QuestionCategory",
    "ScenarioType",
    "Severity",
    "StrategyType",
    "VerificationLevel",
    "VerificationStrategy",
    # Configuration classes
    "StrategyConfig",
    "QAInterrogatorConfig",
    "MaskedReconstructorConfig",
    "ScenarioWalkerConfig",
    "MutationDetectorConfig",
    "ImpactAnalyzerConfig",
    "AdversarialReviewerConfig",
    "TestGeneratorConfig",
    "CodeReconstructorConfig",
    "STRATEGY_CONFIG_TYPES",
    # Base models
    "DocumentationGap",
    "VerificationChallenge",
    "VerificationResult",
    # Q&A models
    "QAChallenge",
    "QAEvaluation",
    "QAResult",
    "Question",
    "TeamAnswer",
    # Masked reconstruction models
    "Mask",
    "MaskedChallenge",
    "MaskEvaluation",
    "MaskReconstruction",
    "ReconstructionResult",
    # Scenario walkthrough models
    "ExecutionTrace",
    "Scenario",
    "ScenarioChallenge",
    "ScenarioEvaluation",
    "ScenarioResult",
    # Mutation detection models
    "Mutation",
    "MutationAssessment",
    "MutationChallenge",
    "MutationEvaluation",
    "MutationResult",
    # Impact analysis models
    "ImpactChallenge",
    "ImpactPrediction",
    "ImpactResult",
    # Adversarial review models
    "AdversarialChallenge",
    "AdversarialFinding",
    "AdversarialResult",
    # Test generation models
    "GeneratedTest",
    "TestExecution",
    "TestGenerationChallenge",
    "TestValidationResult",
    # Code reconstruction models
    "CodeReconstructionChallenge",
    "CodeReconstructionResult",
    "ReconstructedCode",
    # Strategy implementations
    "AdversarialReviewer",
    "CodeReconstructor",
    "ImpactAnalyzer",
    "MaskedReconstructor",
    "MutationDetector",
    "QAInterrogator",
    "ScenarioWalker",
    "StrategyRegistry",
    "TestGenerationValidator",
    # Pipeline integration
    "ComponentVerification",
    "DocumentationProvider",
    "PipelineBuilder",
    "PipelineConfig",
    "PipelinePhase",
    "PipelineRun",
    "PipelineStatus",
    "SourceCodeProvider",
    "StrategyExecution",
    "TicketCreator",
    "VerificationPipeline",
    # Score aggregation
    "QualityGrade",
    "QualityReport",
    "ScoreAggregator",
    "ScoreAnalyzer",
    "ScoreComparison",
    "VerificationScores",
    "VerificationThresholds",
    "WeaknessArea",
]
