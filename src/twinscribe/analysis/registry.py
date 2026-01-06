"""
Analyzer Registry.

Manages registration and lookup of analyzer implementations.
Supports plugin-style extension for adding new analyzers.
"""

from typing import Callable, Optional, Type

from twinscribe.analysis.analyzer import (
    Analyzer,
    AnalyzerConfig,
    AnalyzerType,
    Language,
)


# Type alias for analyzer factory function
AnalyzerFactory = Callable[[AnalyzerConfig], Analyzer]


class AnalyzerRegistry:
    """Registry for analyzer implementations.

    Provides centralized registration and lookup of analyzer
    implementations. Supports both class-based and factory-based
    registration.

    Usage:
        # Register an analyzer
        registry.register(AnalyzerType.PYCG, PyCGAnalyzer)

        # Get an analyzer instance
        analyzer = registry.get(AnalyzerType.PYCG, config)

        # Check availability
        if registry.is_registered(AnalyzerType.PYCG):
            ...
    """

    _instance: Optional["AnalyzerRegistry"] = None
    _analyzers: dict[AnalyzerType, Type[Analyzer] | AnalyzerFactory]
    _language_defaults: dict[Language, AnalyzerType]

    def __new__(cls) -> "AnalyzerRegistry":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._analyzers = {}
            cls._instance._language_defaults = {}
        return cls._instance

    def register(
        self,
        analyzer_type: AnalyzerType,
        implementation: Type[Analyzer] | AnalyzerFactory,
        default_for_language: Optional[Language] = None,
    ) -> None:
        """Register an analyzer implementation.

        Args:
            analyzer_type: Type of analyzer being registered
            implementation: Analyzer class or factory function
            default_for_language: Set as default for this language
        """
        self._analyzers[analyzer_type] = implementation

        if default_for_language is not None:
            self._language_defaults[default_for_language] = analyzer_type

    def unregister(self, analyzer_type: AnalyzerType) -> bool:
        """Unregister an analyzer.

        Args:
            analyzer_type: Type to unregister

        Returns:
            True if was registered, False otherwise
        """
        if analyzer_type in self._analyzers:
            del self._analyzers[analyzer_type]
            # Also remove from defaults
            for lang, default in list(self._language_defaults.items()):
                if default == analyzer_type:
                    del self._language_defaults[lang]
            return True
        return False

    def get(
        self,
        analyzer_type: AnalyzerType,
        config: Optional[AnalyzerConfig] = None,
    ) -> Analyzer:
        """Get an analyzer instance.

        Args:
            analyzer_type: Type of analyzer to get
            config: Configuration for the analyzer

        Returns:
            Analyzer instance

        Raises:
            KeyError: If analyzer type not registered
        """
        if analyzer_type not in self._analyzers:
            raise KeyError(f"Analyzer not registered: {analyzer_type}")

        implementation = self._analyzers[analyzer_type]

        # Create default config if not provided
        if config is None:
            config = self._default_config(analyzer_type)

        # Handle both class and factory
        if isinstance(implementation, type):
            return implementation(config)
        else:
            return implementation(config)

    def get_for_language(
        self,
        language: Language,
        config: Optional[AnalyzerConfig] = None,
    ) -> Analyzer:
        """Get the default analyzer for a language.

        Args:
            language: Target language
            config: Optional configuration

        Returns:
            Default analyzer for the language

        Raises:
            KeyError: If no default for language
        """
        if language not in self._language_defaults:
            raise KeyError(f"No default analyzer for language: {language}")

        analyzer_type = self._language_defaults[language]
        return self.get(analyzer_type, config)

    def is_registered(self, analyzer_type: AnalyzerType) -> bool:
        """Check if an analyzer type is registered.

        Args:
            analyzer_type: Type to check

        Returns:
            True if registered
        """
        return analyzer_type in self._analyzers

    def list_registered(self) -> list[AnalyzerType]:
        """List all registered analyzer types.

        Returns:
            List of registered types
        """
        return list(self._analyzers.keys())

    def list_for_language(self, language: Language) -> list[AnalyzerType]:
        """List analyzers that support a language.

        Args:
            language: Target language

        Returns:
            List of compatible analyzer types
        """
        compatible = []
        for analyzer_type in self._analyzers:
            # Check if analyzer supports the language
            # This is a simplified check - real implementation would
            # inspect the analyzer's supported languages
            config = self._default_config(analyzer_type)
            if config.language == language or config.language == Language.MULTI:
                compatible.append(analyzer_type)
        return compatible

    def _default_config(self, analyzer_type: AnalyzerType) -> AnalyzerConfig:
        """Get default configuration for an analyzer type.

        Args:
            analyzer_type: Analyzer type

        Returns:
            Default configuration
        """
        # Import default configs
        from twinscribe.analysis.analyzer import (
            JAVA_CALLGRAPH_CONFIG,
            PYAN3_CONFIG,
            PYCG_CONFIG,
            SOURCETRAIL_CONFIG,
            TS_CALLGRAPH_CONFIG,
        )

        defaults = {
            AnalyzerType.PYCG: PYCG_CONFIG,
            AnalyzerType.PYAN3: PYAN3_CONFIG,
            AnalyzerType.JAVA_CALLGRAPH: JAVA_CALLGRAPH_CONFIG,
            AnalyzerType.TS_CALLGRAPH: TS_CALLGRAPH_CONFIG,
            AnalyzerType.SOURCETRAIL: SOURCETRAIL_CONFIG,
        }

        return defaults.get(
            analyzer_type,
            AnalyzerConfig(
                analyzer_type=analyzer_type,
                language=Language.MULTI,
            ),
        )

    @classmethod
    def reset(cls) -> None:
        """Reset the registry (mainly for testing).

        Clears all registrations.
        """
        if cls._instance is not None:
            cls._instance._analyzers.clear()
            cls._instance._language_defaults.clear()


# Global registry instance
_registry = AnalyzerRegistry()


def register_analyzer(
    analyzer_type: AnalyzerType,
    implementation: Type[Analyzer] | AnalyzerFactory,
    default_for_language: Optional[Language] = None,
) -> None:
    """Register an analyzer with the global registry.

    Convenience function for module-level registration.

    Args:
        analyzer_type: Type of analyzer
        implementation: Analyzer class or factory
        default_for_language: Set as default for language
    """
    _registry.register(analyzer_type, implementation, default_for_language)


def get_analyzer(
    analyzer_type: AnalyzerType,
    config: Optional[AnalyzerConfig] = None,
) -> Analyzer:
    """Get an analyzer from the global registry.

    Convenience function for getting analyzers.

    Args:
        analyzer_type: Type of analyzer
        config: Optional configuration

    Returns:
        Analyzer instance
    """
    return _registry.get(analyzer_type, config)


def get_registry() -> AnalyzerRegistry:
    """Get the global registry instance.

    Returns:
        Global AnalyzerRegistry
    """
    return _registry
