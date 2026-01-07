"""
End-to-end tests for TwinScribe with real LLM calls.

These tests verify the complete documentation workflow using actual
LLM API calls through OpenRouter. They are designed to:

1. Only run when OPENROUTER_API_KEY is available
2. Use minimal sample code to reduce API costs
3. Include timeouts to prevent hanging
4. Verify output schemas and call graph accuracy

Test markers:
- @pytest.mark.e2e: End-to-end tests
- @pytest.mark.slow: Tests that may take significant time

To run these tests:
    OPENROUTER_API_KEY=your_key pytest tests/e2e/ -v --timeout=300
"""
