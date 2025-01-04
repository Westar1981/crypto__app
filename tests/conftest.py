import pytest

@pytest.fixture(scope="function")  # Set the scope for the asyncio fixture loop
def sample_response():
    """Fixture for providing a sample AI response."""
    return "The AI suggests optimizing the algorithm for better performance."

@pytest.fixture
def blackbox_analyzer():
    from core.analysis.blackbox_analyzer import BlackboxAnalyzer
    from core.analysis.cot_analyzer import CoTAnalyzer
    cot_analyzer = CoTAnalyzer()
    return BlackboxAnalyzer(cot_analyzer)

@pytest.fixture
def cot_analyzer():
    from core.analysis.cot_analyzer import CoTAnalyzer
    return CoTAnalyzer()
