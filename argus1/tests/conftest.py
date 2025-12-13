"""
Pytest Configuration and Fixtures

Provides shared fixtures and configuration for all tests.
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(autouse=True)
def reset_resilience_state():
    """Reset all resilience state before each test"""
    from shared.resilience import reset_all
    reset_all()
    yield
    reset_all()


@pytest.fixture
def sample_raw_article():
    """Sample raw article for testing"""
    from argus1.aggregator_enhanced import RawArticle
    
    return RawArticle(
        source_type="newsapi",
        source_name="Reuters",
        title="Microsoft announces $80B AI infrastructure investment",
        url="https://reuters.com/article/msft-ai-investment",
        published_at="2025-01-15T10:00:00Z",
        content="Microsoft announced today a massive $80 billion investment in AI infrastructure spanning data centers across 15 countries...",
        author="John Doe",
        ticker="MSFT"
    )


@pytest.fixture
def sample_processed_article():
    """Sample processed article for testing"""
    from argus1.processor_enhanced import ProcessedArticle
    
    return ProcessedArticle(
        article_hash="abc123def456",
        source_type="newsapi",
        source_name="Reuters",
        title="Microsoft announces $80B AI infrastructure investment",
        url="https://reuters.com/article/msft-ai-investment",
        published_at="2025-01-15T10:00:00Z",
        y2ai_category="spending",
        extracted_facts=["Microsoft investing $80B in AI", "Investment spans 15 countries"],
        impact_score=0.9,
        sentiment="bullish",
        companies_mentioned=["Microsoft", "NVIDIA"],
        dollar_amounts=["$80 billion"],
        key_quotes=["This is our largest infrastructure commitment"],
        processed_at="2025-01-15T12:00:00Z"
    )


@pytest.fixture
def sample_bubble_reading():
    """Sample bubble index reading for testing"""
    from y2ai.bubble_index_enhanced import BubbleIndexReading
    
    return BubbleIndexReading(
        date="2025-01-15",
        vix=15.0,
        cape=30.0,
        credit_spread_ig=100.0,
        credit_spread_hy=350.0,
        vix_zscore=-0.5,
        cape_zscore=0.5,
        credit_zscore=0.0,
        bubble_index=57.5,
        bifurcation_score=0.45,
        regime="ADOPTION",
        data_sources={"vix": "live", "cape": "live", "credit": "live"}
    )


@pytest.fixture
def sample_stock_data():
    """Sample stock data dictionary for testing"""
    return {
        # Supply Constraint Pillar
        "TSM": {"price": 100.0, "change_today": 1.0, "change_5day": 2.0, "change_ytd": 10.0},
        "ASML": {"price": 500.0, "change_today": 1.5, "change_5day": 3.0, "change_ytd": 15.0},
        "VRT": {"price": 80.0, "change_today": 2.0, "change_5day": 4.0, "change_ytd": 20.0},
        # Capital Efficiency Pillar
        "GOOGL": {"price": 150.0, "change_today": 0.5, "change_5day": 1.0, "change_ytd": 5.0},
        "MSFT": {"price": 400.0, "change_today": 0.8, "change_5day": 1.5, "change_ytd": 8.0},
        "AMZN": {"price": 180.0, "change_today": 1.2, "change_5day": 2.5, "change_ytd": 12.0},
        # Demand Depth Pillar
        "NVDA": {"price": 500.0, "change_today": 3.0, "change_5day": 5.0, "change_ytd": 50.0},
        "SNOW": {"price": 150.0, "change_today": -1.0, "change_5day": -2.0, "change_ytd": -5.0},
        "NOW": {"price": 700.0, "change_today": 0.5, "change_5day": 1.0, "change_ytd": 10.0},
        # Benchmarks
        "SPY": {"price": 500.0, "change_today": 0.5, "change_5day": 1.0, "change_ytd": 10.0},
        "QQQ": {"price": 400.0, "change_today": 0.8, "change_5day": 1.5, "change_ytd": 15.0},
    }


@pytest.fixture
def mock_claude_response():
    """Mock Claude API response for testing"""
    return '''```json
{
    "category": "spending",
    "extracted_facts": [
        "Microsoft announced $80 billion AI infrastructure investment",
        "Investment spans data centers across 15 countries"
    ],
    "impact_score": 0.9,
    "sentiment": "bullish",
    "companies_mentioned": ["Microsoft", "NVIDIA"],
    "dollar_amounts": ["$80 billion"],
    "key_quotes": ["This represents our largest infrastructure commitment in company history"]
}
```'''


# =============================================================================
# MARKERS
# =============================================================================

def pytest_configure(config):
    """Configure custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# =============================================================================
# HOOKS
# =============================================================================

def pytest_collection_modifyitems(config, items):
    """Automatically add markers based on test location"""
    for item in items:
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)
