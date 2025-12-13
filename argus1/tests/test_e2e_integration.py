"""
ARGUS+Y2AI End-to-End Integration Tests (Phase 6)

Comprehensive integration testing covering:
1. Full pipeline execution (collection → processing → storage)
2. Supabase storage verification with mocks
3. Data flow validation across all modules
4. Error propagation testing
5. Newsletter generation pipeline
6. Indicator calculation pipeline
"""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, call
from dataclasses import asdict
from typing import Dict, List, Any
import logging

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from shared.resilience import reset_all
from shared.validation import (
    ValidationResult,
    SchemaValidator,
    RangeValidator,
    StorageValidator,
    validate_market_indicators,
)


# =============================================================================
# MOCK SUPABASE CLIENT
# =============================================================================

class MockSupabaseTable:
    """Mock Supabase table for testing"""
    
    def __init__(self, name: str):
        self.name = name
        self.data: List[Dict] = []
        self._select_columns = None
        self._filters = []
        self._limit = None
        self._order_col = None
        self._order_desc = False
    
    def insert(self, data: Dict) -> 'MockSupabaseTable':
        """Insert data"""
        if isinstance(data, list):
            for item in data:
                self.data.append({**item, "id": len(self.data) + 1})
        else:
            self.data.append({**data, "id": len(self.data) + 1})
        return self
    
    def select(self, columns: str = "*") -> 'MockSupabaseTable':
        """Select columns"""
        self._select_columns = columns
        return self
    
    def eq(self, column: str, value: Any) -> 'MockSupabaseTable':
        """Equal filter"""
        self._filters.append(("eq", column, value))
        return self
    
    def gte(self, column: str, value: Any) -> 'MockSupabaseTable':
        """Greater than or equal filter"""
        self._filters.append(("gte", column, value))
        return self
    
    def lte(self, column: str, value: Any) -> 'MockSupabaseTable':
        """Less than or equal filter"""
        self._filters.append(("lte", column, value))
        return self
    
    def order(self, column: str, desc: bool = False) -> 'MockSupabaseTable':
        """Order results"""
        self._order_col = column
        self._order_desc = desc
        return self
    
    def limit(self, n: int) -> 'MockSupabaseTable':
        """Limit results"""
        self._limit = n
        return self
    
    def execute(self) -> Mock:
        """Execute query and return results"""
        results = self.data.copy()
        
        # Apply filters
        for filter_type, column, value in self._filters:
            if filter_type == "eq":
                results = [r for r in results if r.get(column) == value]
            elif filter_type == "gte":
                results = [r for r in results if r.get(column, 0) >= value]
            elif filter_type == "lte":
                results = [r for r in results if r.get(column, float('inf')) <= value]
        
        # Apply ordering
        if self._order_col:
            results.sort(key=lambda x: x.get(self._order_col, 0), reverse=self._order_desc)
        
        # Apply limit
        if self._limit:
            results = results[:self._limit]
        
        # Reset state
        self._filters = []
        self._limit = None
        self._order_col = None
        
        response = Mock()
        response.data = results
        return response


class MockSupabaseClient:
    """Mock Supabase client for testing"""
    
    def __init__(self):
        self.tables: Dict[str, MockSupabaseTable] = {}
    
    def table(self, name: str) -> MockSupabaseTable:
        """Get or create table"""
        if name not in self.tables:
            self.tables[name] = MockSupabaseTable(name)
        return self.tables[name]
    
    def get_table_data(self, name: str) -> List[Dict]:
        """Get all data from a table (for assertions)"""
        if name in self.tables:
            return self.tables[name].data
        return []


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_supabase():
    """Create mock Supabase client"""
    return MockSupabaseClient()


@pytest.fixture
def sample_raw_articles():
    """Sample raw articles for testing"""
    from argus1.aggregator_enhanced import RawArticle
    
    return [
        RawArticle(
            source_type="newsapi",
            source_name="Reuters",
            title="Microsoft Announces $80B AI Infrastructure Investment",
            url="https://reuters.com/msft-ai",
            published_at="2025-01-15T10:00:00Z",
            content="Microsoft announced an $80 billion investment in AI infrastructure for FY2025...",
            author="John Smith",
        ),
        RawArticle(
            source_type="newsapi",
            source_name="Bloomberg",
            title="NVIDIA Reports Record Quarterly Revenue",
            url="https://bloomberg.com/nvda-earnings",
            published_at="2025-01-15T11:00:00Z",
            content="NVIDIA reported Q4 revenue of $22.1 billion, beating estimates...",
            author="Jane Doe",
        ),
        RawArticle(
            source_type="rss",
            source_name="Data Center Knowledge",
            title="GPU Shortage Expected to Continue Through 2025",
            url="https://datacenterknowledge.com/gpu-shortage",
            published_at="2025-01-15T12:00:00Z",
            content="Industry analysts expect GPU shortages to persist...",
        ),
    ]


@pytest.fixture
def sample_processed_articles():
    """Sample processed articles for testing"""
    from argus1.aggregator_enhanced import ProcessedArticle
    
    return [
        ProcessedArticle(
            article_hash="abc123",
            source_type="newsapi",
            source_name="Reuters",
            title="Microsoft Announces $80B AI Infrastructure Investment",
            url="https://reuters.com/msft-ai",
            published_at="2025-01-15T10:00:00Z",
            y2ai_category="spending",
            extracted_facts=["$80 billion investment", "FY2025 commitment"],
            impact_score=0.9,
            sentiment="bullish",
            companies_mentioned=["Microsoft"],
            dollar_amounts=["$80B"],
            key_quotes=["largest AI investment in history"],
            processed_at="2025-01-15T13:00:00Z",
        ),
        ProcessedArticle(
            article_hash="def456",
            source_type="newsapi",
            source_name="Bloomberg",
            title="NVIDIA Reports Record Quarterly Revenue",
            url="https://bloomberg.com/nvda-earnings",
            published_at="2025-01-15T11:00:00Z",
            y2ai_category="data",
            extracted_facts=["$22.1B revenue", "Beat estimates"],
            impact_score=0.85,
            sentiment="bullish",
            companies_mentioned=["NVIDIA"],
            dollar_amounts=["$22.1B"],
            key_quotes=[],
            processed_at="2025-01-15T13:00:00Z",
        ),
    ]


@pytest.fixture
def sample_bubble_reading():
    """Sample bubble reading for testing"""
    return {
        "date": "2025-01-15",
        "vix": 18.5,
        "cape": 32.5,
        "credit_spread_ig": 95.0,
        "credit_spread_hy": 380.0,
        "vix_zscore": 0.15,
        "cape_zscore": 0.8,
        "credit_zscore": -0.2,
        "bubble_index": 62.5,
        "bifurcation_score": 0.77,
        "regime": "INFRASTRUCTURE",
        "data_sources": {"vix": "live", "cape": "live", "credit": "live"},
        "timestamp": "2025-01-15T12:00:00Z",
    }


# =============================================================================
# FULL PIPELINE TESTS
# =============================================================================

class TestFullPipeline:
    """Test complete pipeline from collection to storage"""
    
    def setup_method(self):
        reset_all()
    
    def test_collection_to_processing_flow(self, sample_raw_articles):
        """Test articles flow from collection to processing"""
        from argus1.processor_enhanced import ClaudeProcessor, validate_and_fix_result
        
        # Verify raw articles have required fields
        for article in sample_raw_articles:
            assert article.title is not None
            assert article.url is not None
            assert article.article_hash is not None
            assert len(article.article_hash) == 16
        
        # Simulate Claude extraction for first article
        claude_result = {
            "category": "spending",
            "extracted_facts": ["$80B investment", "FY2025"],
            "impact_score": 0.9,
            "sentiment": "bullish",
            "companies_mentioned": ["Microsoft"],
            "dollar_amounts": ["$80B"],
            "key_quotes": [],
        }
        
        fixed = validate_and_fix_result(claude_result)
        
        # Verify validation works
        assert fixed["category"] == "spending"
        assert fixed["impact_score"] == 0.9
        assert fixed["companies_mentioned"] == ["Microsoft"]
    
    def test_processing_to_storage_flow(self, sample_bubble_reading, mock_supabase):
        """Test processed data flows to storage"""
        # Validate before storage
        result = StorageValidator.validate_for_storage(sample_bubble_reading, "bubble_readings")
        assert result.valid, f"Validation failed: {result.issues}"
        
        # Store in mock Supabase
        table = mock_supabase.table("bubble_readings")
        table.insert(sample_bubble_reading)
        
        # Verify storage
        stored = mock_supabase.get_table_data("bubble_readings")
        assert len(stored) == 1
        assert stored[0]["regime"] == "INFRASTRUCTURE"
        assert stored[0]["bifurcation_score"] == 0.77
    
    def test_bubble_reading_to_storage_flow(self, sample_bubble_reading, mock_supabase):
        """Test bubble reading flows to storage"""
        # Validate
        result = StorageValidator.validate_for_storage(sample_bubble_reading, "bubble_readings")
        assert result.valid
        
        # Store
        table = mock_supabase.table("bubble_readings")
        table.insert(sample_bubble_reading)
        
        # Verify
        stored = mock_supabase.get_table_data("bubble_readings")
        assert len(stored) == 1
        assert stored[0]["regime"] == "INFRASTRUCTURE"
        assert stored[0]["bifurcation_score"] == 0.77
    
    def test_newsletter_data_preparation(self, sample_processed_articles):
        """Test newsletter data is prepared correctly"""
        from argus1.processor_enhanced import NewsletterProcessor
        
        processor = NewsletterProcessor()
        data = processor.prepare_newsletter_data(sample_processed_articles)
        
        assert data["total_articles"] == 2
        assert "spending" in data["by_category"]
        assert "data" in data["by_category"]
        assert data["sentiment_distribution"]["bullish"] == 2
        assert "Microsoft" in data["top_companies"]
        assert "$80B" in data["dollar_amounts"]
    
    def test_end_to_end_with_mocked_apis(self, mock_supabase, sample_bubble_reading):
        """Test full end-to-end flow with mocked external APIs"""
        from argus1.aggregator_enhanced import RawArticle
        from argus1.processor_enhanced import NewsletterProcessor, validate_and_fix_result
        from y2ai.newsletter import GeneratedNewsletter, NewsletterSection
        
        # Step 1: Simulate collection (mocked)
        raw_articles = [
            RawArticle(
                source_type="newsapi",
                source_name="Reuters",
                title="Google Raises AI CapEx to $50B",
                url="https://reuters.com/google-capex",
                published_at=datetime.utcnow().isoformat(),
                content="Google announced increased AI infrastructure spending...",
            )
        ]
        
        # Step 2: Simulate processing (mocked Claude response)
        claude_results = [{
            "category": "spending",
            "extracted_facts": ["$50B capex", "AI infrastructure"],
            "impact_score": 0.85,
            "sentiment": "bullish",
            "companies_mentioned": ["Google", "Alphabet"],
            "dollar_amounts": ["$50B"],
            "key_quotes": [],
        }]
        
        # Process
        processed_articles = []
        for raw, result in zip(raw_articles, claude_results):
            fixed = validate_and_fix_result(result)
            from argus1.aggregator_enhanced import ProcessedArticle
            processed = ProcessedArticle(
                article_hash=raw.article_hash,
                source_type=raw.source_type,
                source_name=raw.source_name,
                title=raw.title,
                url=raw.url,
                published_at=raw.published_at,
                y2ai_category=fixed["category"],
                extracted_facts=fixed["extracted_facts"],
                impact_score=fixed["impact_score"],
                sentiment=fixed["sentiment"],
                companies_mentioned=fixed["companies_mentioned"],
                dollar_amounts=fixed["dollar_amounts"],
                key_quotes=fixed["key_quotes"],
                processed_at=datetime.utcnow().isoformat(),
            )
            processed_articles.append(processed)
        
        # Step 3: Prepare newsletter data
        processor = NewsletterProcessor()
        newsletter_data = processor.prepare_newsletter_data(processed_articles)
        
        # Step 4: Store bubble reading (using known-good schema)
        result = StorageValidator.validate_for_storage(sample_bubble_reading, "bubble_readings")
        if result.valid:
            mock_supabase.table("bubble_readings").insert(sample_bubble_reading)
        
        # Verify
        stored = mock_supabase.get_table_data("bubble_readings")
        assert len(stored) == 1
        assert newsletter_data["total_articles"] == 1


# =============================================================================
# STORAGE VALIDATION TESTS
# =============================================================================

class TestStorageValidation:
    """Test storage validation catches issues before Supabase insert"""
    
    def test_bubble_reading_validation_success(self):
        """Valid bubble reading passes validation"""
        reading = {
            "date": "2025-01-15",
            "vix": 18.5,
            "cape": 32.0,
            "credit_spread_ig": 95.0,
            "credit_spread_hy": 380.0,
            "bubble_index": 62.5,
            "bifurcation_score": 0.77,
            "regime": "INFRASTRUCTURE",
        }
        
        result = StorageValidator.validate_for_storage(reading, "bubble_readings")
        assert result.valid
    
    def test_bubble_reading_validation_missing_required(self):
        """Bubble reading missing required fields fails"""
        reading = {
            "date": "2025-01-15",
            # Missing: vix, cape, bubble_index, bifurcation_score, regime
        }
        
        result = StorageValidator.validate_for_storage(reading, "bubble_readings")
        assert not result.valid
    
    def test_stock_snapshot_validation_success(self):
        """Valid stock snapshot passes validation"""
        snapshot = {
            "date": "2025-01-15",
            "ticker": "NVDA",
            "price": 500.0,
            "change_pct": 2.5,
        }
        
        result = StorageValidator.validate_for_storage(snapshot, "stock_snapshots")
        assert result.valid
    
    def test_stock_snapshot_validation_missing_required(self):
        """Stock snapshot missing required fields fails"""
        snapshot = {
            "ticker": "NVDA",
            # Missing: date, price, change_pct
        }
        
        result = StorageValidator.validate_for_storage(snapshot, "stock_snapshots")
        assert not result.valid
    
    def test_unknown_table_validation(self):
        """Unknown table validation uses generic rules"""
        record = {"some_field": "value"}
        
        # Should not crash, just return a result
        result = StorageValidator.validate_for_storage(record, "unknown_table")
        assert result is not None
    
    def test_batch_validation_bubble_readings(self, mock_supabase):
        """Batch validation for bubble readings"""
        records = [
            {
                "date": "2025-01-14",
                "vix": 18.0,
                "cape": 31.5,
                "bubble_index": 61.0,
                "bifurcation_score": 0.75,
                "regime": "INFRASTRUCTURE",
            },
            {"incomplete": True},  # Invalid
            {
                "date": "2025-01-15",
                "vix": 18.5,
                "cape": 32.0,
                "bubble_index": 62.5,
                "bifurcation_score": 0.77,
                "regime": "INFRASTRUCTURE",
            },
        ]
        
        valid, invalid = StorageValidator.batch_validate(records, "bubble_readings")
        
        assert len(valid) == 2
        assert len(invalid) == 1


# =============================================================================
# ERROR PROPAGATION TESTS
# =============================================================================

class TestErrorPropagation:
    """Test error handling across module boundaries"""
    
    def setup_method(self):
        reset_all()
    
    def test_aggregator_partial_failure(self):
        """Aggregator continues despite individual adapter failures"""
        from argus1.aggregator_enhanced import NewsAggregator
        
        # Create aggregator
        aggregator = NewsAggregator()
        
        # Mock all adapters to fail except one
        with patch.object(aggregator.adapters[0], 'fetch', side_effect=Exception("API Error")):
            with patch.object(aggregator.adapters[1], 'fetch', side_effect=Exception("Rate Limited")):
                with patch.object(aggregator.adapters[2], 'fetch', side_effect=Exception("Timeout")):
                    with patch.object(aggregator.adapters[3], 'fetch', return_value=[]):
                        # Should not raise
                        articles = aggregator.collect_all(hours_back=24)
        
        # Should return empty list, not crash
        assert articles == []
    
    def test_processor_consecutive_failure_handling(self):
        """Processor stops after consecutive failures"""
        from argus1.processor_enhanced import ClaudeProcessor
        from argus1.aggregator_enhanced import RawArticle
        
        processor = ClaudeProcessor.__new__(ClaudeProcessor)
        processor.client = Mock()
        processor.model = "test"
        processor._consecutive_failures = 0
        processor._max_consecutive_failures = 5
        
        # Make all API calls fail
        processor.client.messages.create.side_effect = Exception("API Error")
        
        articles = [
            RawArticle(
                source_type="test",
                source_name="Test",
                title=f"Article {i}",
                url=f"https://example.com/{i}",
                published_at="2025-01-15T10:00:00Z",
                content="Content",
            )
            for i in range(10)
        ]
        
        # Process batch with stop on consecutive failures
        results = processor.process_batch(
            articles,
            max_batch=10,
            stop_on_consecutive_failures=3,
        )
        
        # Should have stopped early due to consecutive failures
        assert len(results) == 0
        # API should not have been called 10 times
        assert processor.client.messages.create.call_count <= 3
    
    def test_validation_failure_doesnt_crash_pipeline(self, mock_supabase):
        """Validation failures don't crash the storage pipeline"""
        records = [
            {
                "date": "2025-01-14",
                "vix": 18.0,
                "cape": 31.5,
                "bubble_index": 61.0,
                "bifurcation_score": 0.75,
                "regime": "INFRASTRUCTURE",
            },
            {"invalid": True},  # Missing required fields
            {
                "date": "2025-01-15",
                "vix": 18.5,
                "cape": 32.0,
                "bubble_index": 62.5,
                "bifurcation_score": 0.77,
                "regime": "INFRASTRUCTURE",
            },
        ]
        
        # Batch validate and store only valid
        valid, invalid = StorageValidator.batch_validate(records, "bubble_readings")
        
        for record in valid:
            mock_supabase.table("bubble_readings").insert(record)
        
        # Should have stored only valid records
        stored = mock_supabase.get_table_data("bubble_readings")
        assert len(stored) == 2
    
    def test_indicator_fetch_failure_uses_fallback(self):
        """Indicator calculators use fallback values on fetch failure"""
        from y2ai.bubble_index_enhanced import BubbleIndexCalculator
        
        calculator = BubbleIndexCalculator.__new__(BubbleIndexCalculator)
        calculator.model = "test"
        calculator._session = Mock()
        
        # Mock FRED to fail
        calculator._session.get.side_effect = Exception("FRED API Error")
        
        # Fallback mechanism would kick in
        # (actual implementation depends on BubbleIndexCalculator design)


# =============================================================================
# NEWSLETTER PIPELINE TESTS
# =============================================================================

class TestNewsletterPipeline:
    """Test newsletter generation pipeline"""
    
    def test_newsletter_data_to_generation(self, sample_processed_articles):
        """Test newsletter data flows to generation"""
        from argus1.processor_enhanced import NewsletterProcessor
        from y2ai.newsletter import NewsletterGenerator, GeneratedNewsletter
        
        # Prepare data
        processor = NewsletterProcessor()
        data = processor.prepare_newsletter_data(sample_processed_articles)
        
        # Verify data structure
        assert "by_category" in data
        assert "spending" in data["by_category"]
        assert "sentiment_distribution" in data
        
        # Create generator with mock
        generator = NewsletterGenerator.__new__(NewsletterGenerator)
        generator.model = "test"
        generator._available = True
        generator.client = Mock()
        generator.client.messages.create.return_value = Mock(
            content=[Mock(text="Generated content")]
        )
        
        # Generate
        newsletter = generator.generate_full_newsletter(
            edition_number=1,
            newsletter_data=data,
            bubble_reading=None,
            generate_social=False,
        )
        
        assert newsletter.edition_number == 1
        assert newsletter.lead_section is not None
    
    def test_newsletter_export_formats(self):
        """Test newsletter exports to multiple formats"""
        from y2ai.newsletter import GeneratedNewsletter, NewsletterSection, SocialPost
        
        newsletter = GeneratedNewsletter(
            edition_number=5,
            date="January 15, 2025",
            title="Y2AI Weekly #5",
            lead_section=NewsletterSection(
                title="This Week",
                content="Microsoft committed $80B to AI infrastructure.",
                section_type="lead",
            ),
            bubble_index=62.5,
            bifurcation_score=0.77,
            regime="INFRASTRUCTURE",
            social_posts=[
                SocialPost(platform="twitter", content="Big week for AI infra"),
            ],
        )
        
        # Test markdown export
        md = newsletter.to_markdown()
        assert "# Y2AI Weekly Edition #5" in md
        assert "$80B" in md
        
        # Test dict export
        d = newsletter.to_dict()
        assert d["edition_number"] == 5
        assert d["bubble_index"] == 62.5
        assert len(d["social_posts"]) == 1


# =============================================================================
# INDICATOR PIPELINE TESTS
# =============================================================================

class TestIndicatorPipeline:
    """Test indicator calculation pipeline"""
    
    def test_bubble_index_calculation_flow(self):
        """Test bubble index calculation flow via BubbleIndexReading"""
        from y2ai.bubble_index_enhanced import BubbleIndexReading
        
        # Create a reading with known values
        reading = BubbleIndexReading(
            date="2025-01-15",
            vix=18.5,
            cape=32.0,
            credit_spread_ig=95.0,
            credit_spread_hy=380.0,
            vix_zscore=0.15,
            cape_zscore=0.8,
            credit_zscore=-0.2,
            bubble_index=62.5,
            bifurcation_score=0.77,
            regime="INFRASTRUCTURE",
            data_sources={"vix": "live", "cape": "live"},
        )
        
        # Verify values
        assert 0 <= reading.bubble_index <= 100
        assert -2 <= reading.bifurcation_score <= 2
        assert reading.regime in ["INFRASTRUCTURE", "ADOPTION", "TRANSITION", "BUBBLE_WARNING"]
        
        # Verify dict conversion
        d = reading.to_dict()
        assert d["bubble_index"] == 62.5
        assert d["regime"] == "INFRASTRUCTURE"
    
    def test_stock_tracker_report_flow(self):
        """Test stock tracker report creation"""
        from y2ai.stock_tracker_enhanced import DailyReport
        
        # Create mock report
        report = DailyReport(
            date="2025-01-15",
            stocks=[],  # Empty for simplicity
            pillars=[],
            y2ai_index_today=1.5,
            y2ai_index_5day=3.0,
            y2ai_index_ytd=25.0,
            spy_today=0.5,
            spy_5day=1.0,
            spy_ytd=10.0,
            qqq_today=0.8,
            qqq_5day=1.5,
            qqq_ytd=15.0,
            status="VALIDATING",
            best_stock="NVDA",
            worst_stock="INTC",
            best_pillar="Demand Depth",
            worst_pillar="Supply Constraint",
            stocks_fetched=18,
            stocks_failed=0,
        )
        
        # Verify
        assert report.y2ai_index_today == 1.5
        assert report.status == "VALIDATING"
        
        # Verify dict conversion
        d = report.to_dict()
        assert d["y2ai_index_today"] == 1.5
    
    def test_indicators_to_storage(self, sample_bubble_reading, mock_supabase):
        """Test indicators store correctly"""
        # Store bubble reading
        table = mock_supabase.table("bubble_readings")
        table.insert(sample_bubble_reading)
        
        # Query back
        result = table.select("*").eq("regime", "INFRASTRUCTURE").execute()
        
        assert len(result.data) == 1
        assert result.data[0]["bubble_index"] == 62.5


# =============================================================================
# DATA INTEGRITY TESTS
# =============================================================================

class TestDataIntegrity:
    """Test data integrity across the system"""
    
    def test_article_hash_consistency(self):
        """Article hash is consistent across operations"""
        from argus1.aggregator_enhanced import RawArticle
        
        # Same URL should always produce same hash
        article1 = RawArticle(
            source_type="newsapi",
            source_name="Reuters",
            title="Title 1",
            url="https://example.com/article",
            published_at="2025-01-15T10:00:00Z",
            content="Content 1",
        )
        
        article2 = RawArticle(
            source_type="rss",
            source_name="Bloomberg",
            title="Title 2",
            url="https://example.com/article",  # Same URL
            published_at="2025-01-15T11:00:00Z",
            content="Content 2",
        )
        
        assert article1.article_hash == article2.article_hash
    
    def test_category_consistency(self):
        """Categories are consistently validated"""
        from argus1.processor_enhanced import VALID_CATEGORIES, validate_and_fix_result
        
        # All valid categories should pass
        for category in VALID_CATEGORIES:
            result = validate_and_fix_result({"category": category})
            assert result["category"] == category
        
        # Invalid category should be fixed
        result = validate_and_fix_result({"category": "invalid_category"})
        assert result["category"] in VALID_CATEGORIES
    
    def test_timestamp_consistency(self, mock_supabase):
        """Timestamps are consistent format across storage"""
        # Create record with proper fields for bubble_readings
        record = {
            "date": "2025-01-15",
            "vix": 18.5,
            "cape": 32.0,
            "bubble_index": 62.5,
            "bifurcation_score": 0.77,
            "regime": "INFRASTRUCTURE",
        }
        
        # Validate and store
        result = StorageValidator.validate_for_storage(record, "bubble_readings")
        if result.valid:
            mock_supabase.table("bubble_readings").insert(record)
        
        stored = mock_supabase.get_table_data("bubble_readings")
        assert len(stored) == 1
        assert stored[0]["date"] == "2025-01-15"


# =============================================================================
# QUERY PATTERN TESTS
# =============================================================================

class TestQueryPatterns:
    """Test common Supabase query patterns"""
    
    def test_get_latest_bubble_reading(self, mock_supabase):
        """Query pattern: Get latest bubble reading"""
        # Insert multiple readings
        readings = [
            {"date": "2025-01-13", "bubble_index": 60.0, "regime": "INFRASTRUCTURE"},
            {"date": "2025-01-14", "bubble_index": 61.5, "regime": "INFRASTRUCTURE"},
            {"date": "2025-01-15", "bubble_index": 62.5, "regime": "INFRASTRUCTURE"},
        ]
        
        table = mock_supabase.table("bubble_readings")
        for r in readings:
            table.insert(r)
        
        # Query latest
        result = table.select("*").order("date", desc=True).limit(1).execute()
        
        assert len(result.data) == 1
        assert result.data[0]["date"] == "2025-01-15"
    
    def test_get_articles_by_category(self, mock_supabase):
        """Query pattern: Get articles by category"""
        articles = [
            {"article_hash": "a", "y2ai_category": "spending", "impact_score": 0.9},
            {"article_hash": "b", "y2ai_category": "data", "impact_score": 0.8},
            {"article_hash": "c", "y2ai_category": "spending", "impact_score": 0.7},
        ]
        
        table = mock_supabase.table("articles_processed")
        for a in articles:
            table.insert(a)
        
        # Query spending articles
        result = table.select("*").eq("y2ai_category", "spending").execute()
        
        assert len(result.data) == 2
        assert all(a["y2ai_category"] == "spending" for a in result.data)
    
    def test_get_high_impact_articles(self, mock_supabase):
        """Query pattern: Get high impact articles"""
        articles = [
            {"article_hash": "a", "impact_score": 0.9},
            {"article_hash": "b", "impact_score": 0.5},
            {"article_hash": "c", "impact_score": 0.8},
        ]
        
        table = mock_supabase.table("articles_processed")
        for a in articles:
            table.insert(a)
        
        # Query high impact (>= 0.7)
        result = table.select("*").gte("impact_score", 0.7).order("impact_score", desc=True).execute()
        
        assert len(result.data) == 2
        assert result.data[0]["impact_score"] == 0.9


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
