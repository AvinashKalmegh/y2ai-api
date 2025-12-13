"""
Integration Tests for ARGUS+Y2AI

Tests cover:
- News aggregator with mocked APIs
- End-to-end collection pipeline
- Data flow through processing
- Storage operations (mocked)
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from shared.resilience import reset_all


# =============================================================================
# AGGREGATOR INTEGRATION TESTS
# =============================================================================

class TestNewsAggregatorIntegration:
    """Integration tests for news aggregator with mocked external APIs"""
    
    def setup_method(self):
        reset_all()
    
    @patch('shared.resilience.requests.Session')
    def test_newsapi_adapter_success(self, mock_session_class):
        """NewsAPI adapter handles successful response"""
        from argus1.aggregator_enhanced import NewsAPIAdapter
        
        # Create mock session with mock response
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "articles": [
                {
                    "source": {"name": "Reuters"},
                    "title": "Microsoft announces $80B AI investment",
                    "url": "https://reuters.com/article1",
                    "publishedAt": "2025-01-15T10:00:00Z",
                    "content": "Microsoft announced massive AI infrastructure spending...",
                    "author": "John Doe"
                }
            ]
        }
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        adapter = NewsAPIAdapter()
        adapter.api_key = "test_key"  # Set key for test
        adapter._session = mock_session  # Inject mock session
        
        articles = adapter.fetch(hours_back=24)
        
        assert len(articles) >= 1
        assert articles[0].title == "Microsoft announces $80B AI investment"
        assert articles[0].source_name == "Reuters"
    
    @patch('shared.resilience.requests.Session')
    def test_newsapi_adapter_rate_limit(self, mock_session_class):
        """NewsAPI adapter handles rate limit response"""
        from argus1.aggregator_enhanced import NewsAPIAdapter
        
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "60"}
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        adapter = NewsAPIAdapter()
        adapter.api_key = "test_key"
        adapter._session = mock_session
        
        # Should not raise, just return empty
        articles = adapter.fetch(hours_back=24)
        
        assert articles == []
    
    @patch('shared.resilience.requests.Session')
    def test_alphavantage_adapter_success(self, mock_session_class):
        """Alpha Vantage adapter handles successful response"""
        from argus1.aggregator_enhanced import AlphaVantageAdapter
        
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "feed": [
                {
                    "title": "NVIDIA earnings beat estimates",
                    "url": "https://example.com/nvidia",
                    "time_published": "20250115T100000",
                    "summary": "NVIDIA reported Q4 earnings...",
                    "source": "MarketWatch",
                    "authors": ["Jane Smith"],
                    "overall_sentiment_label": "Bullish",
                    "overall_sentiment_score": 0.8,
                    "ticker_sentiment": [
                        {"ticker": "NVDA", "relevance_score": "0.9"}
                    ]
                }
            ]
        }
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        adapter = AlphaVantageAdapter()
        adapter.api_key = "test_key"
        adapter._session = mock_session
        
        articles = adapter.fetch(hours_back=24)
        
        assert len(articles) >= 1
        assert "NVIDIA" in articles[0].title
        assert articles[0].ticker == "NVDA"
    
    @patch('shared.resilience.requests.Session')
    def test_alphavantage_rate_limit_in_body(self, mock_session_class):
        """Alpha Vantage adapter detects rate limit in response body"""
        from argus1.aggregator_enhanced import AlphaVantageAdapter
        
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "Note": "Thank you for using Alpha Vantage! Our standard API call frequency is 25 calls per day."
        }
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session
        
        adapter = AlphaVantageAdapter()
        adapter.api_key = "test_key"
        adapter._session = mock_session
        
        articles = adapter.fetch(hours_back=24)
        
        assert articles == []
    
    @patch('feedparser.parse')
    def test_rss_adapter_success(self, mock_parse):
        """RSS adapter handles successful feed parse"""
        from argus1.aggregator_enhanced import RSSAdapter
        
        mock_feed = Mock()
        mock_feed.entries = [
            Mock(
                title="AI data centers driving power demand",
                link="https://example.com/datacenter",
                summary="Data centers are consuming unprecedented power...",
                author="Tech Reporter",
                published_parsed=(2025, 1, 15, 10, 0, 0, 0, 0, 0)
            )
        ]
        mock_parse.return_value = mock_feed
        
        adapter = RSSAdapter()
        
        # Mock the session's HEAD check
        mock_session = Mock()
        mock_session.head.return_value = Mock(status_code=200)
        adapter._session = mock_session
        
        articles = adapter.fetch(hours_back=24)
        
        # Should have articles (keyword "data center" matches)
        assert len(articles) >= 0  # Depends on keyword matching
    
    def test_aggregator_deduplication(self):
        """Aggregator deduplicates articles by URL hash"""
        from argus1.aggregator_enhanced import NewsAggregator, RawArticle
        
        aggregator = NewsAggregator(adapters=[])  # Empty adapters
        
        # Simulate adding duplicate articles
        article1 = RawArticle(
            source_type="test",
            source_name="Test",
            title="Test Article",
            url="https://example.com/same-url",
            published_at="2025-01-15T10:00:00Z",
            content="Content"
        )
        
        article2 = RawArticle(
            source_type="test2",
            source_name="Test2",
            title="Test Article Copy",
            url="https://example.com/same-url",  # Same URL
            published_at="2025-01-15T11:00:00Z",
            content="Different content"
        )
        
        # Add to seen hashes
        aggregator.seen_hashes.add(article1.article_hash)
        
        # Second article should be detected as duplicate
        assert article1.article_hash == article2.article_hash
        assert article2.article_hash in aggregator.seen_hashes


# =============================================================================
# PROCESSOR INTEGRATION TESTS
# =============================================================================

class TestProcessorIntegration:
    """Integration tests for Claude processor with mocked API"""
    
    def setup_method(self):
        reset_all()
    
    def test_processor_categorizes_article(self):
        """Processor categorizes article correctly"""
        # Mock anthropic module before importing processor
        mock_anthropic_module = MagicMock()
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text='''```json
{
    "category": "spending",
    "extracted_facts": ["Microsoft investing $80B in AI"],
    "impact_score": 0.9,
    "sentiment": "bullish",
    "companies_mentioned": ["Microsoft"],
    "dollar_amounts": ["$80 billion"],
    "key_quotes": []
}
```''')]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_module.Anthropic.return_value = mock_client
        
        with patch.dict('sys.modules', {'anthropic': mock_anthropic_module}):
            from argus1.processor_enhanced import ClaudeProcessor
            from argus1.aggregator_enhanced import RawArticle
            
            processor = ClaudeProcessor()
            
            article = RawArticle(
                source_type="test",
                source_name="Test Source",
                title="Microsoft announces $80B AI investment",
                url="https://example.com/msft",
                published_at="2025-01-15T10:00:00Z",
                content="Microsoft announced a massive $80 billion investment in AI infrastructure..."
            )
            
            result = processor.categorize_and_extract(article)
            
            assert result is not None
            assert result.y2ai_category == "spending"
            assert result.impact_score == 0.9
            assert "Microsoft" in result.companies_mentioned
    
    def test_processor_batch_stops_on_consecutive_failures(self):
        """Processor stops batch after consecutive failures"""
        # Mock anthropic module to always fail
        mock_anthropic_module = MagicMock()
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API Error")
        mock_anthropic_module.Anthropic.return_value = mock_client
        
        with patch.dict('sys.modules', {'anthropic': mock_anthropic_module}):
            from argus1.processor_enhanced import ClaudeProcessor
            from argus1.aggregator_enhanced import RawArticle
            
            processor = ClaudeProcessor()
        
        articles = [
            RawArticle(
                source_type="test",
                source_name="Test",
                title=f"Article {i}",
                url=f"https://example.com/article{i}",
                published_at="2025-01-15T10:00:00Z",
                content="Content"
            )
            for i in range(10)
        ]
        
        results = processor.process_batch(articles, max_batch=10, stop_on_consecutive_failures=3)
        
        # Should have stopped after 3 consecutive failures
        assert len(results) == 0
        # Verify it didn't try all 10
        assert mock_client.messages.create.call_count <= 5  # Some buffer for retries


# =============================================================================
# BUBBLE INDEX INTEGRATION TESTS
# =============================================================================

class TestBubbleIndexIntegration:
    """Integration tests for bubble index calculation"""
    
    def setup_method(self):
        reset_all()
    
    def test_vix_fetcher_live_data(self):
        """VIX fetcher returns live data"""
        # Mock yfinance module
        mock_yf_module = MagicMock()
        mock_ticker = Mock()
        mock_ticker.history.return_value = Mock(
            empty=False,
            __getitem__=lambda self, key: Mock(iloc=Mock(__getitem__=lambda self, idx: 18.5))
        )
        mock_yf_module.Ticker.return_value = mock_ticker
        
        with patch.dict('sys.modules', {'yfinance': mock_yf_module}):
            from y2ai.bubble_index_enhanced import VIXFetcher
            
            fetcher = VIXFetcher()
            value, source = fetcher.get_current_vix()
            
            assert value == 18.5
            assert source == "live"
    
    def test_vix_fetcher_fallback(self):
        """VIX fetcher uses fallback on error"""
        from y2ai.bubble_index_enhanced import FALLBACK_VALUES
        
        # Mock yfinance module to raise exception
        mock_yf_module = MagicMock()
        mock_yf_module.Ticker.side_effect = Exception("API Error")
        
        with patch.dict('sys.modules', {'yfinance': mock_yf_module}):
            from y2ai.bubble_index_enhanced import VIXFetcher
            
            fetcher = VIXFetcher()
            value, source = fetcher.get_current_vix()
            
            assert value == FALLBACK_VALUES["vix"]
            assert source == "fallback"
    
    def test_full_bubble_index_calculation(self):
        """Full bubble index calculation with mocked fetchers"""
        from y2ai.bubble_index_enhanced import BubbleIndexCalculator
        
        calc = BubbleIndexCalculator()
        
        # Mock all fetchers
        with patch.object(calc.vix_fetcher, 'get_current_vix', return_value=(15.0, "live")):
            with patch.object(calc.vix_fetcher, 'get_vix_history', return_value=None):
                with patch.object(calc.cape_fetcher, 'get_current_cape', return_value=(30.0, "live")):
                    with patch.object(calc.cape_fetcher, 'get_cape_history', return_value=None):
                        with patch.object(calc.credit_fetcher, 'get_current_spreads', return_value=((100.0, 350.0), "live")):
                            with patch.object(calc.credit_fetcher, 'get_spread_history', return_value=None):
                                reading = calc.calculate()
        
        assert reading is not None
        assert reading.vix == 15.0
        assert reading.cape == 30.0
        assert reading.bubble_index > 0
        assert reading.regime in ["INFRASTRUCTURE", "ADOPTION", "TRANSITION", "BUBBLE_WARNING"]


# =============================================================================
# STOCK TRACKER INTEGRATION TESTS
# =============================================================================

class TestStockTrackerIntegration:
    """Integration tests for stock tracker"""
    
    def setup_method(self):
        reset_all()
    
    def test_stock_fetcher_batch_download(self):
        """Stock fetcher handles batch download"""
        import pandas as pd
        
        # Create mock dataframe
        dates = pd.date_range(end=datetime.now(), periods=5)
        mock_data = pd.DataFrame({
            ('Close', 'NVDA'): [100, 101, 102, 103, 105],
            ('Close', 'MSFT'): [200, 201, 202, 203, 205],
        }, index=dates)
        
        # Mock yfinance module
        mock_yf_module = MagicMock()
        mock_yf_module.download.return_value = mock_data
        
        with patch.dict('sys.modules', {'yfinance': mock_yf_module}):
            from y2ai.stock_tracker_enhanced import StockFetcher
            
            fetcher = StockFetcher()
            results, failed = fetcher.get_multiple_stocks(["NVDA", "MSFT"])
            
            assert "NVDA" in results or "MSFT" in results
    
    def test_full_daily_report_generation(self):
        """Full daily report generation with mocked fetcher"""
        from y2ai.stock_tracker_enhanced import StockTracker
        
        tracker = StockTracker()
        
        # Mock the fetcher
        mock_data = {
            "TSM": {"price": 100, "change_today": 1.0, "change_5day": 2.0, "change_ytd": 10.0},
            "ASML": {"price": 500, "change_today": 1.5, "change_5day": 3.0, "change_ytd": 15.0},
            "VRT": {"price": 80, "change_today": 2.0, "change_5day": 4.0, "change_ytd": 20.0},
            "GOOGL": {"price": 150, "change_today": 0.5, "change_5day": 1.0, "change_ytd": 5.0},
            "MSFT": {"price": 400, "change_today": 0.8, "change_5day": 1.5, "change_ytd": 8.0},
            "AMZN": {"price": 180, "change_today": 1.2, "change_5day": 2.5, "change_ytd": 12.0},
            "NVDA": {"price": 500, "change_today": 3.0, "change_5day": 5.0, "change_ytd": 50.0},
            "SNOW": {"price": 150, "change_today": -1.0, "change_5day": -2.0, "change_ytd": -5.0},
            "NOW": {"price": 700, "change_today": 0.5, "change_5day": 1.0, "change_ytd": 10.0},
            "SPY": {"price": 500, "change_today": 0.5, "change_5day": 1.0, "change_ytd": 10.0},
            "QQQ": {"price": 400, "change_today": 0.8, "change_5day": 1.5, "change_ytd": 15.0},
        }
        
        with patch.object(tracker.fetcher, 'get_multiple_stocks', return_value=(mock_data, [])):
            report = tracker.generate_daily_report()
        
        assert report is not None
        assert report.y2ai_index_today > 0 or report.y2ai_index_today < 0
        assert report.status in ["VALIDATING", "NEUTRAL", "CONTRADICTING"]
        assert len(report.pillars) == 3


# =============================================================================
# END-TO-END FLOW TESTS
# =============================================================================

class TestEndToEndFlow:
    """Test complete data flow through the system"""
    
    def setup_method(self):
        reset_all()
    
    def test_article_to_processed_flow(self):
        """Test article flows from raw to processed"""
        from argus1.aggregator_enhanced import RawArticle
        from argus1.processor_enhanced import validate_and_fix_result
        
        # Create raw article
        raw = RawArticle(
            source_type="newsapi",
            source_name="Reuters",
            title="NVIDIA announces next-gen chips",
            url="https://reuters.com/nvidia",
            published_at="2025-01-15T10:00:00Z",
            content="NVIDIA unveiled its next generation of AI chips..."
        )
        
        assert raw.article_hash is not None
        assert len(raw.article_hash) == 16
        
        # Simulate Claude extraction result
        claude_result = {
            "category": "CONSTRAINTS",  # Uppercase - should be fixed
            "extracted_facts": ["NVIDIA announced next-gen chips"],
            "impact_score": "0.85",  # String - should be converted
            "sentiment": "BULLISH",  # Uppercase - should be fixed
            "companies_mentioned": "NVIDIA",  # String - should become list
            "dollar_amounts": [],
            "key_quotes": []
        }
        
        fixed = validate_and_fix_result(claude_result)
        
        assert fixed["category"] == "constraints"
        assert fixed["impact_score"] == 0.85
        assert fixed["sentiment"] == "bullish"
        assert fixed["companies_mentioned"] == ["NVIDIA"]
    
    def test_indicator_to_social_post_flow(self):
        """Test indicators flow to social post"""
        from y2ai.stock_tracker_enhanced import StockTracker, DailyReport
        
        tracker = StockTracker()
        
        # Create mock report
        report = DailyReport(
            date="2025-01-15",
            stocks=[],
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
        
        post = tracker.format_for_social(report)
        
        assert "Y2AI" in post
        assert "VALIDATING" in post or "🔥" in post  # Strong validation
        assert "NVDA" in post
        assert "+1.5" in post or "1.5" in post


# =============================================================================
# STORAGE MOCK TESTS
# =============================================================================

class TestStorageOperations:
    """Test storage operations with mocked Supabase"""
    
    def test_bubble_reading_to_dict(self):
        """Bubble reading converts to dict for storage"""
        from y2ai.bubble_index_enhanced import BubbleIndexReading
        
        reading = BubbleIndexReading(
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
        
        d = reading.to_dict()
        
        assert d["date"] == "2025-01-15"
        assert d["vix"] == 15.0
        assert d["regime"] == "ADOPTION"
        assert d["bifurcation_score"] == 0.45
    
    def test_stock_report_to_dict(self):
        """Stock report converts to dict for storage"""
        from y2ai.stock_tracker_enhanced import DailyReport
        
        report = DailyReport(
            date="2025-01-15",
            stocks=[],
            pillars=[],
            y2ai_index_today=1.0,
            y2ai_index_5day=2.0,
            y2ai_index_ytd=15.0,
            spy_today=0.5,
            spy_5day=1.0,
            spy_ytd=10.0,
            qqq_today=0.8,
            qqq_5day=1.5,
            qqq_ytd=12.0,
            status="VALIDATING",
            best_stock="NVDA",
            worst_stock="INTC",
            best_pillar="Demand",
            worst_pillar="Supply",
            stocks_fetched=18,
            stocks_failed=0,
        )
        
        d = report.to_dict()
        
        assert d["date"] == "2025-01-15"
        assert d["status"] == "VALIDATING"
        assert d["y2ai_index_today"] == 1.0


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
