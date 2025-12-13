"""
Tests for the ARGUS-1 RSS Feed Health Checker

Test coverage:
- FeedHealthCheck model
- FeedHealthReport model
- FeedHealthChecker logic
- Status determination
- Alternative URL finding
- Historical tracking
"""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from argus1.feed_health import (
    FeedStatus,
    FeedHealthCheck,
    FeedHealthReport,
    FeedHealthChecker,
    FeedHealthHistory,
    FEED_REGISTRY,
)


# =============================================================================
# FEED HEALTH CHECK MODEL TESTS
# =============================================================================

class TestFeedHealthCheck:
    """Tests for FeedHealthCheck dataclass"""
    
    def test_healthy_check_is_usable(self):
        """Healthy feed should be usable"""
        check = FeedHealthCheck(
            feed_id="test",
            feed_name="Test Feed",
            url="https://example.com/feed",
            category="test",
            status=FeedStatus.HEALTHY,
            checked_at=datetime.utcnow().isoformat(),
        )
        assert check.is_usable is True
    
    def test_degraded_check_is_usable(self):
        """Degraded feed should still be usable"""
        check = FeedHealthCheck(
            feed_id="test",
            feed_name="Test Feed",
            url="https://example.com/feed",
            category="test",
            status=FeedStatus.DEGRADED,
            checked_at=datetime.utcnow().isoformat(),
        )
        assert check.is_usable is True
    
    def test_redirected_check_is_usable(self):
        """Redirected feed should still be usable"""
        check = FeedHealthCheck(
            feed_id="test",
            feed_name="Test Feed",
            url="https://example.com/feed",
            category="test",
            status=FeedStatus.REDIRECTED,
            checked_at=datetime.utcnow().isoformat(),
            redirect_url="https://example.com/new-feed",
        )
        assert check.is_usable is True
    
    def test_unavailable_check_not_usable(self):
        """Unavailable feed should not be usable"""
        check = FeedHealthCheck(
            feed_id="test",
            feed_name="Test Feed",
            url="https://example.com/feed",
            category="test",
            status=FeedStatus.UNAVAILABLE,
            checked_at=datetime.utcnow().isoformat(),
        )
        assert check.is_usable is False
    
    def test_dead_check_not_usable(self):
        """Dead feed should not be usable"""
        check = FeedHealthCheck(
            feed_id="test",
            feed_name="Test Feed",
            url="https://example.com/feed",
            category="test",
            status=FeedStatus.DEAD,
            checked_at=datetime.utcnow().isoformat(),
        )
        assert check.is_usable is False
    
    def test_to_dict(self):
        """to_dict should serialize correctly"""
        check = FeedHealthCheck(
            feed_id="test",
            feed_name="Test Feed",
            url="https://example.com/feed",
            category="test",
            status=FeedStatus.HEALTHY,
            checked_at="2025-01-15T12:00:00",
            response_time_ms=150.5,
            item_count=10,
        )
        
        d = check.to_dict()
        assert d["feed_id"] == "test"
        assert d["status"] == "healthy"
        assert d["response_time_ms"] == 150.5
        assert d["item_count"] == 10


# =============================================================================
# FEED HEALTH REPORT TESTS
# =============================================================================

class TestFeedHealthReport:
    """Tests for FeedHealthReport"""
    
    def test_report_summary(self):
        """Report should calculate summaries correctly"""
        checks = [
            FeedHealthCheck(
                feed_id="healthy1", feed_name="H1", url="http://h1", 
                category="test", status=FeedStatus.HEALTHY, checked_at="2025-01-15"
            ),
            FeedHealthCheck(
                feed_id="healthy2", feed_name="H2", url="http://h2",
                category="test", status=FeedStatus.HEALTHY, checked_at="2025-01-15"
            ),
            FeedHealthCheck(
                feed_id="dead1", feed_name="D1", url="http://d1",
                category="test", status=FeedStatus.DEAD, checked_at="2025-01-15"
            ),
        ]
        
        report = FeedHealthReport(
            generated_at="2025-01-15",
            total_feeds=3,
            healthy_count=2,
            degraded_count=0,
            unavailable_count=0,
            dead_count=1,
            checks=checks,
        )
        
        d = report.to_dict()
        assert d["summary"]["total_feeds"] == 3
        assert d["summary"]["healthy"] == 2
        assert d["summary"]["dead"] == 1
    
    def test_recommendations_for_dead_feeds(self):
        """Report should recommend removing dead feeds"""
        report = FeedHealthReport(
            generated_at="2025-01-15",
            total_feeds=1,
            healthy_count=0,
            degraded_count=0,
            unavailable_count=0,
            dead_count=1,
            checks=[],
            feeds_to_remove=["dead_feed"],
        )
        
        d = report.to_dict()
        assert "dead_feed" in d["recommendations"]["feeds_to_remove"]
    
    def test_recommendations_for_redirected_feeds(self):
        """Report should recommend updating redirected feeds"""
        report = FeedHealthReport(
            generated_at="2025-01-15",
            total_feeds=1,
            healthy_count=0,
            degraded_count=1,
            unavailable_count=0,
            dead_count=0,
            checks=[],
            feeds_to_update=[{
                "feed_id": "moved_feed",
                "old_url": "http://old",
                "new_url": "http://new",
                "reason": "Redirect",
            }],
        )
        
        d = report.to_dict()
        assert len(d["recommendations"]["feeds_to_update"]) == 1
        assert d["recommendations"]["feeds_to_update"][0]["new_url"] == "http://new"


# =============================================================================
# FEED HEALTH CHECKER TESTS
# =============================================================================

class TestFeedHealthChecker:
    """Tests for FeedHealthChecker"""
    
    @pytest.fixture
    def mock_feeds(self):
        """Simple test feed registry"""
        return {
            "test_feed": {
                "url": "https://example.com/feed.xml",
                "name": "Test Feed",
                "category": "test",
                "priority": "high",
                "alternatives": ["https://example.com/feed2.xml"],
            }
        }
    
    @pytest.fixture
    def checker(self, mock_feeds):
        """Create checker with test feeds"""
        return FeedHealthChecker(feeds=mock_feeds, timeout=5)
    
    def test_registry_loaded(self):
        """FEED_REGISTRY should have feeds defined"""
        assert len(FEED_REGISTRY) > 0
        assert "mit_ai" in FEED_REGISTRY
    
    def test_registry_structure(self):
        """Each feed should have required fields"""
        for feed_id, info in FEED_REGISTRY.items():
            assert "url" in info, f"{feed_id} missing url"
            assert "name" in info, f"{feed_id} missing name"
            assert "category" in info, f"{feed_id} missing category"
    
    def test_check_healthy_feed(self, checker):
        """Should detect healthy feed correctly"""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.history = []
        mock_response.url = "https://example.com/feed.xml"
        mock_response.text = """<?xml version="1.0"?>
        <rss version="2.0">
            <channel>
                <title>Test</title>
                <item>
                    <title>Item 1</title>
                    <pubDate>Mon, 15 Jan 2025 12:00:00 GMT</pubDate>
                </item>
            </channel>
        </rss>
        """
        
        # Patch the session on the checker instance
        checker._session = Mock()
        checker._session.get.return_value = mock_response
        
        # Mock feedparser with recent date
        with patch.object(checker, '_get_feedparser') as mock_fp:
            mock_parser = Mock()
            mock_entry = Mock()
            # Use current time so feed isn't "stale"
            now = datetime.utcnow()
            mock_entry.published_parsed = now.timetuple()[:9]
            mock_parser.parse.return_value = Mock(
                bozo=False,
                entries=[mock_entry]
            )
            mock_fp.return_value = mock_parser
            
            check = checker.check_feed("test_feed", checker.feeds["test_feed"])
            
            assert check.status == FeedStatus.HEALTHY
            assert check.item_count == 1
    
    def test_check_404_feed(self, checker):
        """Should detect dead feed (404)"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.history = []
        
        checker._session = Mock()
        checker._session.get.return_value = mock_response
        checker._session.head.return_value = Mock(status_code=404)
        
        check = checker.check_feed("test_feed", checker.feeds["test_feed"])
        
        assert check.status == FeedStatus.DEAD
        assert check.http_status == 404
    
    def test_check_server_error_feed(self, checker):
        """Should detect unavailable feed (5xx)"""
        mock_response = Mock()
        mock_response.status_code = 503
        mock_response.history = []
        
        checker._session = Mock()
        checker._session.get.return_value = mock_response
        
        check = checker.check_feed("test_feed", checker.feeds["test_feed"])
        
        assert check.status == FeedStatus.UNAVAILABLE
        assert "503" in check.error_message
    
    def test_check_timeout_feed(self, checker):
        """Should handle timeout gracefully"""
        import requests
        
        checker._session = Mock()
        checker._session.get.side_effect = requests.exceptions.Timeout("Timed out")
        
        check = checker.check_feed("test_feed", checker.feeds["test_feed"])
        
        assert check.status == FeedStatus.UNAVAILABLE
        assert "timed out" in check.error_message.lower()
    
    def test_check_connection_error_feed(self, checker):
        """Should detect dead feed on connection error"""
        import requests
        
        checker._session = Mock()
        checker._session.get.side_effect = requests.exceptions.ConnectionError(
            "Connection refused"
        )
        checker._session.head.side_effect = Exception("Also fails")
        
        check = checker.check_feed("test_feed", checker.feeds["test_feed"])
        
        assert check.status == FeedStatus.DEAD
    
    def test_check_redirect_detected(self, checker):
        """Should detect and report redirects"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.history = [Mock()]  # Has redirect history
        mock_response.url = "https://example.com/new-feed.xml"  # Different from original
        mock_response.text = "<rss><channel></channel></rss>"
        
        checker._session = Mock()
        checker._session.get.return_value = mock_response
        
        with patch.object(checker, '_get_feedparser') as mock_fp:
            mock_parser = Mock()
            mock_entry = Mock()
            # Use current time so feed isn't "stale"
            now = datetime.utcnow()
            mock_entry.published_parsed = now.timetuple()[:9]
            mock_parser.parse.return_value = Mock(bozo=False, entries=[mock_entry])
            mock_fp.return_value = mock_parser
            
            check = checker.check_feed("test_feed", checker.feeds["test_feed"])
            
            assert check.status == FeedStatus.REDIRECTED
            assert check.redirect_url == "https://example.com/new-feed.xml"
    
    def test_check_empty_feed_degraded(self, checker):
        """Should mark empty feed as degraded"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.history = []
        mock_response.url = "https://example.com/feed.xml"
        mock_response.text = "<rss><channel></channel></rss>"
        
        checker._session = Mock()
        checker._session.get.return_value = mock_response
        
        with patch.object(checker, '_get_feedparser') as mock_fp:
            mock_parser = Mock()
            mock_parser.parse.return_value = Mock(bozo=False, entries=[])
            mock_fp.return_value = mock_parser
            
            check = checker.check_feed("test_feed", checker.feeds["test_feed"])
            
            assert check.status == FeedStatus.DEGRADED
            assert check.item_count == 0
    
    def test_check_stale_feed_degraded(self, checker):
        """Should mark stale feed (no updates in 7+ days) as degraded"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.history = []
        mock_response.url = "https://example.com/feed.xml"
        mock_response.text = "<rss><channel></channel></rss>"
        
        checker._session = Mock()
        checker._session.get.return_value = mock_response
        
        with patch.object(checker, '_get_feedparser') as mock_fp:
            mock_parser = Mock()
            # Entry from 10 days ago
            old_date = datetime.utcnow() - timedelta(days=10)
            mock_entry = Mock()
            mock_entry.published_parsed = old_date.timetuple()[:9]
            mock_parser.parse.return_value = Mock(bozo=False, entries=[mock_entry])
            mock_fp.return_value = mock_parser
            
            check = checker.check_feed("test_feed", checker.feeds["test_feed"])
            
            assert check.status == FeedStatus.DEGRADED
            assert "stale" in check.error_message.lower()
    
    def test_find_working_alternative(self, checker):
        """Should find working alternative URL"""
        checker._session = Mock()
        checker._session.head.return_value = Mock(status_code=200)
        
        alt = checker._find_working_alternative(["https://alt.com/feed"])
        assert alt == "https://alt.com/feed"
    
    def test_no_working_alternative(self, checker):
        """Should return None if no alternatives work"""
        checker._session = Mock()
        checker._session.head.side_effect = Exception("All fail")
        
        alt = checker._find_working_alternative(["https://alt1.com", "https://alt2.com"])
        assert alt is None
    
    def test_check_single_unknown_feed(self, checker):
        """Should raise for unknown feed ID"""
        with pytest.raises(ValueError, match="Unknown feed"):
            checker.check_single("nonexistent_feed")


# =============================================================================
# FEED HEALTH HISTORY TESTS
# =============================================================================

class TestFeedHealthHistory:
    """Tests for FeedHealthHistory"""
    
    @pytest.fixture
    def temp_history_file(self):
        """Create temporary history file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{}')
            return f.name
    
    @pytest.fixture
    def history(self, temp_history_file):
        """Create history tracker with temp file"""
        return FeedHealthHistory(temp_history_file)
    
    def test_record_check(self, history):
        """Should record check to history"""
        check = FeedHealthCheck(
            feed_id="test_feed",
            feed_name="Test",
            url="http://test",
            category="test",
            status=FeedStatus.HEALTHY,
            checked_at=datetime.utcnow().isoformat(),
            response_time_ms=100,
        )
        
        report = FeedHealthReport(
            generated_at=datetime.utcnow().isoformat(),
            total_feeds=1,
            healthy_count=1,
            degraded_count=0,
            unavailable_count=0,
            dead_count=0,
            checks=[check],
        )
        
        history.record(report)
        
        assert "test_feed" in history._history
        assert len(history._history["test_feed"]) == 1
    
    def test_get_trend_no_history(self, history):
        """Should return error for feed with no history"""
        result = history.get_trend("nonexistent")
        assert "error" in result
    
    def test_get_trend_with_history(self, history):
        """Should calculate trend from history"""
        # Manually add history entries
        now = datetime.utcnow()
        history._history["test_feed"] = [
            {"timestamp": (now - timedelta(days=i)).isoformat(), 
             "status": "healthy", 
             "response_time_ms": 100,
             "item_count": 10,
             "error": None}
            for i in range(5)
        ]
        
        trend = history.get_trend("test_feed")
        
        assert trend["current_status"] == "healthy"
        assert trend["healthy_pct"] == 100.0
        assert trend["checks_last_7d"] == 5
    
    def test_get_problem_feeds(self, history):
        """Should identify consistently problematic feeds"""
        now = datetime.utcnow()
        
        # Good feed
        history._history["good_feed"] = [
            {"timestamp": (now - timedelta(hours=i)).isoformat(),
             "status": "healthy", "response_time_ms": 100,
             "item_count": 10, "error": None}
            for i in range(5)
        ]
        
        # Bad feed
        history._history["bad_feed"] = [
            {"timestamp": (now - timedelta(hours=i)).isoformat(),
             "status": "dead", "response_time_ms": None,
             "item_count": None, "error": "404"}
            for i in range(5)
        ]
        
        problems = history.get_problem_feeds()
        
        assert "bad_feed" in problems
        assert "good_feed" not in problems
    
    def test_history_persistence(self, temp_history_file):
        """Should persist history across instances"""
        # First instance records data
        history1 = FeedHealthHistory(temp_history_file)
        history1._history["test"] = [{"timestamp": "2025-01-15", "status": "healthy",
                                       "response_time_ms": 100, "item_count": 10, "error": None}]
        history1._save()
        
        # Second instance should load it
        history2 = FeedHealthHistory(temp_history_file)
        
        assert "test" in history2._history
    
    def test_history_trims_old_entries(self, history):
        """Should remove entries older than 30 days"""
        now = datetime.utcnow()
        
        history._history["test_feed"] = [
            {"timestamp": (now - timedelta(days=5)).isoformat(),
             "status": "healthy", "response_time_ms": 100,
             "item_count": 10, "error": None},
            {"timestamp": (now - timedelta(days=40)).isoformat(),  # Old
             "status": "healthy", "response_time_ms": 100,
             "item_count": 10, "error": None},
        ]
        
        # Recording should trigger trim
        check = FeedHealthCheck(
            feed_id="test_feed", feed_name="Test", url="http://test",
            category="test", status=FeedStatus.HEALTHY,
            checked_at=now.isoformat(),
        )
        report = FeedHealthReport(
            generated_at=now.isoformat(), total_feeds=1,
            healthy_count=1, degraded_count=0,
            unavailable_count=0, dead_count=0,
            checks=[check],
        )
        
        history.record(report)
        
        # Should have 2 entries (new + 5 days ago), not the 40-day-old one
        assert len(history._history["test_feed"]) == 2


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestFeedHealthIntegration:
    """Integration-style tests"""
    
    def test_full_report_generation(self):
        """Should generate complete report from mock feeds"""
        test_feeds = {
            "feed1": {"url": "http://feed1", "name": "Feed 1", "category": "test", "alternatives": []},
            "feed2": {"url": "http://feed2", "name": "Feed 2", "category": "test", "alternatives": []},
        }
        
        checker = FeedHealthChecker(feeds=test_feeds, timeout=1, max_workers=2)
        
        # Mock all network calls to fail fast
        with patch.object(checker, '_session') as mock_session:
            mock_session.get.side_effect = Exception("Mocked failure")
            mock_session.head.side_effect = Exception("Mocked failure")
            
            report = checker.check_all()
            
            assert report.total_feeds == 2
            assert len(report.checks) == 2
    
    def test_category_filtering(self):
        """Should filter by category"""
        test_feeds = {
            "ai_feed": {"url": "http://ai", "name": "AI", "category": "ai", "alternatives": []},
            "tech_feed": {"url": "http://tech", "name": "Tech", "category": "technology", "alternatives": []},
        }
        
        checker = FeedHealthChecker(feeds=test_feeds, timeout=1)
        
        with patch.object(checker, '_session') as mock_session:
            mock_session.get.side_effect = Exception("Mocked")
            
            report = checker.check_all(categories=["ai"])
            
            assert report.total_feeds == 1
            assert report.checks[0].feed_id == "ai_feed"
    
    def test_priority_filtering(self):
        """Should filter by priority"""
        test_feeds = {
            "high_feed": {"url": "http://high", "name": "High", "category": "test", 
                         "priority": "high", "alternatives": []},
            "low_feed": {"url": "http://low", "name": "Low", "category": "test",
                        "priority": "low", "alternatives": []},
        }
        
        checker = FeedHealthChecker(feeds=test_feeds, timeout=1)
        
        with patch.object(checker, '_session') as mock_session:
            mock_session.get.side_effect = Exception("Mocked")
            
            report = checker.check_all(priorities=["high"])
            
            assert report.total_feeds == 1
            assert report.checks[0].feed_id == "high_feed"
    
    def test_export_working_feeds(self):
        """Should export only working feeds"""
        test_feeds = {
            "working": {"url": "http://works", "name": "Works", "category": "test", "alternatives": []},
            "broken": {"url": "http://broken", "name": "Broken", "category": "test", "alternatives": []},
        }
        
        checker = FeedHealthChecker(feeds=test_feeds, timeout=1)
        
        # Mock: first feed works, second fails
        with patch.object(checker, 'check_feed') as mock_check:
            mock_check.side_effect = [
                FeedHealthCheck(
                    feed_id="working", feed_name="Works", url="http://works",
                    category="test", status=FeedStatus.HEALTHY,
                    checked_at="2025-01-15",
                ),
                FeedHealthCheck(
                    feed_id="broken", feed_name="Broken", url="http://broken",
                    category="test", status=FeedStatus.DEAD,
                    checked_at="2025-01-15",
                ),
            ]
            
            working = checker.get_working_feeds()
            
            assert "working" in working
            assert "broken" not in working


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
