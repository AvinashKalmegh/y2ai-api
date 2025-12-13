# ARGUS+Y2AI Phase 4: RSS Feed Health

## Summary

Phase 4 added a comprehensive RSS feed health monitoring system to detect dead feeds, track availability over time, and recommend replacements. The module expands the feed registry from 13 to 35 feeds and provides CLI tooling for ongoing maintenance.

## What Was Built

### 1. Feed Health Module (`argus1/feed_health.py`)

A complete feed monitoring system with:

**Feed Registry**
- Expanded from 13 to 35 curated RSS feeds
- Organized by category: technology, business, AI, infrastructure, semiconductors, cloud, energy, government, research
- Priority levels (high/medium/low) for filtering
- Alternative URLs for each feed when available

**FeedHealthChecker**
- Parallel checking with configurable workers
- HTTP status validation
- XML/RSS parsing validation via feedparser
- Content freshness detection (flags feeds with no updates in 7+ days)
- Response time measurement
- Redirect detection and tracking
- Alternative URL testing when primary fails

**Status Classification**
- HEALTHY: Feed working normally
- DEGRADED: Feed works but has issues (slow, stale, empty)
- REDIRECTED: URL has permanently moved
- UNAVAILABLE: Temporarily down (5xx, timeout, SSL errors)
- DEAD: Permanently gone (404, DNS failure, connection refused)

**FeedHealthReport**
- Summary statistics by status
- Category-level health breakdown
- Recommendations for feeds to remove or update
- Human-readable print_summary() method
- JSON export for automation

**FeedHealthHistory**
- Persists check results to JSON file
- Tracks health over 30-day window
- Calculates trends (improving/stable/degrading)
- Identifies consistently problematic feeds

### 2. Test Suite (`tests/test_feed_health.py`)

32 test cases covering:

| Category | Tests | Coverage |
|----------|-------|----------|
| FeedHealthCheck model | 6 | Status usability, serialization |
| FeedHealthReport | 3 | Summary calculation, recommendations |
| FeedHealthChecker | 13 | Status detection, error handling |
| FeedHealthHistory | 6 | Persistence, trends, problem detection |
| Integration | 4 | Filtering, exports |

## Feed Registry

### By Category

| Category | Count | Priority High |
|----------|-------|---------------|
| Technology | 6 | 3 |
| Business | 5 | 4 |
| AI | 6 | 5 |
| Infrastructure | 3 | 2 |
| Semiconductors | 3 | 1 |
| Cloud | 3 | 3 |
| Energy | 3 | 1 |
| Government | 3 | 0 |
| Research | 2 | 1 |

### Sample Feeds

**AI Category (High Priority)**
- mit_ai: MIT AI News
- openai_blog: OpenAI Blog
- google_ai_blog: Google AI Blog
- deepmind_blog: DeepMind Blog
- anthropic_news: Anthropic News

**Infrastructure Category**
- datacenter_knowledge: Data Center Knowledge
- datacenter_dynamics: Data Center Dynamics
- datacenter_frontier: Data Center Frontier

**Business Category**
- bloomberg_tech: Bloomberg Technology
- wsj_tech: WSJ Tech
- ft_tech: Financial Times Tech

## File Structure

```
argus_y2ai/
├── argus1/
│   ├── aggregator_enhanced.py  # (unchanged)
│   ├── processor_enhanced.py   # (unchanged)
│   └── feed_health.py          # NEW: Feed health monitoring
├── tests/
│   └── test_feed_health.py     # NEW: 32 feed health tests
```

## Usage

### CLI Examples

```bash
# Check all feeds
python -m argus1.feed_health

# Check only high-priority feeds
python -m argus1.feed_health --priority high

# Check specific category
python -m argus1.feed_health --category ai

# Check single feed
python -m argus1.feed_health --feed mit_ai

# Export working feeds to Python file
python -m argus1.feed_health --export working_feeds.py

# Output as JSON
python -m argus1.feed_health --json

# List all registered feeds
python -m argus1.feed_health --list-feeds
```

### Programmatic Usage

```python
from argus1.feed_health import FeedHealthChecker, FeedHealthHistory

# Check all feeds
checker = FeedHealthChecker()
report = checker.check_all()
report.print_summary()

# Check specific categories
report = checker.check_all(categories=["ai", "technology"])

# Check single feed
check = checker.check_single("mit_ai")
print(f"Status: {check.status.value}")
print(f"Response time: {check.response_time_ms}ms")

# Get only working feeds (for aggregator update)
working = checker.get_working_feeds()
# Returns: {"mit_ai": "https://...", "openai_blog": "https://...", ...}

# Track history over time
history = FeedHealthHistory("feed_health.json")
history.record(report)

# Get trend for specific feed
trend = history.get_trend("mit_ai")
# Returns: {"current_status": "healthy", "healthy_pct": 95.0, "trend": "stable"}

# Find consistently broken feeds
problems = history.get_problem_feeds()
```

### Updating Aggregator Feeds

```python
from argus1.feed_health import FeedHealthChecker

# Get working feeds and export
checker = FeedHealthChecker()
code = checker.export_working_feeds("updated_feeds.py")

# Or get dict directly
working = checker.get_working_feeds()

# Update aggregator's RSSAdapter.feeds with working dict
```

## Sample Output

```
============================================================
RSS FEED HEALTH REPORT
============================================================
Generated: 2025-01-15T12:00:00

Overall Status:
  ✓ Healthy:     28
  ~ Degraded:    4
  ! Unavailable: 2
  ✗ Dead:        1
  Total:         35

By Category:
  ai: 5/6 healthy
  business: 4/5 healthy
  cloud: 3/3 healthy
  energy: 2/3 healthy
  government: 2/3 healthy
  infrastructure: 3/3 healthy
  research: 2/2 healthy
  semiconductors: 3/3 healthy
  technology: 4/6 healthy

⚠ Feeds to Remove (1):
  - defunct_feed

↻ Feeds to Update (2):
  - moved_feed: http://old.url
    → http://new.url
  - redirected_feed: http://original
    → http://final
============================================================
```

## Key Design Decisions

1. **Parallel Checking**: Uses ThreadPoolExecutor for concurrent feed checks (default 10 workers). A full registry check completes in ~30 seconds rather than 5+ minutes.

2. **Conservative Status**: A feed must fail multiple criteria to be marked DEAD. Temporary issues (timeouts, 5xx) are UNAVAILABLE, which allows retry on next run.

3. **Staleness Detection**: Feeds with no items newer than 7 days are flagged DEGRADED. This catches abandoned feeds that still return 200 OK.

4. **Alternative URLs**: Each feed can specify fallback URLs. When the primary fails, alternatives are tested automatically.

5. **History Trimming**: Only 30 days of history is retained to prevent unbounded file growth while still enabling trend analysis.

## Integration with Aggregator

The feed health checker can automatically generate an updated feed list for the RSSAdapter:

```python
# In maintenance script
from argus1.feed_health import FeedHealthChecker
from argus1.aggregator_enhanced import RSSAdapter

checker = FeedHealthChecker()
working_feeds = checker.get_working_feeds()

# RSSAdapter can be initialized with custom feeds
rss = RSSAdapter()
rss.feeds = working_feeds  # Override default feeds
```

## Recommended Maintenance Schedule

- **Daily**: Run `check_all()` and record to history
- **Weekly**: Review `FeedHealthReport.feeds_to_remove` and `feeds_to_update`
- **Monthly**: Update FEED_REGISTRY based on accumulated data, add new sources

## Next Steps (Phase 5: Newsletter Generation)

The newsletter generation TODO in the orchestrator needs Claude API integration for:
1. Article summarization
2. Trend synthesis
3. Y2AI framework commentary
4. Social media post generation
