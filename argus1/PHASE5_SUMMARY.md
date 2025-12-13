# ARGUS+Y2AI Phase 5: Newsletter Generation

## Summary

Phase 5 adds Claude-powered newsletter generation to produce complete Y2AI Weekly editions from processed articles and market data. The module generates multiple sections following your prose-heavy writing style, plus social media posts for Twitter, LinkedIn, and Bluesky.

## What Was Built

### 1. Newsletter Module (`y2ai/newsletter.py`)

A complete content generation system with:

**Data Models**
- `NewsletterSection`: Individual newsletter section with title, content, type, sources
- `SocialPost`: Social media post with platform, content, hashtags, thread support
- `GeneratedNewsletter`: Complete newsletter with all sections, metadata, export methods

**NewsletterGenerator Class**
- Claude API integration with resilience patterns from Phase 1
- Writing style enforcement embedded in system prompts
- Y2AI Framework context injection for thesis-aligned commentary
- Section-by-section generation with error handling

**Generated Sections**
| Section | Purpose | Typical Length |
|---------|---------|----------------|
| Lead | Week's most important development | 2-3 paragraphs |
| Spending Analysis | CapEx and infrastructure investment | 2-3 paragraphs |
| Thesis Status | Framework assessment vs indicators | 2 paragraphs |
| Outlook | Forward-looking watch items | 1-2 paragraphs |

**Social Media Generation**
- Twitter/X thread (3-5 tweets, each <280 chars)
- LinkedIn post (150-300 words, professional tone)
- Bluesky post (<300 chars, conversational)

**Helper Functions**
- `generate_quick_summary()`: Single paragraph daily update
- `generate_single_tweet()`: One-off tweet about a headline

### 2. Test Suite (`tests/test_newsletter.py`)

36 test cases covering:

| Category | Tests | Coverage |
|----------|-------|----------|
| NewsletterSection model | 2 | Creation, serialization |
| SocialPost model | 3 | Platforms, threads |
| GeneratedNewsletter model | 5 | Creation, export formats |
| NewsletterGenerator | 17 | All generation methods |
| Helper functions | 4 | Quick summary, tweets |
| Configuration | 3 | Style guidelines, framework |
| Integration | 2 | Full workflow |

## Writing Style Enforcement

The module embeds your writing style preferences directly in Claude's system prompts:

```
Writing Style Requirements:
- Write in flowing paragraphs, NOT bullet points or lists
- Each paragraph should be 3-5 sentences
- NO single-sentence paragraphs
- Use natural transitions like "Here's the thing" or "And then"
- Integrate data into narrative flow
- Use contractions naturally (it's, that's, we're)
- Avoid corporate jargon: no "leverage", "synergy", "paradigm"
- Professional but conversational tone
```

This ensures generated content matches your established voice.

## Y2AI Framework Integration

Every generation includes framework context:

```
Bifurcation Formula: Score = 0.6×BI_normalized - 0.2×VI - 0.2×CS

Regimes:
- INFRASTRUCTURE (score > +0.5): Strong infrastructure cycle
- ADOPTION (+0.2 to +0.5): Healthy adoption phase  
- TRANSITION (-0.2 to +0.2): Watching for regime change
- BUBBLE_WARNING (< -0.2): Elevated risk signals
```

The thesis status section specifically addresses whether current evidence supports "infrastructure cycle" or "bubble" interpretation.

## File Structure

```
argus_y2ai/
├── y2ai/
│   ├── bubble_index_enhanced.py  # (unchanged)
│   ├── stock_tracker_enhanced.py # (unchanged)
│   └── newsletter.py             # NEW: Newsletter generation
├── tests/
│   └── test_newsletter.py        # NEW: 36 newsletter tests
```

## Usage

### CLI

```bash
# Generate newsletter from data files
python -m y2ai.newsletter --edition 5 \
  --data newsletter_data.json \
  --bubble bubble_reading.json \
  --output edition5.md

# Output as JSON instead of markdown
python -m y2ai.newsletter --edition 5 --format json

# Skip social media posts
python -m y2ai.newsletter --edition 5 --no-social
```

### Programmatic Usage

```python
from y2ai.newsletter import NewsletterGenerator, GeneratedNewsletter
from argus1.processor_enhanced import NewsletterProcessor

# Prepare data (from processed articles)
processor = NewsletterProcessor()
newsletter_data = processor.prepare_newsletter_data(processed_articles)

# Get current bubble reading
bubble_reading = {
    "vix": 18.5,
    "bubble_index": 62.5,
    "bifurcation_score": 0.77,
    "regime": "INFRASTRUCTURE",
    # ... other indicators
}

# Generate newsletter
generator = NewsletterGenerator()
newsletter = generator.generate_full_newsletter(
    edition_number=5,
    newsletter_data=newsletter_data,
    bubble_reading=bubble_reading,
    generate_social=True,
)

# Export
print(newsletter.to_markdown())

# Or as JSON
import json
print(json.dumps(newsletter.to_dict(), indent=2))

# Access individual sections
print(newsletter.lead_section.content)
print(newsletter.thesis_status.content)

# Get social posts
for post in newsletter.social_posts:
    print(f"{post.platform}: {post.content[:100]}...")
```

### Quick Generation Helpers

```python
from y2ai.newsletter import generate_quick_summary, generate_single_tweet

# Daily summary (1 paragraph)
summary = generate_quick_summary(
    articles=processed_articles,
    bubble_reading=bubble_reading,
)

# Single tweet about news
tweet = generate_single_tweet(
    headline="Microsoft Announces $80B AI Infrastructure Plan",
    bubble_index=62.5,
)
```

## Sample Output

### Markdown Export

```markdown
# Y2AI Weekly Edition #5
*January 15, 2025*

**Current Readings:** Bubble Index 62.5 | Bifurcation Score +0.77 | Regime: INFRASTRUCTURE

## This Week in AI Infrastructure

Microsoft's announcement of an $80 billion AI infrastructure commitment marks the 
largest single-year capex guidance in tech history. The spending, spread across 
data center construction and GPU procurement, signals continued confidence in the 
infrastructure thesis we've been tracking.

Here's the thing about this number: it represents nearly double Microsoft's 
traditional capex run rate. Combined with Google's raised guidance to $50 billion, 
we're seeing hyperscalers commit over $130 billion to AI infrastructure in a single 
quarter of announcements.

## Infrastructure Spending Analysis

The combined hyperscaler commitments paint a clear picture of infrastructure 
conviction...

## Thesis Status: INFRASTRUCTURE

With a bifurcation score of +0.77, the framework continues to support the 
infrastructure interpretation...

## What to Watch

Next week brings NVIDIA earnings on Wednesday, which will provide the clearest 
signal yet on whether demand continues to outstrip supply...
```

### Social Posts

**Twitter Thread:**
```
1/ Big week for AI infrastructure: $130B in hyperscaler commitments 📊

2/ Microsoft's $80B is nearly 2x their traditional run rate. Google raised to $50B. This isn't speculation—it's infrastructure buildout.

3/ Bubble Index at 62.5, bifurcation score +0.77. Framework says: infrastructure cycle, not bubble.

4/ Full analysis in Y2AI Weekly #5 → [link]
```

**LinkedIn:**
```
This week's AI infrastructure announcements reinforce a key distinction we track at Y2AI: the difference between genuine infrastructure investment cycles and speculative bubbles.

Microsoft's $80B commitment and Google's $50B guidance represent the largest sustained infrastructure investment we've seen since the telecom buildout of the late 1990s. But unlike the dot-com era, these investments target real capacity constraints—GPU supply, data center power, and inference infrastructure.

Our Bifurcation Framework currently reads +0.77, firmly in "Infrastructure" territory. Here's what that means for investors...
```

## Key Design Decisions

1. **Section-by-Section Generation**: Each section gets its own Claude call with tailored system prompts. This allows for better quality control and easier debugging when output doesn't match expectations.

2. **Framework Injection**: The Y2AI framework is injected into every prompt so Claude always has context for thesis-aligned commentary without needing to search or recall.

3. **Writing Style as System Prompt**: Style guidelines are embedded in system prompts rather than user prompts, giving them higher priority in Claude's attention.

4. **Graceful Degradation**: If a section fails to generate, the newsletter still completes with a placeholder rather than failing entirely.

5. **Separate Social Generation**: Social posts are generated after the main newsletter, using the completed newsletter as context for consistent messaging.

## Integration with Pipeline

```python
# Full pipeline example
from argus1.aggregator_enhanced import NewsAggregator
from argus1.processor_enhanced import ClaudeProcessor, NewsletterProcessor
from y2ai.bubble_index_enhanced import BubbleIndexCalculator
from y2ai.newsletter import NewsletterGenerator

# Collect
aggregator = NewsAggregator()
raw_articles = aggregator.collect_all(hours_back=168)  # 1 week

# Process
processor = ClaudeProcessor()
filtered = processor.quick_relevance_filter(raw_articles)
processed = processor.process_batch(filtered, max_batch=50)

# Prepare newsletter data
newsletter_proc = NewsletterProcessor()
newsletter_data = newsletter_proc.prepare_newsletter_data(processed)

# Get market indicators
calculator = BubbleIndexCalculator()
bubble_reading = calculator.calculate()

# Generate newsletter
generator = NewsletterGenerator()
newsletter = generator.generate_full_newsletter(
    edition_number=5,
    newsletter_data=newsletter_data,
    bubble_reading=bubble_reading.to_dict(),
)

# Export
with open("edition5.md", "w") as f:
    f.write(newsletter.to_markdown())
```

## Environment Requirements

```bash
# Required environment variable
export ANTHROPIC_API_KEY=your-key-here
```

## Next Steps (Phase 6: Integration Testing)

Phase 6 will add end-to-end integration tests covering:
1. Full pipeline execution with real/mock APIs
2. Supabase storage verification
3. Data flow validation from collection to newsletter
4. Error propagation testing across modules
