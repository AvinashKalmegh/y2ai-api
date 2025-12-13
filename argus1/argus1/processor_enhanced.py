"""
ARGUS-1 PROCESSOR (ENHANCED)
Claude-based extraction and Y2AI categorization with robust error handling

Features:
- Retry logic for transient Claude API failures
- Rate limiting awareness
- JSON parsing resilience with multiple fallback strategies
- Batch processing with partial failure tolerance
"""

import os
import json
import re
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import logging

# Import from enhanced aggregator
from .aggregator_enhanced import RawArticle, ProcessedArticle

# Import resilience module
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from shared.resilience import (
    resilient_call,
    get_health_tracker,
    CircuitOpenError,
    RateLimitError,
    RetryExhaustedError,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Y2AI CATEGORY DEFINITIONS
# =============================================================================

Y2AI_CATEGORIES = {
    "spending": {
        "description": "Capital expenditure announcements, infrastructure investments, data center construction",
        "keywords": ["capex", "capital expenditure", "billion", "investment", "data center", "infrastructure"],
        "examples": ["Microsoft announces $80B AI infrastructure investment", "Google raises capex guidance"]
    },
    "constraints": {
        "description": "Supply chain constraints, GPU shortages, power limitations",
        "keywords": ["shortage", "constraint", "supply", "bottleneck", "power", "capacity"],
        "examples": ["NVIDIA H100 backlog extends to 2025", "Data centers face power constraints"]
    },
    "data": {
        "description": "Earnings data, revenue figures, margin analysis",
        "keywords": ["earnings", "revenue", "margin", "profit", "guidance", "forecast"],
        "examples": ["Google Cloud revenue up 34% YoY", "NVIDIA beats EPS estimates"]
    },
    "policy": {
        "description": "Government policy, regulations, export controls",
        "keywords": ["regulation", "policy", "government", "export", "ban", "restriction"],
        "examples": ["US tightens chip export controls to China", "EU AI Act implementation"]
    },
    "skepticism": {
        "description": "Bubble warnings, skeptical analysis, critical coverage",
        "keywords": ["bubble", "overvalued", "correction", "warning", "skeptic", "hype"],
        "examples": ["Fund managers call AI bubble biggest risk", "Analysts question AI valuations"]
    },
    "smartmoney": {
        "description": "Institutional investor moves, hedge fund positions, insider activity",
        "keywords": ["hedge fund", "institutional", "position", "bet", "insider", "allocation"],
        "examples": ["Burry takes $1.1B put position", "Institutions increase AI allocations"]
    },
    "china": {
        "description": "China-specific AI developments, US-China competition",
        "keywords": ["china", "chinese", "huawei", "baidu", "alibaba", "tencent", "beijing"],
        "examples": ["China develops domestic GPU alternative", "Huawei AI chip production"]
    },
    "energy": {
        "description": "Power consumption, energy infrastructure, sustainability",
        "keywords": ["power", "energy", "electricity", "nuclear", "renewable", "megawatt", "gigawatt"],
        "examples": ["AI data centers drive 10% power demand increase", "Microsoft signs nuclear deal"]
    },
    "adoption": {
        "description": "Enterprise AI adoption, use cases, ROI evidence",
        "keywords": ["adoption", "enterprise", "roi", "productivity", "deploy", "implement"],
        "examples": ["Enterprise AI adoption reaches 65%", "Companies report 30% productivity gains"]
    }
}

VALID_CATEGORIES = set(Y2AI_CATEGORIES.keys())
VALID_SENTIMENTS = {"bullish", "bearish", "neutral"}


# =============================================================================
# JSON PARSING HELPERS
# =============================================================================

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from Claude's response with multiple fallback strategies.
    
    Handles:
    - Clean JSON
    - JSON wrapped in markdown code blocks
    - JSON with leading/trailing text
    - Malformed JSON with common errors
    """
    if not text:
        return None
    
    # Strategy 1: Try direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Extract from markdown code blocks
    patterns = [
        r'```json\s*([\s\S]*?)\s*```',  # ```json ... ```
        r'```\s*([\s\S]*?)\s*```',       # ``` ... ```
        r'\{[\s\S]*\}',                   # Raw JSON object
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                # Clean up common issues
                cleaned = match.strip()
                cleaned = re.sub(r',\s*}', '}', cleaned)  # Trailing commas
                cleaned = re.sub(r',\s*]', ']', cleaned)  # Trailing commas in arrays
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue
    
    # Strategy 3: Find JSON-like structure
    start_idx = text.find('{')
    if start_idx != -1:
        # Find matching closing brace
        depth = 0
        for i, char in enumerate(text[start_idx:], start_idx):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start_idx:i+1])
                    except json.JSONDecodeError:
                        break
    
    return None


def validate_and_fix_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and fix common issues in Claude's extracted data.
    """
    # Fix category
    category = result.get("category", "").lower().strip()
    if category not in VALID_CATEGORIES:
        # Try to match partial category name (only if we have a non-empty category)
        matched = False
        if category:  # Only attempt matching if category is non-empty
            for valid_cat in VALID_CATEGORIES:
                if valid_cat in category or category in valid_cat:
                    category = valid_cat
                    matched = True
                    break
        if not matched:
            category = "data"  # Default fallback
    result["category"] = category
    
    # Fix sentiment
    sentiment = result.get("sentiment", "").lower().strip()
    if sentiment not in VALID_SENTIMENTS:
        result["sentiment"] = "neutral"
    else:
        result["sentiment"] = sentiment
    
    # Fix impact score
    try:
        impact = float(result.get("impact_score", 0.5))
        impact = max(0.0, min(1.0, impact))  # Clamp to 0-1
    except (ValueError, TypeError):
        impact = 0.5
    result["impact_score"] = impact
    
    # Ensure lists are lists (always set the field, even if defaulting to [])
    for field in ["extracted_facts", "companies_mentioned", "dollar_amounts", "key_quotes"]:
        value = result.get(field)
        if value is None:
            result[field] = []
        elif isinstance(value, str):
            result[field] = [value] if value else []
        elif not isinstance(value, list):
            result[field] = []
        else:
            result[field] = value  # Already a list, but ensure it's assigned
    
    return result


# =============================================================================
# CLAUDE PROCESSOR (Enhanced)
# =============================================================================

class ClaudeProcessor:
    """Process articles using Claude for extraction and categorization with resilience"""
    
    def __init__(self):
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        except ImportError:
            logger.error("anthropic package not installed")
            self.client = None
        
        self.model = "claude-sonnet-4-20250514"
        self._consecutive_failures = 0
        self._max_consecutive_failures = 5
    
    def is_available(self) -> bool:
        """Check if Claude API is available"""
        if not self.client:
            return False
        if self._consecutive_failures >= self._max_consecutive_failures:
            logger.warning(
                f"Claude processor disabled after {self._consecutive_failures} "
                "consecutive failures"
            )
            return False
        return True
    
    def _build_prompt(self, article: RawArticle) -> str:
        """Build the extraction prompt for Claude"""
        return f"""Analyze this news article for the Y2AI infrastructure investment thesis.

ARTICLE:
Title: {article.title}
Source: {article.source_name}
Published: {article.published_at}
Content: {article.content[:3000]}

CATEGORIES (select the single most relevant):
{json.dumps({k: v['description'] for k, v in Y2AI_CATEGORIES.items()}, indent=2)}

Respond with ONLY valid JSON in this exact format (no markdown, no explanation):
{{
    "category": "one of: spending, constraints, data, policy, skepticism, smartmoney, china, energy, adoption",
    "extracted_facts": ["List of 2-4 specific factual claims from the article"],
    "impact_score": 0.0 to 1.0 (how relevant is this to AI infrastructure investment thesis),
    "sentiment": "bullish, bearish, or neutral (regarding AI infrastructure investment)",
    "companies_mentioned": ["List of company names mentioned"],
    "dollar_amounts": ["List of specific dollar amounts mentioned, e.g. '$80 billion'"],
    "key_quotes": ["1-2 notable quotes from the article if any"]
}}"""

    @resilient_call(
        service_name="anthropic",
        max_retries=3,
        base_delay=2.0,
        use_circuit_breaker=True,
        use_rate_limiter=True,
        circuit_failure_threshold=5,
        circuit_reset_timeout=120,
    )
    def _call_claude(self, prompt: str) -> str:
        """Make a Claude API call with resilience"""
        import anthropic
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
            
        except anthropic.RateLimitError as e:
            raise RateLimitError("anthropic", retry_after=60)
        except anthropic.APIStatusError as e:
            if e.status_code == 529:  # Overloaded
                raise RateLimitError("anthropic", retry_after=30)
            raise
    
    def categorize_and_extract(self, article: RawArticle) -> Optional[ProcessedArticle]:
        """
        Use Claude to categorize and extract structured data from article.
        
        Returns None on failure, but tracks health metrics.
        """
        if not self.is_available():
            return None
        
        prompt = self._build_prompt(article)
        
        try:
            # Call Claude with resilience
            result_text = self._call_claude(prompt)
            
            # Parse JSON with fallback strategies
            result = extract_json_from_text(result_text)
            
            if result is None:
                logger.warning(f"Could not parse JSON for '{article.title[:50]}...'")
                logger.debug(f"Raw response: {result_text[:500]}")
                self._consecutive_failures += 1
                return None
            
            # Validate and fix result
            result = validate_and_fix_result(result)
            
            # Reset failure counter on success
            self._consecutive_failures = 0
            
            return ProcessedArticle(
                article_hash=article.article_hash,
                source_type=article.source_type,
                source_name=article.source_name,
                title=article.title,
                url=article.url,
                published_at=article.published_at,
                y2ai_category=result["category"],
                extracted_facts=result.get("extracted_facts", []),
                impact_score=result["impact_score"],
                sentiment=result["sentiment"],
                companies_mentioned=result.get("companies_mentioned", []),
                dollar_amounts=result.get("dollar_amounts", []),
                key_quotes=result.get("key_quotes", []),
                processed_at=datetime.utcnow().isoformat()
            )
            
        except CircuitOpenError as e:
            logger.warning(f"Claude circuit open: {e}")
            return None
        except RateLimitError as e:
            logger.warning(f"Claude rate limited: {e}")
            self._consecutive_failures += 1
            return None
        except RetryExhaustedError as e:
            logger.error(f"Claude failed after retries for '{article.title[:50]}...': {e}")
            self._consecutive_failures += 1
            return None
        except Exception as e:
            logger.error(f"Unexpected error processing '{article.title[:50]}...': {e}")
            self._consecutive_failures += 1
            return None
    
    def process_batch(
        self, 
        articles: List[RawArticle], 
        max_batch: int = 50,
        stop_on_consecutive_failures: int = 3,
    ) -> List[ProcessedArticle]:
        """
        Process a batch of articles with partial failure tolerance.
        
        Args:
            articles: List of articles to process
            max_batch: Maximum articles to process
            stop_on_consecutive_failures: Stop if this many consecutive articles fail
        
        Returns:
            List of successfully processed articles
        """
        if not self.is_available():
            logger.error("Claude processor not available")
            return []
        
        processed = []
        consecutive_failures = 0
        
        for i, article in enumerate(articles[:max_batch]):
            logger.info(f"Processing {i+1}/{min(len(articles), max_batch)}: {article.title[:50]}...")
            
            result = self.categorize_and_extract(article)
            
            if result:
                processed.append(result)
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                
                if consecutive_failures >= stop_on_consecutive_failures:
                    logger.warning(
                        f"Stopping batch: {consecutive_failures} consecutive failures. "
                        f"Processed {len(processed)} of {i+1} attempted."
                    )
                    break
        
        success_rate = (len(processed) / min(len(articles), max_batch)) * 100 if articles else 0
        logger.info(
            f"Batch complete: {len(processed)}/{min(len(articles), max_batch)} "
            f"({success_rate:.1f}% success)"
        )
        
        return processed
    
    def quick_relevance_filter(self, articles: List[RawArticle]) -> List[RawArticle]:
        """Quick filter to reduce API calls - check relevance before full processing"""
        
        high_priority_keywords = [
            "capex", "capital expenditure", "billion", "data center",
            "gpu", "nvidia", "infrastructure", "ai spending",
            "earnings", "guidance", "microsoft", "google", "amazon", "meta",
            "bubble", "overvalued", "shortage"
        ]
        
        filtered = []
        for article in articles:
            text = f"{article.title} {article.content}".lower()
            if any(kw in text for kw in high_priority_keywords):
                filtered.append(article)
        
        logger.info(f"Quick filter: {len(filtered)}/{len(articles)} articles pass relevance check")
        return filtered
    
    def get_health(self) -> Dict[str, Any]:
        """Get processor health status"""
        tracker = get_health_tracker("anthropic")
        return {
            "available": self.is_available(),
            "consecutive_failures": self._consecutive_failures,
            "max_consecutive_failures": self._max_consecutive_failures,
            **tracker.to_dict()
        }
    
    def reset(self):
        """Reset processor state (clear failure counter)"""
        self._consecutive_failures = 0
        logger.info("Claude processor reset")


# =============================================================================
# BATCH PROCESSOR FOR NEWSLETTER GENERATION (Enhanced)
# =============================================================================

class NewsletterProcessor:
    """Process and organize articles for Y2AI Weekly generation"""
    
    def __init__(self):
        self.processor = ClaudeProcessor()
    
    def prepare_newsletter_data(self, processed_articles: List[ProcessedArticle]) -> dict:
        """Organize processed articles into newsletter-ready structure"""
        
        if not processed_articles:
            return {
                "total_articles": 0,
                "by_category": {},
                "category_counts": {},
                "top_companies": [],
                "dollar_amounts": [],
                "sentiment_distribution": {"bullish": 0, "bearish": 0, "neutral": 0},
                "high_impact_articles": [],
                "generated_at": datetime.utcnow().isoformat()
            }
        
        # Group by category
        by_category = {}
        for article in processed_articles:
            cat = article.y2ai_category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(article)
        
        # Sort each category by impact score
        for cat in by_category:
            by_category[cat].sort(key=lambda x: x.impact_score, reverse=True)
        
        # Extract key statistics
        all_amounts = []
        all_companies = set()
        for article in processed_articles:
            all_amounts.extend(article.dollar_amounts)
            all_companies.update(article.companies_mentioned)
        
        # Count sentiment distribution
        sentiment_counts = {"bullish": 0, "bearish": 0, "neutral": 0}
        for article in processed_articles:
            sentiment = article.sentiment
            if sentiment in sentiment_counts:
                sentiment_counts[sentiment] += 1
        
        return {
            "total_articles": len(processed_articles),
            "by_category": {k: [a.to_dict() for a in v] for k, v in by_category.items()},
            "category_counts": {k: len(v) for k, v in by_category.items()},
            "top_companies": list(all_companies)[:20],
            "dollar_amounts": all_amounts[:20],
            "sentiment_distribution": sentiment_counts,
            "high_impact_articles": [
                a.to_dict() for a in processed_articles 
                if a.impact_score >= 0.7
            ][:10],
            "generated_at": datetime.utcnow().isoformat()
        }


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    import json as json_module
    from .aggregator_enhanced import NewsAggregator
    
    # Check processor health
    processor = ClaudeProcessor()
    print(f"\n{'='*60}")
    print("CLAUDE PROCESSOR HEALTH")
    print(f"{'='*60}")
    health = processor.get_health()
    print(json_module.dumps(health, indent=2))
    
    if not processor.is_available():
        print("\nProcessor not available, exiting")
        exit(1)
    
    # Collect and process
    print(f"\n{'='*60}")
    print("COLLECTING ARTICLES")
    print(f"{'='*60}")
    aggregator = NewsAggregator()
    raw_articles = aggregator.collect_all(hours_back=24)
    
    # Quick filter first
    filtered = processor.quick_relevance_filter(raw_articles)
    
    # Process filtered articles
    print(f"\n{'='*60}")
    print("PROCESSING ARTICLES")
    print(f"{'='*60}")
    processed = processor.process_batch(filtered, max_batch=20)
    
    # Prepare newsletter data
    newsletter_proc = NewsletterProcessor()
    newsletter_data = newsletter_proc.prepare_newsletter_data(processed)
    
    print(f"\n{'='*60}")
    print("NEWSLETTER DATA SUMMARY")
    print(f"{'='*60}")
    print(f"Total processed: {newsletter_data['total_articles']}")
    print(f"Categories: {newsletter_data['category_counts']}")
    print(f"Sentiment: {newsletter_data['sentiment_distribution']}")
    print(f"High-impact articles: {len(newsletter_data['high_impact_articles'])}")
    
    if newsletter_data['top_companies']:
        print(f"Top companies: {', '.join(newsletter_data['top_companies'][:10])}")
    
    if newsletter_data['dollar_amounts']:
        print(f"Dollar amounts: {', '.join(newsletter_data['dollar_amounts'][:5])}")
