# """
# ARGUS-1 PROCESSOR
# Claude-based extraction and Y2AI categorization
# """

# import os
# import json
# from typing import List, Optional
# from dataclasses import dataclass
# import logging
# import anthropic

# from .aggregator import RawArticle, ProcessedArticle

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# # =============================================================================
# # Y2AI CATEGORY DEFINITIONS
# # =============================================================================

# Y2AI_CATEGORIES = {
#     "spending": {
#         "description": "Capital expenditure announcements, infrastructure investments, data center construction",
#         "keywords": ["capex", "capital expenditure", "billion", "investment", "data center", "infrastructure"],
#         "examples": ["Microsoft announces $80B AI infrastructure investment", "Google raises capex guidance"]
#     },
#     "constraints": {
#         "description": "Supply chain constraints, GPU shortages, power limitations",
#         "keywords": ["shortage", "constraint", "supply", "bottleneck", "power", "capacity"],
#         "examples": ["NVIDIA H100 backlog extends to 2025", "Data centers face power constraints"]
#     },
#     "data": {
#         "description": "Earnings data, revenue figures, margin analysis",
#         "keywords": ["earnings", "revenue", "margin", "profit", "guidance", "forecast"],
#         "examples": ["Google Cloud revenue up 34% YoY", "NVIDIA beats EPS estimates"]
#     },
#     "policy": {
#         "description": "Government policy, regulations, export controls",
#         "keywords": ["regulation", "policy", "government", "export", "ban", "restriction"],
#         "examples": ["US tightens chip export controls to China", "EU AI Act implementation"]
#     },
#     "skepticism": {
#         "description": "Bubble warnings, skeptical analysis, critical coverage",
#         "keywords": ["bubble", "overvalued", "correction", "warning", "skeptic", "hype"],
#         "examples": ["Fund managers call AI bubble biggest risk", "Analysts question AI valuations"]
#     },
#     "smartmoney": {
#         "description": "Institutional investor moves, hedge fund positions, insider activity",
#         "keywords": ["hedge fund", "institutional", "position", "bet", "insider", "allocation"],
#         "examples": ["Burry takes $1.1B put position", "Institutions increase AI allocations"]
#     },
#     "china": {
#         "description": "China-specific AI developments, US-China competition",
#         "keywords": ["china", "chinese", "huawei", "baidu", "alibaba", "tencent", "beijing"],
#         "examples": ["China develops domestic GPU alternative", "Huawei AI chip production"]
#     },
#     "energy": {
#         "description": "Power consumption, energy infrastructure, sustainability",
#         "keywords": ["power", "energy", "electricity", "nuclear", "renewable", "megawatt", "gigawatt"],
#         "examples": ["AI data centers drive 10% power demand increase", "Microsoft signs nuclear deal"]
#     },
#     "adoption": {
#         "description": "Enterprise AI adoption, use cases, ROI evidence",
#         "keywords": ["adoption", "enterprise", "roi", "productivity", "deploy", "implement"],
#         "examples": ["Enterprise AI adoption reaches 65%", "Companies report 30% productivity gains"]
#     }
# }


# # =============================================================================
# # CLAUDE PROCESSOR
# # =============================================================================

# class ClaudeProcessor:
#     """Process articles using Claude for extraction and categorization"""
    
#     def __init__(self):
#         self.client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
#         self.model = "claude-sonnet-4-20250514"
        
#     def _derive_keywords_from_text(self, article: RawArticle) -> Optional[List[str]]:
#         """Best-effort: derive keyword tags from the article title+content using Y2AI_CATEGORIES."""
#         text = f"{article.title or ''} {article.content or ''}".lower()
#         found = set()
#         for cat in Y2AI_CATEGORIES.values():
#             for kw in cat.get("keywords", []):
#                 if kw and kw.lower() in text:
#                     found.add(kw.lower())
#         return sorted(found) if found else None
    
    
#     def categorize_and_extract(self, article: RawArticle) -> Optional[ProcessedArticle]:
#         """Use Claude to categorize and extract structured data from article"""
        
#         prompt = f"""Analyze this news article for the Y2AI infrastructure investment thesis.

# ARTICLE:
# Title: {article.title}
# Source: {article.source_name}
# Published: {article.published_at}
# Content: {article.content[:3000]}

# CATEGORIES (select the single most relevant):
# {json.dumps({k: v['description'] for k, v in Y2AI_CATEGORIES.items()}, indent=2)}

# Respond with ONLY valid JSON in this exact format:
# {{
#     "category": "one of: spending, constraints, data, policy, skepticism, smartmoney, china, energy, adoption",
#     "impact_score": 0.0 to 1.0 (how relevant is this to AI infrastructure investment thesis),
#     "sentiment": "bullish, bearish, or neutral (regarding AI infrastructure investment)",
    
#     "extracted_facts": ["List of 2-4 specific factual claims from the article"],
#     "companies_mentioned": ["List of company names mentioned"],
#     "tickers_mentioned": ["Stock tickers if identifiable, e.g. NVDA, META, MSFT"],
#     "dollar_amounts": ["List of specific dollar amounts mentioned, e.g. '$80 billion'"],
#     "key_quotes": ["1-2 notable quotes from the article if any"],
    
#     "capex_signal": {{
#         "detected": true or false,
#         "direction": "increase, decrease, maintain, cut, or null if not detected",
#         "magnitude": "major (>$10B), moderate ($1-10B), minor (<$1B), or null",
#         "company": "company name or null",
#         "amount": "dollar amount string or null",
#         "context": "brief description of capex news or null"
#     }},
    
#     "energy_signal": {{
#         "detected": true or false,
#         "event_type": "permit_denied, permit_approved, grid_constraint, grid_expansion, power_contract, power_shortage, or null",
#         "direction": "positive (eases constraints), negative (increases constraints), or neutral",
#         "region": "geographic region or null",
#         "context": "brief description of energy news or null"
#     }},
    
#     "compute_signal": {{
#         "detected": true or false,
#         "event_type": "fab_delay, fab_expansion, chip_shortage, chip_surplus, export_ban, export_easing, capacity_expansion, or null",
#         "direction": "positive (eases supply), negative (tightens supply), or neutral",
#         "companies_affected": ["list of affected companies or empty"],
#         "context": "brief description of compute/supply chain news or null"
#     }},
    
#     "depreciation_signal": {{
#         "detected": true or false,
#         "event_type": "impairment, writedown, useful_life_change, asset_sale, or null",
#         "amount": "dollar amount string or null",
#         "company": "company name or null",
#         "context": "brief description or null"
#     }},
    
#     "veto_trigger": {{
#         "detected": true or false,
#         "trigger_type": "regulatory_ban, grid_emergency, credit_freeze, geopolitical_shock, or null",
#         "severity": "high, medium, low, or null",
#         "context": "brief description or null"
#     }},
    
#     "newsletter_relevance": {{
#         "include_in_weekly": true or false (is this significant enough for weekly newsletter),
#         "suggested_pillar": "which evidence pillar this supports: software_margin, physical_assets, smart_money, or null",
#         "one_line_summary": "one sentence summary suitable for newsletter if relevant"
#     }}
# }}"""

#         try:
#             response = self.client.messages.create(
#                 model=self.model,
#                 max_tokens=1000,
#                 messages=[{"role": "user", "content": prompt}]
#             )
            
#             # Parse JSON response
#             result_text = response.content[0].text
            
#             # Clean up potential markdown formatting
#             if "```json" in result_text:
#                 result_text = result_text.split("```json")[1].split("```")[0]
#             elif "```" in result_text:
#                 result_text = result_text.split("```")[1].split("```")[0]
            
#             result = json.loads(result_text.strip())
            
#             # Extract nested signal objects with CORRECT key names matching the prompt
#             capex = result.get("capex_signal", {}) or {}
#             energy = result.get("energy_signal", {}) or {}
#             compute = result.get("compute_signal", {}) or {}
#             depr = result.get("depreciation_signal", {}) or {}
#             veto = result.get("veto_trigger", {}) or {}
#             newsletter = result.get("newsletter_relevance", {}) or {}
            
#             kws = article.keywords_used
#             if not kws:
#                 kws = self._derive_keywords_from_text(article)
            
#             return ProcessedArticle(
#                 article_hash=article.article_hash,
#                 source_type=article.source_type,
#                 source_name=article.source_name,
#                 title=article.title,
#                 url=article.url,
#                 published_at=article.published_at,
#                 y2ai_category=result.get("category", "data"),
#                 extracted_facts=result.get("extracted_facts", []),
#                 impact_score=float(result.get("impact_score", 0.5)),
#                 sentiment=result.get("sentiment", "neutral"),
#                 companies_mentioned=result.get("companies_mentioned", []),
#                 dollar_amounts=result.get("dollar_amounts", []),
#                 key_quotes=result.get("key_quotes", []),
#                 processed_at=__import__('datetime').datetime.utcnow().isoformat(),
#                 keywords_used=kws,
                
#                 capex_detected=capex.get("detected", False),
#     capex_direction=capex.get("direction"),
#     capex_magnitude=capex.get("magnitude"),
#     capex_company=capex.get("company"),
#     capex_amount=capex.get("amount"),
#     capex_context=capex.get("context"),
#     # energy
#     energy_detected=energy.get("detected", False),
#     energy_event_type=energy.get("event_type"),
#     energy_direction=energy.get("direction"),
#     energy_region=energy.get("region"),
#     energy_context=energy.get("context"),
#     # compute
#     compute_detected=compute.get("detected", False),
#     compute_event_type=compute.get("event_type"),
#     compute_direction=compute.get("direction"),
#     compute_companies_affected=compute.get("companies_affected", []),
#     compute_context=compute.get("context"),
#     # depreciation
#     depreciation_detected=depr.get("detected", False),
#     depreciation_event_type=depr.get("event_type"),
#     depreciation_amount=depr.get("amount"),
#     depreciation_company=depr.get("company"),
#     depreciation_context=depr.get("context"),
#     # veto
#     veto_detected=veto.get("detected", False),
#     veto_trigger_type=veto.get("trigger_type"),
#     veto_severity=veto.get("severity"),
#     veto_context=veto.get("context"),
#     # newsletter hints
#     include_in_weekly=newsletter.get("include_in_weekly", False),
#     suggested_pillar=newsletter.get("suggested_pillar"),
#     one_line_summary=newsletter.get("one_line_summary"),
#             )
            
#         except json.JSONDecodeError as e:
#             logger.error(f"JSON parse error for {article.title}: {e}")
#             return None
#         except Exception as e:
#             logger.error(f"Claude processing error for {article.title}: {e}")
#             return None
    
#     def process_batch(self, articles: List[RawArticle], max_batch: int = 50) -> List[ProcessedArticle]:
#         """Process a batch of articles"""
#         processed = []
        
#         for i, article in enumerate(articles[:max_batch]):
#             logger.info(f"Processing {i+1}/{min(len(articles), max_batch)}: {article.title[:50]}...")
#             result = self.categorize_and_extract(article)
#             if result:
#                 processed.append(result)
        
#         logger.info(f"Successfully processed {len(processed)}/{len(articles)} articles")
#         return processed
    
#     def quick_relevance_filter(self, articles: List[RawArticle]) -> List[RawArticle]:
#         """Quick filter to reduce API calls - check relevance before full processing"""
        
#         # Keywords that indicate high relevance
#         high_priority_keywords = [
#             "capex", "capital expenditure", "billion", "data center",
#             "gpu", "nvidia", "infrastructure", "ai spending",
#             "earnings", "guidance", "microsoft", "google", "amazon", "meta",
#             "bubble", "overvalued", "shortage"
#         ]
        
#         filtered = []
#         for article in articles:
#             text = f"{article.title} {article.content}".lower()
#             if any(kw in text for kw in high_priority_keywords):
#                 filtered.append(article)
        
#         logger.info(f"Quick filter: {len(filtered)}/{len(articles)} articles pass relevance check")
#         return filtered


# # =============================================================================
# # BATCH PROCESSOR FOR NEWSLETTER GENERATION
# # =============================================================================

# class NewsletterProcessor:
#     """Process and organize articles for Y2AI Weekly generation"""
    
#     def __init__(self):
#         self.processor = ClaudeProcessor()
    
#     def prepare_newsletter_data(self, processed_articles: List[ProcessedArticle]) -> dict:
#         """Organize processed articles into newsletter-ready structure"""
        
#         # Group by category
#         by_category = {}
#         for article in processed_articles:
#             cat = article.y2ai_category
#             if cat not in by_category:
#                 by_category[cat] = []
#             by_category[cat].append(article)
        
#         # Sort each category by impact score
#         for cat in by_category:
#             by_category[cat].sort(key=lambda x: x.impact_score, reverse=True)
        
#         # Extract key statistics
#         all_amounts = []
#         all_companies = set()
#         for article in processed_articles:
#             all_amounts.extend(article.dollar_amounts)
#             all_companies.update(article.companies_mentioned)
        
#         # Count sentiment distribution
#         sentiment_counts = {"bullish": 0, "bearish": 0, "neutral": 0}
#         for article in processed_articles:
#             sentiment_counts[article.sentiment] = sentiment_counts.get(article.sentiment, 0) + 1
        
#         return {
#             "total_articles": len(processed_articles),
#             "by_category": {k: [a.to_dict() for a in v] for k, v in by_category.items()},
#             "category_counts": {k: len(v) for k, v in by_category.items()},
#             "top_companies": list(all_companies)[:20],
#             "dollar_amounts": all_amounts[:20],
#             "sentiment_distribution": sentiment_counts,
#             "high_impact_articles": [
#                 a.to_dict() for a in processed_articles 
#                 if a.impact_score >= 0.7
#             ][:10]
#         }


# # =============================================================================
# # COMMAND LINE INTERFACE
# # =============================================================================

# if __name__ == "__main__":
#     from .aggregator import NewsAggregator
    
#     # Collect and process
#     aggregator = NewsAggregator()
#     raw_articles = aggregator.collect_all(hours_back=24)
    
#     processor = ClaudeProcessor()
    
#     # Quick filter first
#     filtered = processor.quick_relevance_filter(raw_articles)
    
#     # Process filtered articles
#     processed = processor.process_batch(filtered, max_batch=20)
    
#     # Prepare newsletter data
#     newsletter_proc = NewsletterProcessor()
#     newsletter_data = newsletter_proc.prepare_newsletter_data(processed)
    
#     print(f"\n{'='*60}")
#     print("Newsletter Data Summary")
#     print(f"{'='*60}")
#     print(f"Total processed: {newsletter_data['total_articles']}")
#     print(f"Categories: {newsletter_data['category_counts']}")
#     print(f"Sentiment: {newsletter_data['sentiment_distribution']}")
#     print(f"High-impact articles: {len(newsletter_data['high_impact_articles'])}")




"""
ARGUS-1 PROCESSOR
Claude-based extraction and Y2AI categorization
Enhanced with aggressive signal detection and daily aggregation

Version: 2.0 - Merged enhanced signal detection
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import logging
import anthropic
from supabase import create_client, Client
from dotenv import load_dotenv

from .aggregator import RawArticle, ProcessedArticle

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Y2AI CATEGORY DEFINITIONS (Enhanced with signal weights)
# =============================================================================

Y2AI_CATEGORIES = {
    "spending": {
        "description": "Capital expenditure announcements, infrastructure investments, data center construction",
        "keywords": ["capex", "capital expenditure", "billion", "investment", "data center", "infrastructure"],
        "examples": ["Microsoft announces $80B AI infrastructure investment", "Google raises capex guidance"],
        "signal_weight": 1.0
    },
    "constraints": {
        "description": "Supply chain constraints, GPU shortages, power limitations",
        "keywords": ["shortage", "constraint", "supply", "bottleneck", "power", "capacity"],
        "examples": ["NVIDIA H100 backlog extends to 2025", "Data centers face power constraints"],
        "signal_weight": 0.9
    },
    "data": {
        "description": "Earnings data, revenue figures, margin analysis",
        "keywords": ["earnings", "revenue", "margin", "profit", "guidance", "forecast"],
        "examples": ["Google Cloud revenue up 34% YoY", "NVIDIA beats EPS estimates"],
        "signal_weight": 0.7
    },
    "policy": {
        "description": "Government policy, regulations, export controls",
        "keywords": ["regulation", "policy", "government", "export", "ban", "restriction"],
        "examples": ["US tightens chip export controls to China", "EU AI Act implementation"],
        "signal_weight": 0.8
    },
    "skepticism": {
        "description": "Bubble warnings, skeptical analysis, critical coverage",
        "keywords": ["bubble", "overvalued", "correction", "warning", "skeptic", "hype"],
        "examples": ["Fund managers call AI bubble biggest risk", "Analysts question AI valuations"],
        "signal_weight": 0.8
    },
    "smartmoney": {
        "description": "Institutional investor moves, hedge fund positions, insider activity",
        "keywords": ["hedge fund", "institutional", "position", "bet", "insider", "allocation"],
        "examples": ["Burry takes $1.1B put position", "Institutions increase AI allocations"],
        "signal_weight": 0.7
    },
    "china": {
        "description": "China-specific AI developments, US-China competition",
        "keywords": ["china", "chinese", "huawei", "baidu", "alibaba", "tencent", "beijing"],
        "examples": ["China develops domestic GPU alternative", "Huawei AI chip production"],
        "signal_weight": 0.8
    },
    "energy": {
        "description": "Power consumption, energy infrastructure, sustainability",
        "keywords": ["power", "energy", "electricity", "nuclear", "renewable", "megawatt", "gigawatt"],
        "examples": ["AI data centers drive 10% power demand increase", "Microsoft signs nuclear deal"],
        "signal_weight": 0.9
    },
    "adoption": {
        "description": "Enterprise AI adoption, use cases, ROI evidence",
        "keywords": ["adoption", "enterprise", "roi", "productivity", "deploy", "implement"],
        "examples": ["Enterprise AI adoption reaches 65%", "Companies report 30% productivity gains"],
        "signal_weight": 0.8
    }
}


# =============================================================================
# SIGNAL AGGREGATION WEIGHTS
# =============================================================================

DIRECTION_WEIGHTS = {
    "increase": 1.0,
    "expansion": 1.0,
    "positive": 1.0,
    "decrease": -1.0,
    "constraint": -0.5,
    "negative": -0.8,
    "maintain": 0.0,
    "neutral": 0.0,
    "discussed": 0.25,
    "mixed": 0.0,
    None: 0.0
}

MAGNITUDE_MULTIPLIERS = {
    "major": 2.0,
    "moderate": 1.0,
    "minor": 0.5,
    "mentioned": 0.25,
    None: 0.5
}


# =============================================================================
# ENHANCED EXTRACTION PROMPT
# =============================================================================

def build_extraction_prompt(title: str, source: str, published_at: str, content: str) -> str:
    """Build the enhanced signal extraction prompt with aggressive detection"""
    
    categories_json = json.dumps(
        {k: v['description'] for k, v in Y2AI_CATEGORIES.items()}, 
        indent=2
    )
    
    return f"""Analyze this news article for Y2AI infrastructure investment signals.

ARTICLE:
Title: {title}
Source: {source}
Published: {published_at}
Content: {content[:4000]}

SIGNAL DETECTION RULES (be aggressive - detect signals even from indirect mentions):

CAPEX SIGNALS - Detect if ANY of these appear:
- Direct: "raising capex", "increasing investment", "spending $X billion", "capital expenditure"
- Indirect: "spending plans", "infrastructure investment", "capital allocation", "buildout", "expansion plans", "investment cycle"
- Negative: "cutting spending", "reducing capex", "scaling back investment", "capex discipline"
- Discussion: analysts discussing spending, earnings calls mentioning investment, forecasts about capex
- Even mentions of OTHER ANALYSTS discussing company spending counts as detected

ENERGY SIGNALS - Detect if ANY of these appear:
- Constraints: "waiting for power", "grid capacity", "interconnection queue", "permit denied/delayed", "power bottleneck"
- Expansion: "power contract", "renewable deal", "nuclear partnership", "MW commitment", "energy agreement"
- Data centers: "datacenter power", "cooling requirements", "power density"
- Regional: load growth, transmission bottlenecks, utility negotiations, grid upgrades

COMPUTE SIGNALS - Detect if ANY of these appear:
- Supply constraints: "sold out", "allocation", "lead times", "shortages", "capacity constraints", "waitlist"
- Expansion: "new fab", "capacity increase", "production ramp", "foundry expansion"
- Technology: "CoWoS", "advanced packaging", "HBM", "GPU availability"
- Trade: "export controls", "restrictions", "banned", "sanctions", "CHIPS Act"

DEPRECIATION SIGNALS - Detect if ANY of these appear:
- "useful life", "depreciation policy", "write-down", "impairment", "asset life change"
- "accelerated depreciation", "server refresh", "infrastructure refresh cycle"

VETO TRIGGERS - High severity events that could halt the cycle:
- Regulatory: "banned", "prohibited", "forced divestiture", "antitrust action"
- Crisis: "grid emergency", "blackout", "credit freeze", "liquidity crisis"
- Geopolitical: "sanctions escalation", "trade war", "export ban expansion"

IMPORTANT INSTRUCTIONS:
1. A signal is "detected" if the topic is DISCUSSED AT ALL, even without specific numbers
2. "direction" reflects the IMPLICATION for AI infrastructure investment
3. "magnitude" reflects importance: major (strategic shift), moderate (meaningful), minor (passing mention), mentioned (just referenced)
4. WHEN IN DOUBT, SET DETECTED TO TRUE and explain in context field
5. Every signal should have a "context" explanation when detected

CATEGORIES (select single most relevant):
{categories_json}

Respond with ONLY valid JSON (no markdown, no explanation):
{{
    "category": "spending|constraints|data|policy|skepticism|smartmoney|china|energy|adoption",
    "extracted_facts": ["2-4 specific factual claims from the article"],
    "impact_score": 0.0-1.0,
    "sentiment": "bullish|bearish|neutral",
    "companies_mentioned": ["company names"],
    "tickers_mentioned": ["NVDA", "META", "GOOGL", etc or empty list],
    "dollar_amounts": ["$X billion amounts mentioned or empty list"],
    "key_quotes": ["1-2 notable direct quotes or empty list"],
    
    "capex_signal": {{
        "detected": true/false,
        "direction": "increase|decrease|maintain|discussed|null",
        "magnitude": "major|moderate|minor|mentioned|null",
        "company": "primary company or null",
        "amount": "dollar amount string or null",
        "context": "brief explanation of what was detected and why"
    }},
    
    "energy_signal": {{
        "detected": true/false,
        "event_type": "constraint|expansion|contract|grid_issue|permit|datacenter_power|null",
        "direction": "positive|negative|neutral|null",
        "region": "geographic region or null",
        "context": "brief explanation"
    }},
    
    "compute_signal": {{
        "detected": true/false,
        "event_type": "shortage|expansion|restriction|delay|technology|trade|null",
        "direction": "positive|negative|neutral|null",
        "companies_affected": ["list of companies or empty"],
        "context": "brief explanation"
    }},
    
    "depreciation_signal": {{
        "detected": true/false,
        "event_type": "writedown|useful_life|impairment|refresh_cycle|null",
        "company": "company name or null",
        "amount": "dollar amount or null",
        "context": "brief explanation"
    }},
    
    "veto_trigger": {{
        "detected": true/false,
        "trigger_type": "regulatory|crisis|geopolitical|null",
        "severity": "high|medium|low|null",
        "context": "brief explanation"
    }},
    
    "thesis_relevance": {{
        "infrastructure_cycle_support": true/false,
        "bubble_warning": true/false,
        "constraint_evidence": true/false,
        "demand_validation": true/false,
        "explanation": "one sentence connecting this to infrastructure cycle vs bubble thesis"
    }},
    
    "newsletter_relevance": {{
        "include_in_weekly": true/false,
        "suggested_pillar": "software_margin|physical_assets|smart_money|null",
        "one_line_summary": "one sentence summary suitable for newsletter if relevant"
    }}
}}"""


# =============================================================================
# CLAUDE PROCESSOR (Enhanced)
# =============================================================================

class ClaudeProcessor:
    """Process articles using Claude for extraction and categorization"""
    
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.model = "claude-sonnet-4-20250514"
        
    def _derive_keywords_from_text(self, article: RawArticle) -> Optional[List[str]]:
        """Best-effort: derive keyword tags from the article title+content using Y2AI_CATEGORIES."""
        text = f"{article.title or ''} {article.content or ''}".lower()
        found = set()
        for cat in Y2AI_CATEGORIES.values():
            for kw in cat.get("keywords", []):
                if kw and kw.lower() in text:
                    found.add(kw.lower())
        return sorted(found) if found else None
    
    def categorize_and_extract(self, article: RawArticle) -> Optional[ProcessedArticle]:
        """Use Claude to categorize and extract structured data from article"""
        
        prompt = build_extraction_prompt(
            title=article.title,
            source=article.source_name,
            published_at=str(article.published_at),
            content=article.content[:4000] if article.content else ""
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result_text = response.content[0].text
            
            # Clean up potential markdown formatting
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]
            
            result = json.loads(result_text.strip())
            
            # Extract nested signal objects with correct key names
            capex = result.get("capex_signal", {}) or {}
            energy = result.get("energy_signal", {}) or {}
            compute = result.get("compute_signal", {}) or {}
            depr = result.get("depreciation_signal", {}) or {}
            veto = result.get("veto_trigger", {}) or {}
            newsletter = result.get("newsletter_relevance", {}) or {}
            thesis = result.get("thesis_relevance", {}) or {}
            
            kws = article.keywords_used
            if not kws:
                kws = self._derive_keywords_from_text(article)
            
            return ProcessedArticle(
                article_hash=article.article_hash,
                source_type=article.source_type,
                source_name=article.source_name,
                title=article.title,
                url=article.url,
                published_at=article.published_at,
                y2ai_category=result.get("category", "data"),
                extracted_facts=result.get("extracted_facts", []),
                impact_score=float(result.get("impact_score", 0.5)),
                sentiment=result.get("sentiment", "neutral"),
                companies_mentioned=result.get("companies_mentioned", []),
                dollar_amounts=result.get("dollar_amounts", []),
                key_quotes=result.get("key_quotes", []),
                processed_at=datetime.utcnow().isoformat(),
                keywords_used=kws,
                
                # Capex signal
                capex_detected=capex.get("detected", False),
                capex_direction=capex.get("direction"),
                capex_magnitude=capex.get("magnitude"),
                capex_company=capex.get("company"),
                capex_amount=capex.get("amount"),
                capex_context=capex.get("context"),
                
                # Energy signal
                energy_detected=energy.get("detected", False),
                energy_event_type=energy.get("event_type"),
                energy_direction=energy.get("direction"),
                energy_region=energy.get("region"),
                energy_context=energy.get("context"),
                
                # Compute signal
                compute_detected=compute.get("detected", False),
                compute_event_type=compute.get("event_type"),
                compute_direction=compute.get("direction"),
                compute_companies_affected=compute.get("companies_affected", []),
                compute_context=compute.get("context"),
                
                # Depreciation signal
                depreciation_detected=depr.get("detected", False),
                depreciation_event_type=depr.get("event_type"),
                depreciation_amount=depr.get("amount"),
                depreciation_company=depr.get("company"),
                depreciation_context=depr.get("context"),
                
                # Veto signal
                veto_detected=veto.get("detected", False),
                veto_trigger_type=veto.get("trigger_type"),
                veto_severity=veto.get("severity"),
                veto_context=veto.get("context"),
                
                # Newsletter hints
                include_in_weekly=newsletter.get("include_in_weekly", False),
                suggested_pillar=newsletter.get("suggested_pillar"),
                one_line_summary=newsletter.get("one_line_summary"),
                
                # Thesis relevance (new fields)
                thesis_infrastructure_support=thesis.get("infrastructure_cycle_support", False),
                thesis_bubble_warning=thesis.get("bubble_warning", False),
                thesis_constraint_evidence=thesis.get("constraint_evidence", False),
                thesis_demand_validation=thesis.get("demand_validation", False),
                thesis_explanation=thesis.get("explanation"),
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error for {article.title}: {e}")
            return None
        except Exception as e:
            logger.error(f"Claude processing error for {article.title}: {e}")
            return None
    
    def process_batch(self, articles: List[RawArticle], max_batch: int = 50) -> List[ProcessedArticle]:
        """Process a batch of articles"""
        processed = []
        
        for i, article in enumerate(articles[:max_batch]):
            logger.info(f"Processing {i+1}/{min(len(articles), max_batch)}: {article.title[:50]}...")
            result = self.categorize_and_extract(article)
            if result:
                processed.append(result)
        
        logger.info(f"Successfully processed {len(processed)}/{len(articles)} articles")
        return processed
    
    def quick_relevance_filter(self, articles: List[RawArticle]) -> List[RawArticle]:
        """Quick filter to reduce API calls - check relevance before full processing"""
        
        high_priority_keywords = [
            "capex", "capital expenditure", "billion", "data center",
            "gpu", "nvidia", "infrastructure", "ai spending",
            "earnings", "guidance", "microsoft", "google", "amazon", "meta",
            "bubble", "overvalued", "shortage", "power", "energy",
            "datacenter", "chip", "semiconductor", "tsmc", "asml"
        ]
        
        filtered = []
        for article in articles:
            text = f"{article.title} {article.content}".lower()
            if any(kw in text for kw in high_priority_keywords):
                filtered.append(article)
        
        logger.info(f"Quick filter: {len(filtered)}/{len(articles)} articles pass relevance check")
        return filtered


# =============================================================================
# DAILY SIGNAL AGGREGATOR (New from y2ai_signal_processor)
# =============================================================================

class DailySignalAggregator:
    """Aggregate article signals into daily composite scores"""
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        supabase_url = supabase_url or os.getenv('SUPABASE_URL')
        supabase_key = supabase_key or os.getenv('SUPABASE_KEY')
        
        if supabase_url and supabase_key:
            self.supabase: Client = create_client(supabase_url, supabase_key)
        else:
            self.supabase = None
            logger.warning("Supabase not configured - daily aggregation will not persist")
    
    def calculate_daily_signals(self, date: str) -> Dict[str, Any]:
        """Calculate aggregated signal scores for a specific date"""
        
        if not self.supabase:
            return self._empty_daily_signals(date)
        
        # Fetch articles for the date
        response = self.supabase.table('processed_articles').select('*').gte(
            'published_at', f"{date}T00:00:00"
        ).lt(
            'published_at', f"{date}T23:59:59"
        ).execute()
        
        articles = response.data
        if not articles:
            return self._empty_daily_signals(date)
        
        # Initialize accumulators
        capex_scores = []
        energy_scores = []
        compute_scores = []
        depreciation_count = 0
        veto_count = 0
        
        thesis_support = 0
        bubble_warnings = 0
        constraint_evidence = 0
        demand_validation = 0
        
        total_impact = 0
        article_count = len(articles)
        
        for article in articles:
            impact = article.get('impact_score', 0.5)
            total_impact += impact
            
            # Capex signal aggregation
            if article.get('capex_detected'):
                direction = article.get('capex_direction', 'discussed')
                magnitude = article.get('capex_magnitude', 'mentioned')
                score = (
                    DIRECTION_WEIGHTS.get(direction, 0) * 
                    MAGNITUDE_MULTIPLIERS.get(magnitude, 0.5) * 
                    impact
                )
                capex_scores.append(score)
            
            # Energy signal aggregation
            if article.get('energy_detected'):
                direction = article.get('energy_direction', 'neutral')
                score = DIRECTION_WEIGHTS.get(direction, 0) * impact
                energy_scores.append(score)
            
            # Compute signal aggregation
            if article.get('compute_detected'):
                direction = article.get('compute_direction', 'neutral')
                score = DIRECTION_WEIGHTS.get(direction, 0) * impact
                compute_scores.append(score)
            
            # Depreciation count
            if article.get('depreciation_detected'):
                depreciation_count += 1
            
            # Veto trigger count
            if article.get('veto_detected'):
                veto_count += 1
            
            # Thesis relevance
            if article.get('thesis_infrastructure_support'):
                thesis_support += 1
            if article.get('thesis_bubble_warning'):
                bubble_warnings += 1
            if article.get('thesis_constraint_evidence'):
                constraint_evidence += 1
            if article.get('thesis_demand_validation'):
                demand_validation += 1
        
        # Calculate composite scores (-100 to +100 scale)
        def safe_avg(scores, default=0):
            return sum(scores) / len(scores) if scores else default
        
        capex_composite = safe_avg(capex_scores) * 100
        energy_composite = safe_avg(energy_scores) * 100
        compute_composite = safe_avg(compute_scores) * 100
        
        # Overall infrastructure signal
        infra_signal = (
            capex_composite * 0.4 +
            energy_composite * 0.3 +
            compute_composite * 0.3
        )
        
        # Thesis balance (positive = infrastructure cycle, negative = bubble)
        thesis_balance = (
            (thesis_support - bubble_warnings) / article_count * 100
            if article_count > 0 else 0
        )
        
        return {
            "date": date,
            "article_count": article_count,
            "avg_impact_score": total_impact / article_count if article_count > 0 else 0,
            
            # Signal composites (-100 to +100)
            "capex_signal": round(capex_composite, 2),
            "capex_articles": len(capex_scores),
            
            "energy_signal": round(energy_composite, 2),
            "energy_articles": len(energy_scores),
            
            "compute_signal": round(compute_composite, 2),
            "compute_articles": len(compute_scores),
            
            "depreciation_articles": depreciation_count,
            "veto_triggers": veto_count,
            
            # Thesis indicators
            "thesis_support_count": thesis_support,
            "bubble_warning_count": bubble_warnings,
            "constraint_evidence_count": constraint_evidence,
            "demand_validation_count": demand_validation,
            "thesis_balance": round(thesis_balance, 2),
            
            # Overall score
            "infrastructure_signal": round(infra_signal, 2),
            
            # Regime classification
            "signal_regime": self._classify_signal_regime(infra_signal, veto_count, thesis_balance)
        }
    
    def _classify_signal_regime(self, infra_signal: float, veto_count: int, thesis_balance: float) -> str:
        """Classify the overall signal regime"""
        
        if veto_count > 0:
            return "VETO_ALERT"
        
        if infra_signal > 30 and thesis_balance > 20:
            return "STRONG_CYCLE"
        elif infra_signal > 10 and thesis_balance > 0:
            return "CYCLE_INTACT"
        elif infra_signal > -10 and thesis_balance > -20:
            return "MIXED_SIGNALS"
        elif infra_signal > -30 or thesis_balance > -40:
            return "WEAKENING"
        else:
            return "CYCLE_RISK"
    
    def _empty_daily_signals(self, date: str) -> Dict[str, Any]:
        """Return empty signal structure for days with no articles"""
        return {
            "date": date,
            "article_count": 0,
            "avg_impact_score": 0,
            "capex_signal": 0,
            "capex_articles": 0,
            "energy_signal": 0,
            "energy_articles": 0,
            "compute_signal": 0,
            "compute_articles": 0,
            "depreciation_articles": 0,
            "veto_triggers": 0,
            "thesis_support_count": 0,
            "bubble_warning_count": 0,
            "constraint_evidence_count": 0,
            "demand_validation_count": 0,
            "thesis_balance": 0,
            "infrastructure_signal": 0,
            "signal_regime": "NO_DATA"
        }
    
    def store_daily_signals(self, date: str) -> bool:
        """Calculate and store daily signals in Supabase"""
        
        if not self.supabase:
            logger.error("Supabase not configured")
            return False
        
        daily_signals = self.calculate_daily_signals(date)
        
        try:
            self.supabase.table('daily_signals').upsert(
                daily_signals,
                on_conflict='date'
            ).execute()
            
            logger.info(f"Stored daily signals for {date}: {daily_signals['signal_regime']}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing daily signals for {date}: {e}")
            return False
    
    def backfill_daily_signals(self, start_date: str, end_date: str) -> Dict[str, int]:
        """Backfill daily signals for a date range"""
        
        stats = {"success": 0, "failed": 0}
        
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            if self.store_daily_signals(date_str):
                stats['success'] += 1
            else:
                stats['failed'] += 1
            current += timedelta(days=1)
        
        return stats


# =============================================================================
# API ENDPOINT FOR GOOGLE APPS SCRIPT
# =============================================================================

def get_daily_signals_for_sheets(supabase: Client, date: str) -> Dict[str, Any]:
    """
    Fetch daily signals formatted for Google Apps Script consumption
    Returns a simplified structure optimized for Sheets integration
    """
    
    response = supabase.table('daily_signals').select('*').eq('date', date).execute()
    
    if not response.data:
        return {"error": "No data for date", "date": date}
    
    signals = response.data[0]
    
    return {
        "date": signals['date'],
        "infrastructure_signal": signals['infrastructure_signal'],
        "signal_regime": signals['signal_regime'],
        "capex": {
            "score": signals['capex_signal'],
            "articles": signals['capex_articles']
        },
        "energy": {
            "score": signals['energy_signal'],
            "articles": signals['energy_articles']
        },
        "compute": {
            "score": signals['compute_signal'],
            "articles": signals['compute_articles']
        },
        "thesis": {
            "balance": signals['thesis_balance'],
            "support": signals['thesis_support_count'],
            "warnings": signals['bubble_warning_count']
        },
        "alerts": {
            "veto_triggers": signals['veto_triggers'],
            "depreciation_flags": signals['depreciation_articles']
        }
    }


# =============================================================================
# BATCH PROCESSOR FOR NEWSLETTER GENERATION
# =============================================================================

class NewsletterProcessor:
    """Process and organize articles for Y2AI Weekly generation"""
    
    def __init__(self):
        self.processor = ClaudeProcessor()
    
    def prepare_newsletter_data(self, processed_articles: List[ProcessedArticle]) -> dict:
        """Organize processed articles into newsletter-ready structure"""
        
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
            sentiment_counts[article.sentiment] = sentiment_counts.get(article.sentiment, 0) + 1
        
        # Count signal detections
        signal_counts = {
            "capex_detected": sum(1 for a in processed_articles if a.capex_detected),
            "energy_detected": sum(1 for a in processed_articles if a.energy_detected),
            "compute_detected": sum(1 for a in processed_articles if a.compute_detected),
            "depreciation_detected": sum(1 for a in processed_articles if a.depreciation_detected),
            "veto_detected": sum(1 for a in processed_articles if a.veto_detected),
        }
        
        # Thesis summary
        thesis_summary = {
            "infrastructure_support": sum(1 for a in processed_articles if getattr(a, 'thesis_infrastructure_support', False)),
            "bubble_warnings": sum(1 for a in processed_articles if getattr(a, 'thesis_bubble_warning', False)),
            "constraint_evidence": sum(1 for a in processed_articles if getattr(a, 'thesis_constraint_evidence', False)),
            "demand_validation": sum(1 for a in processed_articles if getattr(a, 'thesis_demand_validation', False)),
        }
        
        return {
            "total_articles": len(processed_articles),
            "by_category": {k: [a.to_dict() for a in v] for k, v in by_category.items()},
            "category_counts": {k: len(v) for k, v in by_category.items()},
            "top_companies": list(all_companies)[:20],
            "dollar_amounts": all_amounts[:20],
            "sentiment_distribution": sentiment_counts,
            "signal_counts": signal_counts,
            "thesis_summary": thesis_summary,
            "high_impact_articles": [
                a.to_dict() for a in processed_articles 
                if a.impact_score >= 0.7
            ][:10],
            "newsletter_candidates": [
                a.to_dict() for a in processed_articles
                if a.include_in_weekly
            ]
        }


# =============================================================================
# REPROCESSOR FOR EXISTING ARTICLES
# =============================================================================

class ArticleReprocessor:
    """Reprocess existing articles with enhanced signal detection"""
    
    def __init__(self):
        self.processor = ClaudeProcessor()
        
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        if supabase_url and supabase_key:
            self.supabase: Client = create_client(supabase_url, supabase_key)
        else:
            self.supabase = None
    
    def reprocess_all(self, start_date: str = None, end_date: str = None, 
                      batch_size: int = 50) -> Dict[str, int]:
        """Reprocess all articles with enhanced signal detection"""
        
        if not self.supabase:
            logger.error("Supabase not configured")
            return {"processed": 0, "failed": 0, "skipped": 0}
        
        stats = {"processed": 0, "failed": 0, "skipped": 0}
        
        # Build query
        query = self.supabase.table('processed_articles').select('*')
        
        if start_date:
            query = query.gte('published_at', start_date)
        if end_date:
            query = query.lte('published_at', end_date)
        
        response = query.order('published_at', desc=True).limit(1000).execute()
        articles = response.data
        
        logger.info(f"Found {len(articles)} articles to reprocess")
        
        for i, article_data in enumerate(articles):
            logger.info(f"Processing {i+1}/{len(articles)}: {article_data.get('title', '')[:50]}...")
            
            # Create RawArticle from stored data
            raw = RawArticle(
                article_hash=article_data.get('article_hash', ''),
                source_type=article_data.get('source_type', 'unknown'),
                source_name=article_data.get('source_name', ''),
                title=article_data.get('title', ''),
                url=article_data.get('url', ''),
                content=article_data.get('content', ''),
                published_at=article_data.get('published_at', ''),
                keywords_used=article_data.get('keywords_used', [])
            )
            
            result = self.processor.categorize_and_extract(raw)
            
            if result:
                # Update the article in Supabase
                update_data = {
                    "y2ai_category": result.y2ai_category,
                    "impact_score": result.impact_score,
                    "sentiment": result.sentiment,
                    "extracted_facts": result.extracted_facts,
                    "companies_mentioned": result.companies_mentioned,
                    "dollar_amounts": result.dollar_amounts,
                    "key_quotes": result.key_quotes,
                    
                    # Signal fields
                    "capex_detected": result.capex_detected,
                    "capex_direction": result.capex_direction,
                    "capex_magnitude": result.capex_magnitude,
                    "capex_company": result.capex_company,
                    "capex_amount": result.capex_amount,
                    "capex_context": result.capex_context,
                    
                    "energy_detected": result.energy_detected,
                    "energy_event_type": result.energy_event_type,
                    "energy_direction": result.energy_direction,
                    "energy_region": result.energy_region,
                    "energy_context": result.energy_context,
                    
                    "compute_detected": result.compute_detected,
                    "compute_event_type": result.compute_event_type,
                    "compute_direction": result.compute_direction,
                    "compute_companies_affected": result.compute_companies_affected,
                    "compute_context": result.compute_context,
                    
                    "depreciation_detected": result.depreciation_detected,
                    "depreciation_event_type": result.depreciation_event_type,
                    "depreciation_amount": result.depreciation_amount,
                    "depreciation_company": result.depreciation_company,
                    "depreciation_context": result.depreciation_context,
                    
                    "veto_detected": result.veto_detected,
                    "veto_trigger_type": result.veto_trigger_type,
                    "veto_severity": result.veto_severity,
                    "veto_context": result.veto_context,
                    
                    "include_in_weekly": result.include_in_weekly,
                    "suggested_pillar": result.suggested_pillar,
                    "one_line_summary": result.one_line_summary,
                    
                    # Thesis fields
                    "thesis_infrastructure_support": getattr(result, 'thesis_infrastructure_support', False),
                    "thesis_bubble_warning": getattr(result, 'thesis_bubble_warning', False),
                    "thesis_constraint_evidence": getattr(result, 'thesis_constraint_evidence', False),
                    "thesis_demand_validation": getattr(result, 'thesis_demand_validation', False),
                    "thesis_explanation": getattr(result, 'thesis_explanation', None),
                    
                    "reprocessed_at": datetime.utcnow().isoformat()
                }
                
                try:
                    self.supabase.table('processed_articles').update(update_data).eq(
                        'id', article_data['id']
                    ).execute()
                    stats['processed'] += 1
                except Exception as e:
                    logger.error(f"Error updating article {article_data.get('id')}: {e}")
                    stats['failed'] += 1
            else:
                stats['failed'] += 1
            
            if (i + 1) % batch_size == 0:
                logger.info(f"Batch complete. Processed: {stats['processed']}, Failed: {stats['failed']}")
        
        return stats


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    from .aggregator import NewsAggregator
    
    print("=" * 60)
    print("ARGUS-1 PROCESSOR - Enhanced Signal Detection")
    print("=" * 60)
    
    # Option 1: Process new articles
    aggregator = NewsAggregator()
    raw_articles = aggregator.collect_all(hours_back=24)
    
    processor = ClaudeProcessor()
    
    # Quick filter first
    filtered = processor.quick_relevance_filter(raw_articles)
    
    # Process filtered articles
    processed = processor.process_batch(filtered, max_batch=20)
    
    # Prepare newsletter data
    newsletter_proc = NewsletterProcessor()
    newsletter_data = newsletter_proc.prepare_newsletter_data(processed)
    
    print(f"\n{'='*60}")
    print("Newsletter Data Summary")
    print(f"{'='*60}")
    print(f"Total processed: {newsletter_data['total_articles']}")
    print(f"Categories: {newsletter_data['category_counts']}")
    print(f"Sentiment: {newsletter_data['sentiment_distribution']}")
    print(f"Signal detections: {newsletter_data['signal_counts']}")
    print(f"Thesis summary: {newsletter_data['thesis_summary']}")
    print(f"High-impact articles: {len(newsletter_data['high_impact_articles'])}")
    print(f"Newsletter candidates: {len(newsletter_data['newsletter_candidates'])}")
    
    # Option 2: Calculate daily signals
    print(f"\n{'='*60}")
    print("Daily Signal Aggregation")
    print(f"{'='*60}")
    
    aggregator = DailySignalAggregator()
    today = datetime.utcnow().strftime("%Y-%m-%d")
    daily = aggregator.calculate_daily_signals(today)
    
    print(f"Date: {daily['date']}")
    print(f"Articles: {daily['article_count']}")
    print(f"Infrastructure Signal: {daily['infrastructure_signal']}")
    print(f"Signal Regime: {daily['signal_regime']}")