"""
Y2AI NEWSLETTER GENERATION MODULE
Claude-powered content generation for Y2AI Weekly

Features:
- Article synthesis and trend analysis
- Y2AI framework commentary integration
- Social media post generation
- Academic-quality formatting with citations
- Writing style enforcement (prose over bullets)
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

# Import resilience module
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
# CONFIGURATION
# =============================================================================

# Writing style guidelines embedded in prompts
WRITING_STYLE_GUIDELINES = """
Writing Style Requirements:
- Write in flowing paragraphs, NOT bullet points or lists
- Each paragraph should be 3-5 sentences
- NO single-sentence paragraphs
- Use natural transitions like "Here's the thing" or "And then" instead of "Moreover" or "Furthermore"
- Integrate data into narrative flow, don't list statistics separately
- Use contractions naturally (it's, that's, we're)
- Avoid corporate jargon: no "leverage", "synergy", "paradigm", "unlock", "harness"
- Professional but conversational tone
- Include parenthetical asides occasionally for color
- Vary sentence length - mix punchy short sentences with longer explanatory ones
"""

# Y2AI Framework reference
Y2AI_FRAMEWORK = """
Y2AI Framework Context:
The Y2AI Infrastructure Bifurcation Framework distinguishes genuine infrastructure investment cycles 
from speculative bubbles using three coupled indicators:

1. Bubble Index (BI): Valuation extremeness on 0-100 scale derived from CAPE ratio
   - 0-30: Undervalued
   - 30-50: Fair value  
   - 50-70: Elevated
   - 70-100: Extreme

2. VIX Z-Score (VI): Market volatility relative to 5-year history
   - Positive: Above-average fear
   - Negative: Complacency

3. Credit Spreads Z-Score (CS): Financial system stress
   - Positive: Elevated stress
   - Negative: Easy conditions

Bifurcation Formula: Score = 0.6×BI_normalized - 0.2×VI - 0.2×CS

Regimes:
- INFRASTRUCTURE (score > +0.5): Strong infrastructure cycle, not a bubble
- ADOPTION (+0.2 to +0.5): Healthy adoption phase
- TRANSITION (-0.2 to +0.2): Watching for regime change
- BUBBLE_WARNING (< -0.2): Elevated risk signals

Key Thesis: Current AI infrastructure spending resembles Y2K (necessary upgrade cycle) 
rather than dot-com (speculation). Companies are investing in real infrastructure that 
will generate returns, not chasing eyeballs.
"""


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class NewsletterSection:
    """A section of the newsletter"""
    title: str
    content: str
    section_type: str  # "lead", "analysis", "data", "outlook", "social"
    sources: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SocialPost:
    """Social media post for various platforms"""
    platform: str  # "twitter", "linkedin", "bluesky"
    content: str
    hashtags: List[str] = field(default_factory=list)
    thread: List[str] = field(default_factory=list)  # For Twitter threads
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GeneratedNewsletter:
    """Complete generated newsletter"""
    edition_number: int
    date: str
    title: str
    
    # Core sections
    lead_section: Optional[NewsletterSection] = None
    market_update: Optional[NewsletterSection] = None
    spending_analysis: Optional[NewsletterSection] = None
    thesis_status: Optional[NewsletterSection] = None
    outlook: Optional[NewsletterSection] = None
    
    # Supplementary
    key_data_points: List[str] = field(default_factory=list)
    sources_cited: List[Dict[str, str]] = field(default_factory=list)
    
    # Social media
    social_posts: List[SocialPost] = field(default_factory=list)
    
    # Metadata
    bubble_index: Optional[float] = None
    bifurcation_score: Optional[float] = None
    regime: Optional[str] = None
    generated_at: str = ""
    
    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.utcnow().isoformat()
    
    def to_dict(self) -> dict:
        d = {
            "edition_number": self.edition_number,
            "date": self.date,
            "title": self.title,
            "key_data_points": self.key_data_points,
            "sources_cited": self.sources_cited,
            "bubble_index": self.bubble_index,
            "bifurcation_score": self.bifurcation_score,
            "regime": self.regime,
            "generated_at": self.generated_at,
        }
        
        if self.lead_section:
            d["lead_section"] = self.lead_section.to_dict()
        if self.market_update:
            d["market_update"] = self.market_update.to_dict()
        if self.spending_analysis:
            d["spending_analysis"] = self.spending_analysis.to_dict()
        if self.thesis_status:
            d["thesis_status"] = self.thesis_status.to_dict()
        if self.outlook:
            d["outlook"] = self.outlook.to_dict()
        
        d["social_posts"] = [p.to_dict() for p in self.social_posts]
        
        return d
    
    def to_markdown(self) -> str:
        """Export newsletter as markdown"""
        lines = []
        
        lines.append(f"# Y2AI Weekly Edition #{self.edition_number}")
        lines.append(f"*{self.date}*\n")
        
        if self.bubble_index is not None:
            lines.append(f"**Current Readings:** Bubble Index {self.bubble_index:.1f} | ")
            lines.append(f"Bifurcation Score {self.bifurcation_score:+.2f} | Regime: {self.regime}\n")
        
        if self.lead_section:
            lines.append(f"## {self.lead_section.title}\n")
            lines.append(self.lead_section.content)
            lines.append("")
        
        if self.market_update:
            lines.append(f"## {self.market_update.title}\n")
            lines.append(self.market_update.content)
            lines.append("")
        
        if self.spending_analysis:
            lines.append(f"## {self.spending_analysis.title}\n")
            lines.append(self.spending_analysis.content)
            lines.append("")
        
        if self.thesis_status:
            lines.append(f"## {self.thesis_status.title}\n")
            lines.append(self.thesis_status.content)
            lines.append("")
        
        if self.outlook:
            lines.append(f"## {self.outlook.title}\n")
            lines.append(self.outlook.content)
            lines.append("")
        
        if self.sources_cited:
            lines.append("---\n")
            lines.append("### Sources\n")
            for i, source in enumerate(self.sources_cited, 1):
                lines.append(f"{i}. [{source.get('title', 'Source')}]({source.get('url', '')})")
        
        return "\n".join(lines)


# =============================================================================
# NEWSLETTER GENERATOR
# =============================================================================

class NewsletterGenerator:
    """
    Generate Y2AI Weekly newsletter content using Claude.
    
    Takes processed articles and bubble index data, produces newsletter sections
    and social media posts following Y2AI's writing style.
    """
    
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        """Initialize the generator with Claude client"""
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
            self.model = model
            self._available = True
        except ImportError:
            logger.error("anthropic package not installed")
            self.client = None
            self._available = False
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            self.client = None
            self._available = False
    
    def is_available(self) -> bool:
        """Check if generator is available"""
        return self._available and self.client is not None
    
    @resilient_call(
        service_name="anthropic_newsletter",
        max_retries=3,
        base_delay=2.0,
        use_circuit_breaker=True,
        use_rate_limiter=True,
    )
    def _call_claude(self, system_prompt: str, user_prompt: str, max_tokens: int = 2000) -> str:
        """Make a resilient call to Claude API"""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        return response.content[0].text
    
    def _build_article_context(self, newsletter_data: Dict[str, Any]) -> str:
        """Build context string from processed articles"""
        lines = []
        
        by_category = newsletter_data.get("by_category", {})
        
        for category, articles in by_category.items():
            if articles:
                lines.append(f"\n### {category.upper()} ({len(articles)} articles)")
                for article in articles[:5]:  # Top 5 per category
                    lines.append(f"\n**{article.get('title', 'Untitled')}**")
                    lines.append(f"Source: {article.get('source_name', 'Unknown')}")
                    lines.append(f"Impact: {article.get('impact_score', 0):.2f}")
                    if article.get('extracted_facts'):
                        lines.append(f"Key facts: {'; '.join(article['extracted_facts'][:3])}")
                    if article.get('dollar_amounts'):
                        lines.append(f"Amounts: {', '.join(article['dollar_amounts'][:3])}")
        
        return "\n".join(lines)
    
    def _build_indicator_context(self, bubble_reading: Optional[Dict] = None) -> str:
        """Build context string from bubble index reading"""
        if not bubble_reading:
            return "No current market indicator data available."
        
        lines = [
            f"Current Market Indicators:",
            f"- VIX: {bubble_reading.get('vix', 'N/A')} (z-score: {bubble_reading.get('vix_zscore', 'N/A')})",
            f"- CAPE: {bubble_reading.get('cape', 'N/A')} (z-score: {bubble_reading.get('cape_zscore', 'N/A')})",
            f"- Credit Spread IG: {bubble_reading.get('credit_spread_ig', 'N/A')} bps",
            f"- Credit Spread HY: {bubble_reading.get('credit_spread_hy', 'N/A')} bps",
            f"- Bubble Index: {bubble_reading.get('bubble_index', 'N/A')}/100",
            f"- Bifurcation Score: {bubble_reading.get('bifurcation_score', 'N/A')}",
            f"- Current Regime: {bubble_reading.get('regime', 'N/A')}",
        ]
        
        return "\n".join(lines)
    
    def generate_lead_section(
        self,
        newsletter_data: Dict[str, Any],
        bubble_reading: Optional[Dict] = None,
    ) -> NewsletterSection:
        """Generate the opening/lead section"""
        
        system_prompt = f"""You are writing the opening section of Y2AI Weekly, a newsletter 
analyzing AI infrastructure investment through the lens of whether it represents a genuine 
upgrade cycle or a speculative bubble.

{WRITING_STYLE_GUIDELINES}

{Y2AI_FRAMEWORK}

Your task: Write a compelling 2-3 paragraph opening that captures the week's most important 
development and connects it to the broader thesis. Lead with the news, then provide context."""

        article_context = self._build_article_context(newsletter_data)
        indicator_context = self._build_indicator_context(bubble_reading)
        
        user_prompt = f"""Write the lead section for this week's Y2AI Weekly.

{indicator_context}

This week's processed articles:
{article_context}

Sentiment distribution: {newsletter_data.get('sentiment_distribution', {})}
High-impact articles: {len(newsletter_data.get('high_impact_articles', []))}
Key companies mentioned: {', '.join(newsletter_data.get('top_companies', [])[:10])}

Write 2-3 paragraphs that open the newsletter. Lead with the most significant development, 
then connect it to the Y2AI infrastructure thesis. Do NOT use bullet points."""

        try:
            content = self._call_claude(system_prompt, user_prompt)
            
            return NewsletterSection(
                title="This Week in AI Infrastructure",
                content=content,
                section_type="lead",
                sources=[a.get('title', '') for a in newsletter_data.get('high_impact_articles', [])[:3]]
            )
        except Exception as e:
            logger.error(f"Failed to generate lead section: {e}")
            return NewsletterSection(
                title="This Week in AI Infrastructure",
                content="[Lead section generation failed. Please review articles manually.]",
                section_type="lead"
            )
    
    def generate_spending_analysis(
        self,
        newsletter_data: Dict[str, Any],
    ) -> NewsletterSection:
        """Generate the spending/capex analysis section"""
        
        spending_articles = newsletter_data.get("by_category", {}).get("spending", [])
        constraints_articles = newsletter_data.get("by_category", {}).get("constraints", [])
        
        if not spending_articles and not constraints_articles:
            return NewsletterSection(
                title="Infrastructure Spending Update",
                content="No significant spending announcements this week.",
                section_type="analysis"
            )
        
        system_prompt = f"""You are analyzing capital expenditure and infrastructure investment 
news for Y2AI Weekly. Focus on concrete numbers, company commitments, and what they signal 
about the infrastructure cycle.

{WRITING_STYLE_GUIDELINES}

Your task: Synthesize the spending news into 2-3 paragraphs of analysis. Highlight the 
most significant commitments, put them in context of annual spending, and note any 
constraints or bottlenecks mentioned."""

        spending_context = "\n".join([
            f"- {a.get('title')}: {'; '.join(a.get('extracted_facts', [])[:2])}"
            for a in spending_articles[:5]
        ])
        
        constraints_context = "\n".join([
            f"- {a.get('title')}: {'; '.join(a.get('extracted_facts', [])[:2])}"
            for a in constraints_articles[:3]
        ])
        
        user_prompt = f"""Analyze this week's infrastructure spending news.

SPENDING/CAPEX NEWS:
{spending_context or 'No major spending announcements'}

Dollar amounts mentioned: {newsletter_data.get('dollar_amounts', [])[:10]}

CONSTRAINTS/BOTTLENECKS:
{constraints_context or 'No constraint news'}

Write 2-3 paragraphs analyzing the spending picture. What's the scale? How does it compare 
to previous guidance? Are there any constraints affecting deployment? NO bullet points."""

        try:
            content = self._call_claude(system_prompt, user_prompt)
            
            return NewsletterSection(
                title="Infrastructure Spending Analysis",
                content=content,
                section_type="analysis",
                sources=[a.get('title', '') for a in spending_articles[:3]]
            )
        except Exception as e:
            logger.error(f"Failed to generate spending analysis: {e}")
            return NewsletterSection(
                title="Infrastructure Spending Analysis",
                content="[Spending analysis generation failed.]",
                section_type="analysis"
            )
    
    def generate_thesis_status(
        self,
        bubble_reading: Optional[Dict] = None,
        newsletter_data: Optional[Dict] = None,
    ) -> NewsletterSection:
        """Generate the thesis status/framework update section"""
        
        system_prompt = f"""You are updating readers on the Y2AI thesis status - whether 
current evidence supports the "infrastructure cycle" interpretation or the "bubble" interpretation.

{WRITING_STYLE_GUIDELINES}

{Y2AI_FRAMEWORK}

Your task: Write 2 paragraphs assessing the current state of the thesis based on the 
indicators and recent news. Be specific about what the data shows and what would change 
your assessment."""

        indicator_context = self._build_indicator_context(bubble_reading)
        
        # Get skepticism vs positive news balance
        skepticism_count = 0
        positive_count = 0
        if newsletter_data:
            by_category = newsletter_data.get("by_category", {})
            skepticism_count = len(by_category.get("skepticism", []))
            positive_count = len(by_category.get("spending", [])) + len(by_category.get("adoption", []))
        
        user_prompt = f"""Write the thesis status update.

{indicator_context}

News balance this week:
- Infrastructure/adoption positive signals: {positive_count} articles
- Skepticism/bubble warning signals: {skepticism_count} articles

Based on the bifurcation score and news flow, write 2 paragraphs on where the thesis stands.
Is the evidence supporting "infrastructure cycle" or showing "bubble warning" signs?
What specific evidence supports your assessment? NO bullet points."""

        try:
            content = self._call_claude(system_prompt, user_prompt)
            
            regime = bubble_reading.get('regime', 'UNKNOWN') if bubble_reading else 'UNKNOWN'
            
            return NewsletterSection(
                title=f"Thesis Status: {regime}",
                content=content,
                section_type="thesis"
            )
        except Exception as e:
            logger.error(f"Failed to generate thesis status: {e}")
            return NewsletterSection(
                title="Thesis Status",
                content="[Thesis status generation failed.]",
                section_type="thesis"
            )
    
    def generate_outlook(
        self,
        newsletter_data: Dict[str, Any],
        bubble_reading: Optional[Dict] = None,
    ) -> NewsletterSection:
        """Generate the forward-looking outlook section"""
        
        system_prompt = f"""You are writing the outlook section of Y2AI Weekly, looking ahead 
to what to watch in the coming week.

{WRITING_STYLE_GUIDELINES}

Your task: Write 1-2 paragraphs on what to watch next week. Focus on upcoming earnings, 
events, or thresholds that could affect the thesis. Be specific about dates and what 
outcomes would signal."""

        # Get data and policy articles for forward-looking context
        data_articles = newsletter_data.get("by_category", {}).get("data", [])
        policy_articles = newsletter_data.get("by_category", {}).get("policy", [])
        
        user_prompt = f"""Write the forward-looking outlook section.

Recent data/earnings coverage:
{[a.get('title', '') for a in data_articles[:3]]}

Policy/regulatory news:
{[a.get('title', '') for a in policy_articles[:3]]}

Companies in focus: {newsletter_data.get('top_companies', [])[:5]}

Write 1-2 paragraphs on what to watch next week. What earnings are coming? Any policy 
decisions pending? What indicator levels would trigger reassessment? NO bullet points."""

        try:
            content = self._call_claude(system_prompt, user_prompt, max_tokens=1000)
            
            return NewsletterSection(
                title="What to Watch",
                content=content,
                section_type="outlook"
            )
        except Exception as e:
            logger.error(f"Failed to generate outlook: {e}")
            return NewsletterSection(
                title="What to Watch",
                content="[Outlook generation failed.]",
                section_type="outlook"
            )
    
    def generate_social_posts(
        self,
        newsletter: GeneratedNewsletter,
    ) -> List[SocialPost]:
        """Generate social media posts for the newsletter"""
        
        posts = []
        
        # Twitter/X thread
        try:
            twitter_post = self._generate_twitter_thread(newsletter)
            posts.append(twitter_post)
        except Exception as e:
            logger.error(f"Failed to generate Twitter post: {e}")
        
        # LinkedIn post
        try:
            linkedin_post = self._generate_linkedin_post(newsletter)
            posts.append(linkedin_post)
        except Exception as e:
            logger.error(f"Failed to generate LinkedIn post: {e}")
        
        # Bluesky post
        try:
            bluesky_post = self._generate_bluesky_post(newsletter)
            posts.append(bluesky_post)
        except Exception as e:
            logger.error(f"Failed to generate Bluesky post: {e}")
        
        return posts
    
    def _generate_twitter_thread(self, newsletter: GeneratedNewsletter) -> SocialPost:
        """Generate a Twitter thread"""
        
        system_prompt = """You are creating a Twitter thread for Y2AI Weekly. 
Each tweet must be under 280 characters. The thread should be 3-5 tweets.
First tweet hooks, middle tweets deliver key insight, final tweet has the link call-to-action.
Use numbers and concrete data. No hashtags in the main text (add separately)."""

        lead_content = newsletter.lead_section.content if newsletter.lead_section else ""
        
        user_prompt = f"""Create a Twitter thread for Y2AI Weekly Edition #{newsletter.edition_number}.

Key data:
- Bubble Index: {newsletter.bubble_index}
- Bifurcation Score: {newsletter.bifurcation_score}
- Regime: {newsletter.regime}

Lead content summary:
{lead_content[:500]}

Create 3-5 tweets. Format as:
1/ [first tweet]
2/ [second tweet]
etc.

Each tweet MUST be under 280 characters."""

        content = self._call_claude(system_prompt, user_prompt, max_tokens=800)
        
        # Parse into thread
        thread = []
        for line in content.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('🧵')):
                # Remove numbering
                tweet = line.lstrip('0123456789/.) ').strip()
                if tweet and len(tweet) <= 280:
                    thread.append(tweet)
        
        return SocialPost(
            platform="twitter",
            content=thread[0] if thread else "",
            hashtags=["AI", "Infrastructure", "Y2AI", "Markets"],
            thread=thread
        )
    
    def _generate_linkedin_post(self, newsletter: GeneratedNewsletter) -> SocialPost:
        """Generate a LinkedIn post"""
        
        system_prompt = """You are creating a LinkedIn post for Y2AI Weekly.
Keep it professional but engaging. 150-300 words. Lead with insight, not promotion.
Include a clear takeaway for readers. End with newsletter link CTA."""

        lead_content = newsletter.lead_section.content if newsletter.lead_section else ""
        thesis_content = newsletter.thesis_status.content if newsletter.thesis_status else ""
        
        user_prompt = f"""Create a LinkedIn post for Y2AI Weekly Edition #{newsletter.edition_number}.

Key data:
- Bubble Index: {newsletter.bubble_index}
- Regime: {newsletter.regime}

Lead insight:
{lead_content[:400]}

Thesis status:
{thesis_content[:400]}

Write a 150-300 word LinkedIn post. Professional tone, lead with the key insight,
end with newsletter CTA."""

        content = self._call_claude(system_prompt, user_prompt, max_tokens=600)
        
        return SocialPost(
            platform="linkedin",
            content=content,
            hashtags=["AI", "Infrastructure", "Investment", "Technology"]
        )
    
    def _generate_bluesky_post(self, newsletter: GeneratedNewsletter) -> SocialPost:
        """Generate a Bluesky post"""
        
        system_prompt = """You are creating a Bluesky post for Y2AI Weekly.
Under 300 characters. Conversational, insightful, no hashtags in text.
Lead with the most interesting finding."""

        user_prompt = f"""Create a Bluesky post for Y2AI Weekly #{newsletter.edition_number}.

Bubble Index: {newsletter.bubble_index}, Regime: {newsletter.regime}

One punchy post under 300 characters summarizing the key insight."""

        content = self._call_claude(system_prompt, user_prompt, max_tokens=200)
        
        # Ensure under 300 chars
        if len(content) > 300:
            content = content[:297] + "..."
        
        return SocialPost(
            platform="bluesky",
            content=content,
            hashtags=[]
        )
    
    def generate_full_newsletter(
        self,
        edition_number: int,
        newsletter_data: Dict[str, Any],
        bubble_reading: Optional[Dict] = None,
        generate_social: bool = True,
    ) -> GeneratedNewsletter:
        """
        Generate a complete newsletter from processed articles and market data.
        
        Args:
            edition_number: Newsletter edition number
            newsletter_data: Output from NewsletterProcessor.prepare_newsletter_data()
            bubble_reading: Output from BubbleIndexCalculator.calculate().to_dict()
            generate_social: Whether to generate social media posts
        
        Returns:
            GeneratedNewsletter with all sections
        """
        if not self.is_available():
            logger.error("Newsletter generator not available")
            return GeneratedNewsletter(
                edition_number=edition_number,
                date=datetime.now().strftime("%Y-%m-%d"),
                title="Y2AI Weekly [Generation Failed]"
            )
        
        logger.info(f"Generating Y2AI Weekly Edition #{edition_number}...")
        
        # Extract indicator values
        bi = bubble_reading.get('bubble_index') if bubble_reading else None
        bf = bubble_reading.get('bifurcation_score') if bubble_reading else None
        regime = bubble_reading.get('regime') if bubble_reading else None
        
        newsletter = GeneratedNewsletter(
            edition_number=edition_number,
            date=datetime.now().strftime("%B %d, %Y"),
            title=f"Y2AI Weekly #{edition_number}",
            bubble_index=bi,
            bifurcation_score=bf,
            regime=regime,
        )
        
        # Generate sections
        logger.info("  Generating lead section...")
        newsletter.lead_section = self.generate_lead_section(newsletter_data, bubble_reading)
        
        logger.info("  Generating spending analysis...")
        newsletter.spending_analysis = self.generate_spending_analysis(newsletter_data)
        
        logger.info("  Generating thesis status...")
        newsletter.thesis_status = self.generate_thesis_status(bubble_reading, newsletter_data)
        
        logger.info("  Generating outlook...")
        newsletter.outlook = self.generate_outlook(newsletter_data, bubble_reading)
        
        # Collect sources
        for section in [newsletter.lead_section, newsletter.spending_analysis]:
            if section and section.sources:
                for source in section.sources:
                    if source:
                        newsletter.sources_cited.append({"title": source, "url": ""})
        
        # Extract key data points
        if newsletter_data.get('dollar_amounts'):
            newsletter.key_data_points.extend(newsletter_data['dollar_amounts'][:5])
        
        # Generate social posts
        if generate_social:
            logger.info("  Generating social media posts...")
            newsletter.social_posts = self.generate_social_posts(newsletter)
        
        logger.info(f"Newsletter generation complete: {len([s for s in [newsletter.lead_section, newsletter.spending_analysis, newsletter.thesis_status, newsletter.outlook] if s])} sections")
        
        return newsletter
    
    def get_health(self) -> Dict[str, Any]:
        """Get generator health status"""
        tracker = get_health_tracker("anthropic_newsletter")
        return {
            "available": self.is_available(),
            **tracker.to_dict()
        }


# =============================================================================
# QUICK GENERATION HELPERS
# =============================================================================

def generate_quick_summary(
    articles: List[Dict],
    bubble_reading: Optional[Dict] = None,
) -> str:
    """
    Generate a quick 1-paragraph summary without full newsletter.
    
    Useful for daily updates or quick posts.
    """
    generator = NewsletterGenerator()
    
    if not generator.is_available():
        return "[Summary generation unavailable]"
    
    system_prompt = f"""Write a single paragraph (4-5 sentences) summarizing the day's 
AI infrastructure news. Lead with the most important development.

{WRITING_STYLE_GUIDELINES}"""

    article_summaries = "\n".join([
        f"- {a.get('title', 'Untitled')}: {a.get('extracted_facts', [''])[0] if a.get('extracted_facts') else ''}"
        for a in articles[:10]
    ])
    
    indicator_str = ""
    if bubble_reading:
        indicator_str = f"Bubble Index: {bubble_reading.get('bubble_index')}, Regime: {bubble_reading.get('regime')}"
    
    user_prompt = f"""Summarize today's AI infrastructure news in one paragraph.

{indicator_str}

Articles:
{article_summaries}

Write one paragraph, 4-5 sentences. NO bullet points."""

    try:
        return generator._call_claude(system_prompt, user_prompt, max_tokens=500)
    except Exception as e:
        logger.error(f"Quick summary failed: {e}")
        return "[Summary generation failed]"


def generate_single_tweet(
    headline: str,
    bubble_index: Optional[float] = None,
) -> str:
    """Generate a single tweet about a news item"""
    generator = NewsletterGenerator()
    
    if not generator.is_available():
        return headline[:280]
    
    system_prompt = "Write a single tweet under 280 characters. Be insightful, not just descriptive."
    
    bi_context = f" (Bubble Index: {bubble_index:.0f})" if bubble_index else ""
    
    user_prompt = f"""Write a tweet about this news{bi_context}:

{headline}

Under 280 characters. Insightful commentary, not just restating the headline."""

    try:
        tweet = generator._call_claude(system_prompt, user_prompt, max_tokens=100)
        return tweet[:280]
    except Exception:
        return headline[:280]


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Y2AI Weekly newsletter")
    parser.add_argument("--edition", "-e", type=int, required=True, help="Edition number")
    parser.add_argument("--data", "-d", help="Path to newsletter data JSON")
    parser.add_argument("--bubble", "-b", help="Path to bubble reading JSON")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--format", "-f", choices=["json", "markdown"], default="markdown")
    parser.add_argument("--no-social", action="store_true", help="Skip social media posts")
    
    args = parser.parse_args()
    
    # Load data
    newsletter_data = {}
    if args.data:
        with open(args.data) as f:
            newsletter_data = json.load(f)
    
    bubble_reading = None
    if args.bubble:
        with open(args.bubble) as f:
            bubble_reading = json.load(f)
    
    # Generate
    generator = NewsletterGenerator()
    
    if not generator.is_available():
        print("Error: Newsletter generator not available. Check ANTHROPIC_API_KEY.")
        exit(1)
    
    newsletter = generator.generate_full_newsletter(
        edition_number=args.edition,
        newsletter_data=newsletter_data,
        bubble_reading=bubble_reading,
        generate_social=not args.no_social,
    )
    
    # Output
    if args.format == "json":
        output = json.dumps(newsletter.to_dict(), indent=2)
    else:
        output = newsletter.to_markdown()
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Newsletter written to {args.output}")
    else:
        print(output)
    
    # Print social posts
    if newsletter.social_posts and not args.no_social:
        print("\n" + "="*60)
        print("SOCIAL MEDIA POSTS")
        print("="*60)
        
        for post in newsletter.social_posts:
            print(f"\n### {post.platform.upper()}")
            if post.thread:
                for i, tweet in enumerate(post.thread, 1):
                    print(f"{i}/ {tweet}")
            else:
                print(post.content)
