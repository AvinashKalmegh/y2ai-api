"""
Tests for the Y2AI Newsletter Generation Module

Test coverage:
- Data models (NewsletterSection, SocialPost, GeneratedNewsletter)
- NewsletterGenerator with mocked Claude API
- Content generation for each section type
- Social media post generation
- Output formatting (JSON, Markdown)
- Helper functions
"""

import pytest
import json
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from y2ai.newsletter import (
    NewsletterSection,
    SocialPost,
    GeneratedNewsletter,
    NewsletterGenerator,
    generate_quick_summary,
    generate_single_tweet,
    WRITING_STYLE_GUIDELINES,
    Y2AI_FRAMEWORK,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_newsletter_data() -> Dict[str, Any]:
    """Sample newsletter data from NewsletterProcessor"""
    return {
        "total_articles": 15,
        "by_category": {
            "spending": [
                {
                    "title": "Microsoft Announces $80B AI Infrastructure Plan",
                    "source_name": "Reuters",
                    "impact_score": 0.9,
                    "extracted_facts": ["$80 billion investment", "FY2025 capex"],
                    "dollar_amounts": ["$80B"],
                    "companies_mentioned": ["Microsoft"],
                    "sentiment": "bullish",
                },
                {
                    "title": "Google Raises Capex Guidance to $50B",
                    "source_name": "Bloomberg",
                    "impact_score": 0.85,
                    "extracted_facts": ["$50 billion guidance", "Data center expansion"],
                    "dollar_amounts": ["$50B"],
                    "companies_mentioned": ["Google", "Alphabet"],
                    "sentiment": "bullish",
                },
            ],
            "constraints": [
                {
                    "title": "NVIDIA H100 Backlog Extends Through 2025",
                    "source_name": "WSJ",
                    "impact_score": 0.75,
                    "extracted_facts": ["Supply constraints continue", "Demand exceeds supply"],
                    "dollar_amounts": [],
                    "companies_mentioned": ["NVIDIA"],
                    "sentiment": "neutral",
                },
            ],
            "skepticism": [
                {
                    "title": "Fund Managers Warn of AI Bubble Risk",
                    "source_name": "FT",
                    "impact_score": 0.6,
                    "extracted_facts": ["Valuation concerns", "Comparison to dot-com"],
                    "dollar_amounts": [],
                    "companies_mentioned": [],
                    "sentiment": "bearish",
                },
            ],
            "data": [
                {
                    "title": "NVIDIA Q3 Earnings Beat Estimates",
                    "source_name": "CNBC",
                    "impact_score": 0.8,
                    "extracted_facts": ["EPS $0.78 vs $0.65 expected", "Revenue up 94%"],
                    "dollar_amounts": ["$18.1B revenue"],
                    "companies_mentioned": ["NVIDIA"],
                    "sentiment": "bullish",
                },
            ],
        },
        "category_counts": {
            "spending": 2,
            "constraints": 1,
            "skepticism": 1,
            "data": 1,
        },
        "top_companies": ["Microsoft", "Google", "NVIDIA", "Alphabet"],
        "dollar_amounts": ["$80B", "$50B", "$18.1B revenue"],
        "sentiment_distribution": {"bullish": 3, "bearish": 1, "neutral": 1},
        "high_impact_articles": [
            {
                "title": "Microsoft Announces $80B AI Infrastructure Plan",
                "impact_score": 0.9,
            },
        ],
        "generated_at": "2025-01-15T12:00:00",
    }


@pytest.fixture
def sample_bubble_reading() -> Dict[str, Any]:
    """Sample bubble index reading"""
    return {
        "vix": 18.5,
        "vix_zscore": 0.15,
        "cape": 32.5,
        "cape_zscore": 0.8,
        "credit_spread_ig": 95,
        "credit_spread_hy": 380,
        "bubble_index": 62.5,
        "bifurcation_score": 0.77,
        "regime": "INFRASTRUCTURE",
        "timestamp": "2025-01-15T12:00:00",
    }


@pytest.fixture
def mock_claude_response():
    """Mock Claude API response"""
    def _mock(content: str):
        mock_resp = Mock()
        mock_resp.content = [Mock(text=content)]
        return mock_resp
    return _mock


# =============================================================================
# DATA MODEL TESTS
# =============================================================================

class TestNewsletterSection:
    """Tests for NewsletterSection dataclass"""
    
    def test_section_creation(self):
        """Should create section with all fields"""
        section = NewsletterSection(
            title="Test Section",
            content="This is test content.",
            section_type="lead",
            sources=["Source 1", "Source 2"],
        )
        
        assert section.title == "Test Section"
        assert section.content == "This is test content."
        assert section.section_type == "lead"
        assert len(section.sources) == 2
    
    def test_section_to_dict(self):
        """Should serialize to dict"""
        section = NewsletterSection(
            title="Test",
            content="Content",
            section_type="analysis",
        )
        
        d = section.to_dict()
        assert d["title"] == "Test"
        assert d["content"] == "Content"
        assert d["section_type"] == "analysis"
        assert d["sources"] == []


class TestSocialPost:
    """Tests for SocialPost dataclass"""
    
    def test_twitter_post_creation(self):
        """Should create Twitter post with thread"""
        post = SocialPost(
            platform="twitter",
            content="First tweet",
            hashtags=["AI", "Tech"],
            thread=["First tweet", "Second tweet", "Third tweet"],
        )
        
        assert post.platform == "twitter"
        assert len(post.thread) == 3
        assert "AI" in post.hashtags
    
    def test_linkedin_post_creation(self):
        """Should create LinkedIn post"""
        post = SocialPost(
            platform="linkedin",
            content="Professional insight about AI infrastructure...",
            hashtags=["AI", "Investment"],
        )
        
        assert post.platform == "linkedin"
        assert len(post.content) > 0
    
    def test_post_to_dict(self):
        """Should serialize to dict"""
        post = SocialPost(
            platform="bluesky",
            content="Short post",
        )
        
        d = post.to_dict()
        assert d["platform"] == "bluesky"
        assert d["thread"] == []


class TestGeneratedNewsletter:
    """Tests for GeneratedNewsletter dataclass"""
    
    def test_newsletter_creation(self):
        """Should create newsletter with basic fields"""
        newsletter = GeneratedNewsletter(
            edition_number=5,
            date="January 15, 2025",
            title="Y2AI Weekly #5",
        )
        
        assert newsletter.edition_number == 5
        assert newsletter.date == "January 15, 2025"
        assert newsletter.generated_at  # Auto-populated
    
    def test_newsletter_with_sections(self):
        """Should create newsletter with sections"""
        lead = NewsletterSection(
            title="This Week",
            content="Lead content here.",
            section_type="lead",
        )
        
        newsletter = GeneratedNewsletter(
            edition_number=5,
            date="January 15, 2025",
            title="Y2AI Weekly #5",
            lead_section=lead,
            bubble_index=62.5,
            bifurcation_score=0.77,
            regime="INFRASTRUCTURE",
        )
        
        assert newsletter.lead_section is not None
        assert newsletter.bubble_index == 62.5
        assert newsletter.regime == "INFRASTRUCTURE"
    
    def test_newsletter_to_dict(self):
        """Should serialize full newsletter to dict"""
        newsletter = GeneratedNewsletter(
            edition_number=5,
            date="January 15, 2025",
            title="Y2AI Weekly #5",
            lead_section=NewsletterSection(
                title="Lead",
                content="Content",
                section_type="lead",
            ),
            social_posts=[
                SocialPost(platform="twitter", content="Tweet"),
            ],
        )
        
        d = newsletter.to_dict()
        assert d["edition_number"] == 5
        assert "lead_section" in d
        assert len(d["social_posts"]) == 1
    
    def test_newsletter_to_markdown(self):
        """Should export newsletter as markdown"""
        newsletter = GeneratedNewsletter(
            edition_number=5,
            date="January 15, 2025",
            title="Y2AI Weekly #5",
            lead_section=NewsletterSection(
                title="This Week in AI",
                content="Microsoft announced a massive $80B infrastructure investment.",
                section_type="lead",
            ),
            bubble_index=62.5,
            bifurcation_score=0.77,
            regime="INFRASTRUCTURE",
        )
        
        md = newsletter.to_markdown()
        
        assert "# Y2AI Weekly Edition #5" in md
        assert "January 15, 2025" in md
        assert "Bubble Index 62.5" in md
        assert "This Week in AI" in md
        assert "$80B" in md
    
    def test_newsletter_sources_in_markdown(self):
        """Should include sources in markdown output"""
        newsletter = GeneratedNewsletter(
            edition_number=5,
            date="January 15, 2025",
            title="Y2AI Weekly #5",
            sources_cited=[
                {"title": "Reuters Article", "url": "https://reuters.com/article"},
                {"title": "Bloomberg Report", "url": "https://bloomberg.com/report"},
            ],
        )
        
        md = newsletter.to_markdown()
        
        assert "### Sources" in md
        assert "Reuters Article" in md


# =============================================================================
# NEWSLETTER GENERATOR TESTS
# =============================================================================

class TestNewsletterGenerator:
    """Tests for NewsletterGenerator class"""
    
    @pytest.fixture
    def generator(self):
        """Create generator with mocked client"""
        # Create generator and replace client directly
        gen = NewsletterGenerator.__new__(NewsletterGenerator)
        gen.client = Mock()
        gen.model = "claude-sonnet-4-20250514"
        gen._available = True
        return gen
    
    def test_generator_initialization(self, generator):
        """Should initialize with client"""
        assert generator.is_available()
    
    def test_generator_unavailable_without_client(self):
        """Should handle missing client"""
        gen = NewsletterGenerator.__new__(NewsletterGenerator)
        gen.client = None
        gen._available = False
        
        assert not gen.is_available()
    
    def test_build_article_context(self, generator, sample_newsletter_data):
        """Should build article context string"""
        context = generator._build_article_context(sample_newsletter_data)
        
        assert "SPENDING" in context
        assert "Microsoft" in context
        assert "$80 billion" in context
    
    def test_build_indicator_context(self, generator, sample_bubble_reading):
        """Should build indicator context string"""
        context = generator._build_indicator_context(sample_bubble_reading)
        
        assert "VIX: 18.5" in context
        assert "Bubble Index: 62.5" in context
        assert "INFRASTRUCTURE" in context
    
    def test_build_indicator_context_empty(self, generator):
        """Should handle missing bubble reading"""
        context = generator._build_indicator_context(None)
        
        assert "No current market indicator data" in context
    
    def test_generate_lead_section(
        self, generator, sample_newsletter_data, sample_bubble_reading, mock_claude_response
    ):
        """Should generate lead section"""
        generator.client.messages.create.return_value = mock_claude_response(
            "Microsoft's $80B commitment represents the largest single infrastructure "
            "investment in AI history. This signals continued confidence in the "
            "infrastructure thesis."
        )
        
        section = generator.generate_lead_section(
            sample_newsletter_data, sample_bubble_reading
        )
        
        assert section.title == "This Week in AI Infrastructure"
        assert section.section_type == "lead"
        assert "Microsoft" in section.content or "$80B" in section.content
        generator.client.messages.create.assert_called_once()
    
    def test_generate_lead_section_failure(self, generator, sample_newsletter_data):
        """Should handle generation failure gracefully"""
        generator.client.messages.create.side_effect = Exception("API Error")
        
        section = generator.generate_lead_section(sample_newsletter_data, None)
        
        assert "failed" in section.content.lower()
    
    def test_generate_spending_analysis(
        self, generator, sample_newsletter_data, mock_claude_response
    ):
        """Should generate spending analysis section"""
        generator.client.messages.create.return_value = mock_claude_response(
            "The combined $130B in announced spending from Microsoft and Google "
            "represents a significant acceleration in infrastructure investment."
        )
        
        section = generator.generate_spending_analysis(sample_newsletter_data)
        
        assert section.section_type == "analysis"
        assert "spending" in section.title.lower() or "infrastructure" in section.title.lower()
    
    def test_generate_spending_analysis_no_articles(self, generator):
        """Should handle no spending articles"""
        empty_data = {"by_category": {}}
        
        section = generator.generate_spending_analysis(empty_data)
        
        assert "No significant" in section.content
    
    def test_generate_thesis_status(
        self, generator, sample_bubble_reading, sample_newsletter_data, mock_claude_response
    ):
        """Should generate thesis status section"""
        generator.client.messages.create.return_value = mock_claude_response(
            "With a bifurcation score of +0.77, the framework continues to support "
            "the infrastructure interpretation over bubble concerns."
        )
        
        section = generator.generate_thesis_status(sample_bubble_reading, sample_newsletter_data)
        
        assert section.section_type == "thesis"
        assert "INFRASTRUCTURE" in section.title
    
    def test_generate_outlook(
        self, generator, sample_newsletter_data, sample_bubble_reading, mock_claude_response
    ):
        """Should generate outlook section"""
        generator.client.messages.create.return_value = mock_claude_response(
            "Next week brings NVIDIA earnings and potential Fed commentary on rates."
        )
        
        section = generator.generate_outlook(sample_newsletter_data, sample_bubble_reading)
        
        assert section.section_type == "outlook"
        assert "Watch" in section.title
    
    def test_generate_twitter_thread(
        self, generator, mock_claude_response
    ):
        """Should generate Twitter thread"""
        generator.client.messages.create.return_value = mock_claude_response(
            "1/ Big week for AI infrastructure\n"
            "2/ Microsoft commits $80B\n"
            "3/ Read more in Y2AI Weekly"
        )
        
        newsletter = GeneratedNewsletter(
            edition_number=5,
            date="January 15, 2025",
            title="Y2AI Weekly #5",
            lead_section=NewsletterSection(
                title="Lead", content="Content", section_type="lead"
            ),
            bubble_index=62.5,
            bifurcation_score=0.77,
            regime="INFRASTRUCTURE",
        )
        
        post = generator._generate_twitter_thread(newsletter)
        
        assert post.platform == "twitter"
        assert len(post.thread) >= 1
        assert len(post.hashtags) > 0
    
    def test_generate_linkedin_post(
        self, generator, mock_claude_response
    ):
        """Should generate LinkedIn post"""
        generator.client.messages.create.return_value = mock_claude_response(
            "This week's AI infrastructure news reinforces our thesis that "
            "we're witnessing a genuine upgrade cycle, not a bubble."
        )
        
        newsletter = GeneratedNewsletter(
            edition_number=5,
            date="January 15, 2025",
            title="Y2AI Weekly #5",
            bubble_index=62.5,
            regime="INFRASTRUCTURE",
        )
        
        post = generator._generate_linkedin_post(newsletter)
        
        assert post.platform == "linkedin"
        assert len(post.content) > 50
    
    def test_generate_bluesky_post(
        self, generator, mock_claude_response
    ):
        """Should generate Bluesky post under 300 chars"""
        generator.client.messages.create.return_value = mock_claude_response(
            "AI infrastructure thesis holds: $130B committed this week alone."
        )
        
        newsletter = GeneratedNewsletter(
            edition_number=5,
            date="January 15, 2025",
            title="Y2AI Weekly #5",
            bubble_index=62.5,
            regime="INFRASTRUCTURE",
        )
        
        post = generator._generate_bluesky_post(newsletter)
        
        assert post.platform == "bluesky"
        assert len(post.content) <= 300
    
    def test_generate_full_newsletter(
        self, generator, sample_newsletter_data, sample_bubble_reading, mock_claude_response
    ):
        """Should generate complete newsletter"""
        # Mock all Claude calls
        generator.client.messages.create.return_value = mock_claude_response(
            "Generated content for the newsletter section."
        )
        
        newsletter = generator.generate_full_newsletter(
            edition_number=5,
            newsletter_data=sample_newsletter_data,
            bubble_reading=sample_bubble_reading,
            generate_social=True,
        )
        
        assert newsletter.edition_number == 5
        assert newsletter.bubble_index == 62.5
        assert newsletter.lead_section is not None
        assert newsletter.spending_analysis is not None
        assert newsletter.thesis_status is not None
        assert newsletter.outlook is not None
        assert len(newsletter.social_posts) > 0
    
    def test_generate_full_newsletter_no_social(
        self, generator, sample_newsletter_data, mock_claude_response
    ):
        """Should skip social media when requested"""
        generator.client.messages.create.return_value = mock_claude_response("Content")
        
        newsletter = generator.generate_full_newsletter(
            edition_number=5,
            newsletter_data=sample_newsletter_data,
            bubble_reading=None,
            generate_social=False,
        )
        
        assert len(newsletter.social_posts) == 0
    
    def test_get_health(self, generator):
        """Should return health status"""
        health = generator.get_health()
        
        assert "available" in health
        assert health["available"] is True


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================

class TestHelperFunctions:
    """Tests for standalone helper functions"""
    
    def test_generate_quick_summary(self, sample_newsletter_data):
        """Should generate quick summary"""
        articles = sample_newsletter_data["by_category"]["spending"]
        
        with patch('y2ai.newsletter.NewsletterGenerator') as MockGen:
            mock_gen = Mock()
            mock_gen.is_available.return_value = True
            mock_gen._call_claude.return_value = "Quick summary of the week's news."
            MockGen.return_value = mock_gen
            
            summary = generate_quick_summary(articles)
            
            # Either we get the mocked response or unavailable message
            assert len(summary) > 0
    
    def test_generate_quick_summary_unavailable(self):
        """Should handle unavailable generator"""
        with patch('y2ai.newsletter.NewsletterGenerator') as MockGen:
            mock_gen = Mock()
            mock_gen.is_available.return_value = False
            MockGen.return_value = mock_gen
            
            summary = generate_quick_summary([])
            
            assert "unavailable" in summary.lower()
    
    def test_generate_single_tweet(self):
        """Should generate single tweet"""
        with patch('y2ai.newsletter.NewsletterGenerator') as MockGen:
            mock_gen = Mock()
            mock_gen.is_available.return_value = True
            mock_gen._call_claude.return_value = "Microsoft's $80B bet signals infrastructure thesis intact."
            MockGen.return_value = mock_gen
            
            tweet = generate_single_tweet(
                "Microsoft Announces $80B AI Infrastructure Plan",
                bubble_index=62.5,
            )
            
            assert len(tweet) <= 280
    
    def test_generate_single_tweet_truncation(self):
        """Should truncate long tweets"""
        with patch('y2ai.newsletter.NewsletterGenerator') as MockGen:
            mock_gen = Mock()
            mock_gen.is_available.return_value = True
            mock_gen._call_claude.return_value = "X" * 300  # Too long
            MockGen.return_value = mock_gen
            
            tweet = generate_single_tweet("Test headline")
            
            assert len(tweet) <= 280


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestConfiguration:
    """Tests for module configuration"""
    
    def test_writing_style_guidelines_exist(self):
        """Writing style guidelines should be defined"""
        assert WRITING_STYLE_GUIDELINES is not None
        assert "bullet points" in WRITING_STYLE_GUIDELINES.lower()
        assert "paragraphs" in WRITING_STYLE_GUIDELINES.lower()
    
    def test_y2ai_framework_defined(self):
        """Y2AI framework should be defined"""
        assert Y2AI_FRAMEWORK is not None
        assert "Bubble Index" in Y2AI_FRAMEWORK
        assert "VIX" in Y2AI_FRAMEWORK
        assert "Bifurcation" in Y2AI_FRAMEWORK
    
    def test_y2ai_framework_includes_regimes(self):
        """Framework should define all regimes"""
        assert "INFRASTRUCTURE" in Y2AI_FRAMEWORK
        assert "ADOPTION" in Y2AI_FRAMEWORK
        assert "TRANSITION" in Y2AI_FRAMEWORK
        assert "BUBBLE_WARNING" in Y2AI_FRAMEWORK


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestNewsletterIntegration:
    """Integration-style tests"""
    
    def test_full_workflow_mock(self, sample_newsletter_data, sample_bubble_reading):
        """Should complete full workflow with mocks"""
        # Create generator with mocked client
        generator = NewsletterGenerator.__new__(NewsletterGenerator)
        generator.model = "claude-sonnet-4-20250514"
        generator._available = True
        
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Generated newsletter content.")]
        mock_client.messages.create.return_value = mock_response
        generator.client = mock_client
        
        # Generate
        newsletter = generator.generate_full_newsletter(
            edition_number=10,
            newsletter_data=sample_newsletter_data,
            bubble_reading=sample_bubble_reading,
        )
        
        # Verify
        assert newsletter.edition_number == 10
        assert newsletter.bubble_index == 62.5
        
        # Check markdown export
        md = newsletter.to_markdown()
        assert "Edition #10" in md
        
        # Check JSON export
        d = newsletter.to_dict()
        assert d["edition_number"] == 10
    
    def test_newsletter_with_empty_data(self):
        """Should handle empty input gracefully"""
        # Create generator with mocked client
        generator = NewsletterGenerator.__new__(NewsletterGenerator)
        generator.model = "claude-sonnet-4-20250514"
        generator._available = True
        
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="No significant news this week.")]
        mock_client.messages.create.return_value = mock_response
        generator.client = mock_client
        
        empty_data = {
            "total_articles": 0,
            "by_category": {},
            "category_counts": {},
            "top_companies": [],
            "dollar_amounts": [],
            "sentiment_distribution": {"bullish": 0, "bearish": 0, "neutral": 0},
            "high_impact_articles": [],
        }
        
        newsletter = generator.generate_full_newsletter(
            edition_number=1,
            newsletter_data=empty_data,
            generate_social=False,
        )
        
        assert newsletter.edition_number == 1
        # Should still have sections (even if content notes lack of news)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
