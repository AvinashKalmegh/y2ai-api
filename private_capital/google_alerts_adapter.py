"""
Google Alerts Adapter for Private Capital Tracker
==================================================

Integrates Google Alerts RSS feeds for AI funding news.

Setup Instructions:
1. Go to https://google.com/alerts
2. Create each alert below
3. Click "Show options" → Delivery: RSS feed
4. Copy the RSS URL and paste it here

Author: ARGUS-1 Development Team
Created: January 2026
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# =============================================================================
# GOOGLE ALERT CONFIGURATION
# =============================================================================

# After creating each alert in Google Alerts with RSS delivery,
# paste the RSS URLs here. They look like:
# https://www.google.com/alerts/feeds/12345678901234567890/12345678901234567890

GOOGLE_ALERT_FEEDS = {
    # Alert: "AI startup" AND ("raised" OR "funding round" OR "Series")
    'ai_funding_rounds': 'https://www.google.com/alerts/feeds/11684174711489635674/9579485282969086948',
    
    # Alert: "artificial intelligence" AND ("Series C" OR "Series D" OR "Series E")
    'ai_late_stage': 'https://www.google.co.in/alerts/feeds/11684174711489635674/5184195265249208169',
    
    # Alert: "AI startup" AND ("seed round" OR "Series A" OR "Series B")
    'ai_early_stage': 'https://www.google.co.in/alerts/feeds/11684174711489635674/12876448282444079283',
    
    # Alert: "AI" AND ("down round" OR "valuation cut" OR "flat round")
    'down_rounds': 'https://www.google.co.in/alerts/feeds/11684174711489635674/12288346869558805306',
    
    # Alert: "AI startup" AND ("acquired" OR "acquisition" OR "IPO")
    'ai_exits': 'https://www.google.co.in/alerts/feeds/11684174711489635674/355014332711286856',
    
    # Alert: ("Sequoia" OR "Andreessen" OR "a16z") AND "artificial intelligence"
    'tier1_vc': 'https://www.google.co.in/alerts/feeds/11684174711489635674/2753479771155929540',
    
    # Alert: "AI" AND ("VC pullback" OR "funding slowdown" OR "investment decline")
    'investment_pullback': 'https://www.google.co.in/alerts/feeds/11684174711489635674/17179013088719007806',
}

# The exact queries to create in Google Alerts
ALERT_QUERIES = {
    'ai_funding_rounds': '"AI startup" AND ("raised" OR "funding round" OR "Series")',
    'ai_late_stage': '"artificial intelligence" AND ("Series C" OR "Series D" OR "Series E")',
    'ai_early_stage': '"AI startup" AND ("seed round" OR "Series A" OR "Series B")',
    'down_rounds': '"AI" AND ("down round" OR "valuation cut" OR "flat round")',
    'ai_exits': '"AI startup" AND ("acquired" OR "acquisition" OR "IPO")',
    'tier1_vc': '("Sequoia" OR "Andreessen" OR "a16z") AND "artificial intelligence"',
    'investment_pullback': '"AI" AND ("VC pullback" OR "funding slowdown" OR "investment decline")',
}


# =============================================================================
# GENERIC PE/VC ALERTS (Sector-agnostic - for PE Funding Tracker)
# =============================================================================

PE_GOOGLE_ALERT_FEEDS = {
    # Alert 1: Mega Rounds
    'mega_rounds': 'https://www.google.co.in/alerts/feeds/11684174711489635674/17703966950070996622',
    
    # Alert 2: Series A Activity
    'series_a': 'https://www.google.co.in/alerts/feeds/11684174711489635674/3568358442484310603',
    
    # Alert 3: Series B Activity
    'series_b': 'https://www.google.co.in/alerts/feeds/11684174711489635674/8992225447639163417',
    
    # Alert 4: Late Stage (C/D/E)
    'late_stage': 'https://www.google.co.in/alerts/feeds/11684174711489635674/13994208772373342460',
    
    # Alert 5: IPO Pipeline
    'ipo_pipeline': 'https://www.google.co.in/alerts/feeds/11684174711489635674/12924698942897999339',
    
    # Alert 6: VC Fund Launches
    'vc_fund_launches': 'https://www.google.co.in/alerts/feeds/11684174711489635674/14906854701486242163',
}

PE_ALERT_QUERIES = {
    'mega_rounds': '"raised" AND ("$100 million" OR "$200 million" OR "$500 million" OR "$1 billion")',
    'series_a': '"Series A" AND ("raised" OR "funding" OR "closes")',
    'series_b': '"Series B" AND ("raised" OR "funding" OR "closes")',
    'late_stage': '("Series C" OR "Series D" OR "Series E" OR "growth round") AND "raised"',
    'ipo_pipeline': '("IPO" OR "going public" OR "S-1 filing") AND ("plans" OR "filed" OR "confidential")',
    'vc_fund_launches': '"venture fund" AND ("closed" OR "raised" OR "launching") AND "million"',
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class GoogleAlertEntry:
    """Single entry from Google Alerts RSS."""
    title: str
    link: str
    published: datetime
    source: str  # Which alert it came from


# =============================================================================
# GOOGLE ALERTS ADAPTER
# =============================================================================

class GoogleAlertsAdapter:
    """
    Fetches and parses Google Alerts RSS feeds for AI funding news.
    """
    
    def __init__(self):
        self.feeds = GOOGLE_ALERT_FEEDS
        self.queries = ALERT_QUERIES
    
    def get_configured_feeds(self) -> dict:
        """Return only feeds that have URLs configured."""
        return {k: v for k, v in self.feeds.items() if v}
    
    def fetch_alerts(self, hours_back: int = 168) -> List[GoogleAlertEntry]:
        """
        Fetch all configured Google Alerts RSS feeds.
        
        Args:
            hours_back: How far back to look (default 168 = 1 week)
            
        Returns:
            List of GoogleAlertEntry objects
        """
        try:
            import feedparser
        except ImportError:
            logger.error("feedparser not installed")
            return []
        
        configured = self.get_configured_feeds()
        
        if not configured:
            logger.warning("No Google Alerts RSS feeds configured")
            return []
        
        entries = []
        cutoff = datetime.utcnow() - timedelta(hours=hours_back)
        
        for alert_name, url in configured.items():
            try:
                logger.info(f"Fetching Google Alert: {alert_name}")
                feed = feedparser.parse(url)
                
                for entry in feed.entries[:20]:
                    pub_date = self._parse_date(entry)
                    
                    if pub_date and pub_date < cutoff:
                        continue
                    
                    entries.append(GoogleAlertEntry(
                        title=entry.get('title', ''),
                        link=entry.get('link', ''),
                        published=pub_date or datetime.utcnow(),
                        source=f'Alert_{alert_name}'
                    ))
                    
            except Exception as e:
                logger.error(f"Error fetching {alert_name}: {e}")
        
        logger.info(f"Google Alerts: {len(entries)} entries from {len(configured)} feeds")
        return entries
    
    def _parse_date(self, entry) -> Optional[datetime]:
        """Parse date from feed entry."""
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            return datetime(*entry.published_parsed[:6])
        if hasattr(entry, 'updated_parsed') and entry.updated_parsed:
            return datetime(*entry.updated_parsed[:6])
        return None
    
    def convert_to_funding_entries(self, alerts: List[GoogleAlertEntry]):
        """
        Convert Google Alert entries to FundingEntry objects.
        Uses the same parsing logic as the main PrivateCapitalTracker.
        """
        from .private_capital import PrivateCapitalTracker
        
        tracker = PrivateCapitalTracker()
        entries = []
        
        for alert in alerts:
            text = alert.title.lower()
            
            # Check AI relevance
            if not tracker._is_ai_related(text):
                continue
            
            # Extract amount
            amount_m = tracker._extract_amount(text)
            if not amount_m or amount_m < tracker.config.MIN_AMOUNT_M:
                continue
            
            from .private_capital import FundingEntry
            
            entry = FundingEntry(
                date=alert.published.strftime('%Y-%m-%d'),
                company=tracker._extract_company(alert.title),
                amount_m=amount_m,
                round_type=tracker._detect_round_type(text),
                category=tracker._classify_category(text),
                lead_investor=tracker._extract_investor(text),
                source=alert.source,
                url=alert.link,
                processed=False
            )
            entries.append(entry)
        
        return entries
    
    def print_setup_instructions(self):
        """Print instructions for setting up Google Alerts."""
        print("\n" + "=" * 60)
        print("GOOGLE ALERTS SETUP INSTRUCTIONS")
        print("=" * 60)
        print("\n1. Go to: https://google.com/alerts\n")
        print("2. Create these 7 alerts (copy each query exactly):\n")
        
        for name, query in self.queries.items():
            print(f"   [{name}]")
            print(f"   {query}\n")
        
        print("3. For EACH alert, click 'Show options' and set:")
        print("   - How often: As-it-happens")
        print("   - Sources: Automatic")
        print("   - Language: English")
        print("   - Region: Any region")
        print("   - How many: All results")
        print("   - Deliver to: RSS feed  <-- IMPORTANT!\n")
        
        print("4. After creating each alert, copy the RSS URL")
        print("   (looks like: https://www.google.com/alerts/feeds/...)\n")
        
        print("5. Paste the URLs into:")
        print("   private_capital/google_alerts_adapter.py")
        print("   in the GOOGLE_ALERT_FEEDS dictionary\n")
        
        print("=" * 60)


def collect_all_sources(supabase_client=None, hours_back: int = 168):
    """
    Collect from both RSS feeds and Google Alerts.
    
    This is the combined collection function that pulls from all sources.
    """
    from .private_capital import PrivateCapitalTracker
    
    tracker = PrivateCapitalTracker(supabase_client=supabase_client)
    
    # Get RSS entries
    rss_entries = tracker.fetch_rss_feeds(hours_back=hours_back)
    logger.info(f"RSS feeds: {len(rss_entries)} entries")
    
    # Get Google Alerts entries
    adapter = GoogleAlertsAdapter()
    alerts = adapter.fetch_alerts(hours_back=hours_back)
    alert_entries = adapter.convert_to_funding_entries(alerts)
    logger.info(f"Google Alerts: {len(alert_entries)} entries")
    
    # Combine and dedupe
    all_entries = rss_entries + alert_entries
    unique = tracker._dedupe_entries(all_entries)
    
    logger.info(f"Combined: {len(unique)} unique entries from {len(all_entries)} total")
    return unique


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Google Alerts Adapter')
    parser.add_argument('--setup', action='store_true', help='Print setup instructions')
    parser.add_argument('--test', action='store_true', help='Test configured feeds')
    parser.add_argument('--status', action='store_true', help='Show which feeds are configured')
    
    args = parser.parse_args()
    
    adapter = GoogleAlertsAdapter()
    
    if args.setup:
        adapter.print_setup_instructions()
    
    elif args.status:
        print("\n=== Google Alerts Configuration Status ===\n")
        configured = adapter.get_configured_feeds()
        
        for name in ALERT_QUERIES.keys():
            status = "✅ Configured" if name in configured else "❌ Not configured"
            print(f"  {name}: {status}")
        
        print(f"\n{len(configured)}/7 alerts configured\n")
    
    elif args.test:
        print("\n=== Testing Google Alerts Feeds ===\n")
        configured = adapter.get_configured_feeds()
        
        if not configured:
            print("No feeds configured. Run --setup for instructions.")
        else:
            entries = adapter.fetch_alerts(hours_back=168)
            print(f"\nFound {len(entries)} entries:\n")
            
            for entry in entries[:10]:
                print(f"  [{entry.source}] {entry.title[:60]}...")
    
    else:
        parser.print_help()