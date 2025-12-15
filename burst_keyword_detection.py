"""
burst_keyword_detection.py
Detects sudden spikes in keyword frequency from processed articles

Burst measures whether specific keywords are appearing at abnormal rates,
indicating potential market-moving narratives forming.

Regimes:
  HIGH_BURST:   3+ keywords with >2x normal frequency
  MODERATE:     1-2 keywords with elevated frequency
  NORMAL:       No significant keyword spikes detected

Location: argus1/burst_keyword_detection.py
"""

import os
from datetime import datetime, timedelta
from collections import Counter
from dotenv import load_dotenv
load_dotenv()

# Database
from supabase import create_client, Client

# Google Sheets
import gspread
from oauth2client.service_account import ServiceAccountCredentials


# ============================================================
# CONFIGURATION
# ============================================================

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

# Google Sheets config (matching NCI/NPD pattern)
GOOGLE_SHEETS_CREDS_FILE = 'credentials.json'
SPREADSHEET_NAME = 'Copy of Copy of Version 3.3 - Development Copy (Active)'
BURST_SHEET_NAME = 'Burst_Dial'

# Burst calculation parameters
BURST_CONFIG = {
    'baseline_days': 14,        # Days to establish baseline frequency
    'current_days': 1,          # Days to measure current frequency
    'burst_threshold': 2.0,     # 2x baseline = burst
    'high_burst_count': 3,      # 3+ bursting keywords = HIGH_BURST
    'moderate_burst_count': 1,  # 1-2 bursting keywords = MODERATE
    'min_baseline_count': 3,    # Keyword must appear 3+ times in baseline to qualify
}

# Keywords to track (AI infrastructure focused)
TRACKED_KEYWORDS = [
    'bubble', 'crash', 'correction', 'selloff', 'capitulation',
    'data center', 'power grid', 'energy', 'infrastructure',
    'nvidia', 'openai', 'microsoft', 'google', 'meta',
    'regulation', 'antitrust', 'china', 'tariff', 'export',
    'layoff', 'hiring freeze', 'cost cutting',
    'ipo', 'funding', 'valuation', 'investment',
    'breakthrough', 'agi', 'superintelligence',
]


# ============================================================
# DATABASE FUNCTIONS
# ============================================================

def get_supabase_client() -> Client:
    """Get Supabase client"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables required")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def get_article_text_history(supabase: Client, days: int = 14):
    """Pull article titles and extracted facts for keyword analysis."""
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    
    response = supabase.table('processed_articles') \
        .select('processed_at, title, extracted_facts, one_line_summary, y2ai_category') \
        .gte('processed_at', cutoff) \
        .order('processed_at', desc=True) \
        .limit(5000) \
        .execute()
    
    return response.data if response.data else []


# ============================================================
# GOOGLE SHEETS FUNCTIONS
# ============================================================

def get_sheets_client():
    """Get authenticated Google Sheets client"""
    scope = [
        'https://spreadsheets.google.com/feeds',
        'https://www.googleapis.com/auth/drive'
    ]
    
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        GOOGLE_SHEETS_CREDS_FILE, scope
    )
    return gspread.authorize(creds)


def write_burst_to_sheets(burst_data: dict):
    """Write Burst result to Google Sheets Burst_Dial tab."""
    try:
        print("Getting sheets client...")
        client = get_sheets_client()
        
        print(f"Opening spreadsheet: {SPREADSHEET_NAME}")
        spreadsheet = client.open(SPREADSHEET_NAME)
        
        print("Looking for Burst_Dial sheet...")
        try:
            sheet = spreadsheet.worksheet(BURST_SHEET_NAME)
            print("Found existing Burst_Dial sheet")
        except gspread.WorksheetNotFound:
            print("Creating Burst_Dial sheet...")
            sheet = spreadsheet.add_worksheet(title=BURST_SHEET_NAME, rows=1000, cols=10)
            headers = [
                'Date', 'Burst Count', 'Regime', 'Top Keyword', 'Top Burst Ratio',
                'Article Count', 'Bursting Keywords', 'Interpretation'
            ]
            sheet.append_row(headers)
            
            # Format header row
            sheet.format('A1:H1', {
                'textFormat': {'bold': True},
                'backgroundColor': {'red': 0.2, 'green': 0.1, 'blue': 0.1}
            })
            print("Created Burst_Dial with headers")
        
        # Prepare row
        row = [
            datetime.utcnow().strftime('%Y-%m-%d'),
            burst_data['burst_count'],
            burst_data['regime'],
            burst_data['top_keyword'],
            burst_data['top_burst_ratio'],
            burst_data['article_count'],
            ', '.join(burst_data['bursting_keywords'][:5]),  # Top 5
            burst_data['interpretation']
        ]
        
        print("Appending row...")
        sheet.append_row(row)
        print(f"Wrote Burst to sheets: {burst_data['burst_count']} ({burst_data['regime']})")
        return True
        
    except Exception as e:
        print(f"Error writing to sheets: {type(e).__name__}: {e}")
        return False


# ============================================================
# BURST CALCULATION FUNCTIONS
# ============================================================

def extract_keywords_from_article(article: dict) -> list:
    """Extract tracked keywords from article title and available text fields."""
    # Build text from available fields
    text_parts = [article.get('title', '')]
    
    # Add one_line_summary if exists
    if article.get('one_line_summary'):
        text_parts.append(article.get('one_line_summary'))
    
    # Extract text from extracted_facts JSONB if exists
    facts = article.get('extracted_facts')
    if facts and isinstance(facts, dict):
        for key, value in facts.items():
            if isinstance(value, str):
                text_parts.append(value)
    
    text = ' '.join(text_parts).lower()
    
    found = []
    for keyword in TRACKED_KEYWORDS:
        if keyword.lower() in text:
            found.append(keyword.lower())
    
    return found


def calculate_keyword_frequencies(articles: list, days_back: int = None) -> Counter:
    """Count keyword occurrences in articles."""
    keyword_counts = Counter()
    
    cutoff = None
    if days_back:
        cutoff = datetime.utcnow() - timedelta(days=days_back)
    
    for article in articles:
        # Filter by date if specified
        if cutoff:
            processed_at = article.get('processed_at', '')
            if processed_at:
                article_date = datetime.fromisoformat(processed_at.replace('Z', '+00:00'))
                if article_date.replace(tzinfo=None) < cutoff:
                    continue
        
        keywords = extract_keywords_from_article(article)
        keyword_counts.update(keywords)
    
    return keyword_counts


def calculate_burst(articles: list) -> tuple:
    """
    Calculate Keyword Burst Detection.
    
    Compares current keyword frequency to baseline frequency.
    Burst ratio = current_freq / baseline_freq
    
    Returns:
        tuple: (burst_data dict, error string or None)
    """
    if not articles:
        return None, "No articles to analyze"
    
    # Split articles into baseline and current periods
    baseline_cutoff = datetime.utcnow() - timedelta(days=BURST_CONFIG['baseline_days'])
    current_cutoff = datetime.utcnow() - timedelta(days=BURST_CONFIG['current_days'])
    
    baseline_articles = []
    current_articles = []
    
    for article in articles:
        processed_at = article.get('processed_at', '')
        if not processed_at:
            continue
        
        try:
            article_date = datetime.fromisoformat(processed_at.replace('Z', '+00:00')).replace(tzinfo=None)
        except:
            continue
        
        if article_date >= current_cutoff:
            current_articles.append(article)
        elif article_date >= baseline_cutoff:
            baseline_articles.append(article)
    
    if not baseline_articles:
        return None, "Insufficient baseline data"
    
    if not current_articles:
        return None, "No current articles"
    
    # Calculate frequencies
    baseline_counts = calculate_keyword_frequencies(baseline_articles)
    current_counts = calculate_keyword_frequencies(current_articles)
    
    # Normalize by days
    baseline_days = BURST_CONFIG['baseline_days'] - BURST_CONFIG['current_days']
    current_days = BURST_CONFIG['current_days']
    
    # Find bursting keywords
    bursting = []
    for keyword in TRACKED_KEYWORDS:
        keyword_lower = keyword.lower()
        baseline_freq = baseline_counts.get(keyword_lower, 0) / max(baseline_days, 1)
        current_freq = current_counts.get(keyword_lower, 0) / max(current_days, 1)
        
        # Must have minimum baseline presence
        if baseline_counts.get(keyword_lower, 0) < BURST_CONFIG['min_baseline_count']:
            continue
        
        # Calculate burst ratio
        if baseline_freq > 0:
            burst_ratio = current_freq / baseline_freq
            if burst_ratio >= BURST_CONFIG['burst_threshold']:
                bursting.append({
                    'keyword': keyword,
                    'burst_ratio': round(burst_ratio, 2),
                    'baseline_freq': round(baseline_freq, 2),
                    'current_freq': round(current_freq, 2)
                })
    
    # Sort by burst ratio
    bursting.sort(key=lambda x: x['burst_ratio'], reverse=True)
    
    # Determine regime
    burst_count = len(bursting)
    if burst_count >= BURST_CONFIG['high_burst_count']:
        regime = "HIGH_BURST"
    elif burst_count >= BURST_CONFIG['moderate_burst_count']:
        regime = "MODERATE"
    else:
        regime = "NORMAL"
    
    # Get interpretation
    interpretation = get_burst_interpretation(regime, bursting)
    
    return {
        'date': datetime.utcnow().strftime('%Y-%m-%d'),
        'burst_count': burst_count,
        'regime': regime,
        'top_keyword': bursting[0]['keyword'] if bursting else '',
        'top_burst_ratio': bursting[0]['burst_ratio'] if bursting else 0,
        'bursting_keywords': [b['keyword'] for b in bursting],
        'burst_details': bursting,
        'article_count': len(current_articles),
        'interpretation': interpretation
    }, None


def get_burst_interpretation(regime: str, bursting: list) -> str:
    """Get human-readable interpretation"""
    if regime == "HIGH_BURST":
        keywords = ', '.join([b['keyword'] for b in bursting[:3]])
        return f"High narrative activity. Keywords spiking: {keywords}. Watch for momentum shift."
    elif regime == "MODERATE":
        keyword = bursting[0]['keyword'] if bursting else 'unknown'
        return f"Elevated attention on '{keyword}'. Monitor for escalation."
    else:
        return "Normal keyword distribution. No unusual narrative activity detected."


# ============================================================
# SUPABASE OUTPUT
# ============================================================

def save_burst_to_supabase(supabase: Client, burst_data: dict):
    """Save Burst to Supabase daily_signals table."""
    today = datetime.utcnow().strftime('%Y-%m-%d')
    
    record = {
        'date': today,
        'burst_count': burst_data['burst_count'],
        'burst_regime': burst_data['regime'],
        'burst_top_keyword': burst_data['top_keyword'],
        'burst_top_ratio': burst_data['top_burst_ratio'],
        'updated_at': datetime.utcnow().isoformat()
    }
    
    try:
        response = supabase.table('daily_signals') \
            .upsert(record, on_conflict='date') \
            .execute()
        
        print(f"Saved Burst to Supabase: {burst_data['burst_count']} ({burst_data['regime']})")
        return response
    except Exception as e:
        print(f"Error saving to Supabase: {e}")
        return None


# ============================================================
# MAIN PIPELINE FUNCTIONS
# ============================================================

def run_daily_burst(write_to_sheets: bool = True, write_to_supabase: bool = True) -> dict:
    """
    Main daily pipeline function.
    Call this after article processing completes.
    """
    print("=" * 50)
    print("KEYWORD BURST DETECTION")
    print(f"Time: {datetime.utcnow().isoformat()}")
    print("=" * 50)
    
    # Get Supabase client
    supabase = get_supabase_client()
    
    # Fetch article history
    print("\n1. Fetching article history...")
    articles = get_article_text_history(supabase, days=BURST_CONFIG['baseline_days'])
    print(f"   Retrieved {len(articles)} articles")
    
    if not articles:
        print("   ERROR: No articles found")
        return {'burst_count': 0, 'regime': 'INSUFFICIENT_DATA', 'article_count': 0}
    
    # Calculate Burst
    print("\n2. Calculating keyword bursts...")
    burst_data, error = calculate_burst(articles)
    
    if error:
        print(f"   ERROR: {error}")
        return {'burst_count': 0, 'regime': 'INSUFFICIENT_DATA', 'article_count': 0}
    
    print(f"   Burst Count: {burst_data['burst_count']}")
    print(f"   Regime: {burst_data['regime']}")
    print(f"   Top Keyword: {burst_data['top_keyword']} ({burst_data['top_burst_ratio']}x)")
    print(f"   Bursting Keywords: {', '.join(burst_data['bursting_keywords'][:5])}")
    print(f"   Interpretation: {burst_data['interpretation']}")
    
    # Write to Google Sheets
    if write_to_sheets:
        print("\n3. Writing to Google Sheets...")
        write_burst_to_sheets(burst_data)
    
    # Save to Supabase
    if write_to_supabase:
        print("\n4. Saving to Supabase...")
        save_burst_to_supabase(supabase, burst_data)
    
    print("\n" + "=" * 50)
    print("BURST DETECTION COMPLETE")
    print("=" * 50)
    
    return burst_data


# ============================================================
# TEST FUNCTION
# ============================================================

def test_burst():
    """Test Burst calculation with sample data"""
    print("=== BURST DETECTION TEST ===")
    
    # Simulate articles with keyword spikes
    test_articles = []
    
    # Baseline period (13 days ago to 1 day ago) - normal distribution
    for i in range(13, 1, -1):
        date = (datetime.utcnow() - timedelta(days=i)).isoformat()
        # 2-3 articles per day mentioning various keywords
        test_articles.extend([
            {'processed_at': date, 'title': 'Data center growth continues', 'summary': 'Infrastructure buildout'},
            {'processed_at': date, 'title': 'Nvidia earnings preview', 'summary': 'Chip demand strong'},
        ])
    
    # Current period (today) - burst on "bubble" and "crash"
    today = datetime.utcnow().isoformat()
    for _ in range(10):  # 10 articles today mentioning bubble
        test_articles.append({
            'processed_at': today,
            'title': 'AI Bubble concerns grow',
            'summary': 'Market crash fears emerge as valuations stretch'
        })
    
    print(f"Testing with {len(test_articles)} simulated articles")
    print("Simulating burst on 'bubble' and 'crash' keywords\n")
    
    burst_data, error = calculate_burst(test_articles)
    
    if error:
        print(f"Error: {error}")
        return
    
    print(f"Burst Count: {burst_data['burst_count']}")
    print(f"Regime: {burst_data['regime']}")
    print(f"Top Keyword: {burst_data['top_keyword']} ({burst_data['top_burst_ratio']}x)")
    print(f"Bursting Keywords: {burst_data['bursting_keywords']}")
    print(f"\nInterpretation: {burst_data['interpretation']}")
    
    return burst_data


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'test':
            test_burst()
        
        elif command == 'run':
            run_daily_burst()
        
        else:
            print("Usage:")
            print("  python burst_keyword_detection.py test  - Run with sample data")
            print("  python burst_keyword_detection.py run   - Run daily Burst pipeline")
    
    else:
        # Default: run test
        test_burst()