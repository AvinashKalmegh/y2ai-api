"""
npd_narrative_polarity_drift.py
Calculates Narrative Polarity Drift from processed article sentiment

NPD measures the direction and speed of sentiment shift in news coverage.
Not "is sentiment positive or negative" but "is sentiment becoming more positive or more negative?"

Regimes:
  BULLISH_DRIFT:  NPD > +0.15  (sentiment accelerating positive)
  STABLE:         NPD between -0.15 and +0.15
  BEARISH_DRIFT:  NPD < -0.15  (sentiment accelerating negative)

Location: argus1/npd_narrative_polarity_drift.py
"""

import os
from datetime import datetime, timedelta
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

# Google Sheets config (matching NCI pattern)
GOOGLE_SHEETS_CREDS_FILE = 'credentials.json'
SPREADSHEET_NAME = 'Copy of Copy of Version 3.3 - Development Copy (Active)'
NPD_SHEET_NAME = 'NPD_Dial'

# NPD calculation parameters
NPD_CONFIG = {
    'short_window': 3,      # 3-day average
    'long_window': 3,       # 7-day average
    'bullish_threshold': 0.15,
    'bearish_threshold': -0.15,
}


# ============================================================
# DATABASE FUNCTIONS
# ============================================================

def get_supabase_client() -> Client:
    """Get Supabase client"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables required")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def get_sentiment_history(supabase: Client, days: int = 14):
    """Pull sentiment from processed_articles for last N days."""
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    
    response = supabase.table('processed_articles') \
        .select('processed_at, sentiment') \
        .gte('processed_at', cutoff) \
        .order('processed_at', desc=True) \
        .limit(5000) \
        .execute()
    
    return response.data if response.data else []


def sentiment_to_score(sentiment_label: str) -> float:
    """Convert sentiment label to numeric score."""
    mapping = {
        'bullish': 1.0,
        'neutral': 0.0,
        'bearish': -1.0
    }
    return mapping.get(sentiment_label.lower() if sentiment_label else '', 0.0)


# ============================================================
# GOOGLE SHEETS FUNCTIONS
# ============================================================

def get_sheets_client():
    """Get authenticated Google Sheets client (matching NCI pattern)"""
    scope = [
        'https://spreadsheets.google.com/feeds',
        'https://www.googleapis.com/auth/drive'
    ]
    
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        GOOGLE_SHEETS_CREDS_FILE, scope
    )
    return gspread.authorize(creds)


def write_npd_to_sheets(npd_data: dict):
    """Write NPD result to Google Sheets NPD_Dial tab."""
    try:
        print("Getting sheets client...")
        client = get_sheets_client()
        
        print(f"Opening spreadsheet: {SPREADSHEET_NAME}")
        spreadsheet = client.open(SPREADSHEET_NAME)
        
        print("Looking for NPD_Dial sheet...")
        try:
            sheet = spreadsheet.worksheet(NPD_SHEET_NAME)
            print("Found existing NPD_Dial sheet")
        except gspread.WorksheetNotFound:
            print("Creating NPD_Dial sheet...")
            sheet = spreadsheet.add_worksheet(title=NPD_SHEET_NAME, rows=1000, cols=10)
            headers = [
                'Date', 'NPD Score', 'Regime', 'Short Avg (3D)', 'Long Avg (7D)',
                'Momentum', 'Article Count', 'Interpretation'
            ]
            sheet.append_row(headers)
            
            # Format header row
            sheet.format('A1:H1', {
                'textFormat': {'bold': True},
                'backgroundColor': {'red': 0.1, 'green': 0.1, 'blue': 0.2}
            })
            print("Created NPD_Dial with headers")
        
        # Prepare interpretation
        interpretation = get_npd_interpretation(npd_data['regime'])
        
        # Prepare row
        row = [
            datetime.utcnow().strftime('%Y-%m-%d'),
            npd_data['npd_score'],
            npd_data['regime'],
            npd_data['short_avg'],
            npd_data['long_avg'],
            npd_data['momentum'],
            npd_data['article_count'],
            interpretation
        ]
        
        print("Appending row...")
        sheet.append_row(row)
        print(f"Wrote NPD to sheets: {npd_data['npd_score']} ({npd_data['regime']})")
        return True
        
    except Exception as e:
        print(f"Error writing to sheets: {type(e).__name__}: {e}")
        return False


# ============================================================
# NPD CALCULATION FUNCTIONS
# ============================================================

def aggregate_daily_sentiment(articles: list) -> dict:
    """Aggregate article sentiment by day."""
    daily = {}
    
    for article in articles:
        # Extract date from processed_at
        processed_at = article.get('processed_at', '')
        date = processed_at[:10] if processed_at else None  # Extract YYYY-MM-DD
        
        if not date:
            continue
        
        # Convert sentiment label to numeric score
        sentiment_label = article.get('sentiment', '')
        score = sentiment_to_score(sentiment_label)
        
        if date not in daily:
            daily[date] = {'scores': [], 'count': 0}
        
        daily[date]['scores'].append(score)
        daily[date]['count'] += 1
    
    # Calculate daily averages
    for date in daily:
        scores = daily[date]['scores']
        daily[date]['avg'] = sum(scores) / len(scores) if scores else 0
    
    return daily


def calculate_npd(daily_sentiment: dict) -> tuple:
    """
    Calculate Narrative Polarity Drift.
    
    NPD = (short_avg - long_avg) / |long_avg|
    
    Returns:
        tuple: (npd_data dict, error string or None)
    """
    dates = sorted(daily_sentiment.keys())
    
    if len(dates) < NPD_CONFIG['long_window']:
        return None, f"Insufficient data: need {NPD_CONFIG['long_window']} days, have {len(dates)}"
    
    # Get recent date windows
    recent_dates = dates[-NPD_CONFIG['long_window']:]
    short_dates = dates[-NPD_CONFIG['short_window']:]
    
    # Calculate window averages
    long_avg = sum(daily_sentiment[d]['avg'] for d in recent_dates) / len(recent_dates)
    short_avg = sum(daily_sentiment[d]['avg'] for d in short_dates) / len(short_dates)
    
    # Calculate NPD (rate of change normalized by baseline)
    if long_avg == 0:
        npd = 0
    else:
        npd = (short_avg - long_avg) / abs(long_avg)
    
    # Determine regime
    if npd > NPD_CONFIG['bullish_threshold']:
        regime = "BULLISH_DRIFT"
    elif npd < NPD_CONFIG['bearish_threshold']:
        regime = "BEARISH_DRIFT"
    else:
        regime = "STABLE"
    
    # Momentum placeholder (would need previous NPD to calculate)
    momentum = "STABLE"
    
    return {
        'date': dates[-1],
        'npd_score': round(npd, 4),
        'regime': regime,
        'short_avg': round(short_avg, 4),
        'long_avg': round(long_avg, 4),
        'momentum': momentum,
        'article_count': sum(daily_sentiment[d]['count'] for d in short_dates),
    }, None


def get_npd_interpretation(regime: str) -> str:
    """Get human-readable interpretation"""
    interpretations = {
        'BULLISH_DRIFT': 'Narrative shifting positive. Watch for flow reversal to bullish.',
        'BEARISH_DRIFT': 'Narrative shifting negative. Watch for flow reversal to bearish.',
        'STABLE': 'Narrative stable. No significant drift detected.',
        'INSUFFICIENT_DATA': 'Not enough data to calculate drift.'
    }
    return interpretations.get(regime, 'Unable to assess.')


# ============================================================
# SUPABASE OUTPUT
# ============================================================

def save_npd_to_supabase(supabase: Client, npd_data: dict):
    """Save NPD to Supabase daily_signals table (upsert - creates row if missing)."""
    today = datetime.utcnow().strftime('%Y-%m-%d')
    
    record = {
        'date': today,
        'npd_score': npd_data['npd_score'],
        'npd_regime': npd_data['regime'],
        'npd_short_avg': npd_data['short_avg'],
        'npd_long_avg': npd_data['long_avg'],
        'updated_at': datetime.utcnow().isoformat()
    }
    
    try:
        # Upsert: insert if missing, update if exists
        response = supabase.table('daily_signals') \
            .upsert(record, on_conflict='date') \
            .execute()
        
        print(f"Saved NPD to Supabase: {npd_data['npd_score']} ({npd_data['regime']})")
        return response
    except Exception as e:
        print(f"Error saving to Supabase: {e}")
        return None


# ============================================================
# MAIN PIPELINE FUNCTIONS
# ============================================================

def run_daily_npd(write_to_sheets: bool = True, write_to_supabase: bool = True) -> dict:
    """
    Main daily pipeline function.
    Call this after article processing completes.
    """
    print("=" * 50)
    print("NARRATIVE POLARITY DRIFT (NPD) CALCULATION")
    print(f"Time: {datetime.utcnow().isoformat()}")
    print("=" * 50)
    
    # Get Supabase client
    supabase = get_supabase_client()
    
    # Fetch sentiment history
    print("\n1. Fetching sentiment history...")
    articles = get_sentiment_history(supabase, days=14)
    print(f"   Retrieved {len(articles)} articles")
    
    if not articles:
        print("   ERROR: No articles found")
        return {'npd_score': 0, 'regime': 'INSUFFICIENT_DATA', 'article_count': 0}
    
    # Aggregate by day
    print("\n2. Aggregating daily sentiment...")
    daily_sentiment = aggregate_daily_sentiment(articles)
    print(f"   Aggregated {len(daily_sentiment)} days")
    
    for date in sorted(daily_sentiment.keys())[-5:]:
        data = daily_sentiment[date]
        print(f"   {date}: avg={data['avg']:.3f}, count={data['count']}")
    
    # Calculate NPD
    print("\n3. Calculating NPD...")
    npd_data, error = calculate_npd(daily_sentiment)
    
    if error:
        print(f"   ERROR: {error}")
        return {'npd_score': 0, 'regime': 'INSUFFICIENT_DATA', 'article_count': 0}
    
    print(f"   NPD Score: {npd_data['npd_score']}")
    print(f"   Regime: {npd_data['regime']}")
    print(f"   Short Avg (3D): {npd_data['short_avg']}")
    print(f"   Long Avg (7D): {npd_data['long_avg']}")
    print(f"   Interpretation: {get_npd_interpretation(npd_data['regime'])}")
    
    # Write to Google Sheets
    if write_to_sheets:
        print("\n4. Writing to Google Sheets...")
        write_npd_to_sheets(npd_data)
    
    # Save to Supabase
    if write_to_supabase:
        print("\n5. Saving to Supabase...")
        save_npd_to_supabase(supabase, npd_data)
    
    print("\n" + "=" * 50)
    print("NPD CALCULATION COMPLETE")
    print("=" * 50)
    
    return npd_data


# ============================================================
# TEST FUNCTION
# ============================================================

def test_npd():
    """Test NPD calculation with sample data"""
    print("=== NPD TEST ===")
    
    # Simulate 10 days of sentiment data
    test_daily = {
        '2025-12-05': {'avg': 0.20, 'count': 8},
        '2025-12-06': {'avg': 0.15, 'count': 10},
        '2025-12-07': {'avg': 0.10, 'count': 12},
        '2025-12-08': {'avg': 0.05, 'count': 9},
        '2025-12-09': {'avg': -0.05, 'count': 11},
        '2025-12-10': {'avg': -0.10, 'count': 8},
        '2025-12-11': {'avg': -0.15, 'count': 10},
        '2025-12-12': {'avg': -0.20, 'count': 12},
        '2025-12-13': {'avg': -0.25, 'count': 9},
        '2025-12-14': {'avg': -0.30, 'count': 11},
    }
    
    print(f"Testing with {len(test_daily)} days of data")
    print("Simulating sentiment declining from +0.20 to -0.30\n")
    
    npd_data, error = calculate_npd(test_daily)
    
    if error:
        print(f"Error: {error}")
        return
    
    print(f"NPD Score: {npd_data['npd_score']}")
    print(f"Regime: {npd_data['regime']}")
    print(f"Short Avg (3D): {npd_data['short_avg']}")
    print(f"Long Avg (7D): {npd_data['long_avg']}")
    print(f"\nInterpretation: {get_npd_interpretation(npd_data['regime'])}")
    
    return npd_data


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'test':
            test_npd()
        
        elif command == 'run':
            run_daily_npd()
        
        else:
            print("Usage:")
            print("  python npd_narrative_polarity_drift.py test  - Run with sample data")
            print("  python npd_narrative_polarity_drift.py run   - Run daily NPD pipeline")
    
    else:
        # Default: run test
        test_npd()