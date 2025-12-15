"""
evi_event_volatility_index.py
Calculates Event Volatility Index from processed article flow

EVI measures the intensity and urgency of news flow.
High EVI = lots of breaking news, rapid updates, crisis-level coverage
Low EVI = steady-state news, normal coverage patterns

Regimes:
  CRISIS:     EVI > 80  (extreme event activity)
  ELEVATED:   EVI 50-80 (above-normal event flow)
  NORMAL:     EVI 20-50 (typical news environment)
  QUIET:      EVI < 20  (unusually low activity)

Location: argus1/evi_event_volatility_index.py
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

# Google Sheets config
GOOGLE_SHEETS_CREDS_FILE = 'credentials.json'
SPREADSHEET_NAME = 'Copy of Copy of Version 3.3 - Development Copy (Active)'
EVI_SHEET_NAME = 'EVI_Dial'

# EVI calculation parameters
EVI_CONFIG = {
    'baseline_days': 14,        # Days to establish baseline
    'current_hours': 24,        # Hours to measure current activity
    'weight_volume': 0.40,      # Article volume component
    'weight_velocity': 0.30,    # Publishing velocity component
    'weight_urgency': 0.30,     # Urgency keyword component
}

# Urgency keywords (indicate breaking/crisis news)
URGENCY_KEYWORDS = [
    'breaking', 'urgent', 'flash', 'alert', 'crisis',
    'crash', 'plunge', 'surge', 'spike', 'collapse',
    'emergency', 'halt', 'suspend', 'investigate',
    'shocking', 'unprecedented', 'historic'
]


# ============================================================
# DATABASE FUNCTIONS
# ============================================================

def get_supabase_client() -> Client:
    """Get Supabase client"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables required")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def get_article_history(supabase: Client, days: int = 14):
    """Pull article metadata for EVI analysis."""
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    
    response = supabase.table('processed_articles') \
        .select('processed_at, title, one_line_summary, extracted_facts, y2ai_category') \
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


def write_evi_to_sheets(evi_data: dict):
    """Write EVI result to Google Sheets EVI_Dial tab."""
    try:
        print("Getting sheets client...")
        client = get_sheets_client()
        
        print(f"Opening spreadsheet: {SPREADSHEET_NAME}")
        spreadsheet = client.open(SPREADSHEET_NAME)
        
        print("Looking for EVI_Dial sheet...")
        try:
            sheet = spreadsheet.worksheet(EVI_SHEET_NAME)
            print("Found existing EVI_Dial sheet")
        except gspread.WorksheetNotFound:
            print("Creating EVI_Dial sheet...")
            sheet = spreadsheet.add_worksheet(title=EVI_SHEET_NAME, rows=1000, cols=10)
            headers = [
                'Date', 'EVI Score', 'Regime', 'Volume Score', 'Velocity Score',
                'Urgency Score', 'Article Count', 'Interpretation'
            ]
            sheet.append_row(headers)
            
            # Format header row
            sheet.format('A1:H1', {
                'textFormat': {'bold': True},
                'backgroundColor': {'red': 0.1, 'green': 0.2, 'blue': 0.1}
            })
            print("Created EVI_Dial with headers")
        
        # Prepare row
        row = [
            datetime.utcnow().strftime('%Y-%m-%d'),
            evi_data['evi_score'],
            evi_data['regime'],
            evi_data['volume_score'],
            evi_data['velocity_score'],
            evi_data['urgency_score'],
            evi_data['article_count'],
            evi_data['interpretation']
        ]
        
        print("Appending row...")
        sheet.append_row(row)
        print(f"Wrote EVI to sheets: {evi_data['evi_score']} ({evi_data['regime']})")
        return True
        
    except Exception as e:
        print(f"Error writing to sheets: {type(e).__name__}: {e}")
        return False


# ============================================================
# EVI CALCULATION FUNCTIONS
# ============================================================

def count_urgency_keywords(article: dict) -> int:
    """Count urgency keywords in article."""
    # Build text from available fields
    text_parts = [article.get('title', '')]
    
    if article.get('one_line_summary'):
        text_parts.append(article.get('one_line_summary'))
    
    # Extract from extracted_facts if available
    facts = article.get('extracted_facts')
    if facts and isinstance(facts, dict):
        for key, value in facts.items():
            if isinstance(value, str):
                text_parts.append(value)
    
    text = ' '.join(text_parts).lower()
    
    count = 0
    for keyword in URGENCY_KEYWORDS:
        if keyword.lower() in text:
            count += 1
    return count


def calculate_hourly_distribution(articles: list, hours: int = 24) -> list:
    """Calculate article count per hour for velocity measurement."""
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    
    hourly_counts = [0] * hours
    
    for article in articles:
        processed_at = article.get('processed_at', '')
        if not processed_at:
            continue
        
        try:
            article_date = datetime.fromisoformat(processed_at.replace('Z', '+00:00')).replace(tzinfo=None)
        except:
            continue
        
        if article_date >= cutoff:
            hours_ago = int((datetime.utcnow() - article_date).total_seconds() / 3600)
            if 0 <= hours_ago < hours:
                hourly_counts[hours_ago] += 1
    
    return hourly_counts


def calculate_evi(articles: list) -> tuple:
    """
    Calculate Event Volatility Index.
    
    Components:
    1. Volume Score: Current article count vs baseline
    2. Velocity Score: Burstiness of article timing
    3. Urgency Score: Presence of crisis/breaking keywords
    
    Returns:
        tuple: (evi_data dict, error string or None)
    """
    if not articles:
        return None, "No articles to analyze"
    
    # Split articles into baseline and current
    current_cutoff = datetime.utcnow() - timedelta(hours=EVI_CONFIG['current_hours'])
    baseline_cutoff = datetime.utcnow() - timedelta(days=EVI_CONFIG['baseline_days'])
    
    current_articles = []
    baseline_articles = []
    
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
    
    # =========================================
    # 1. VOLUME SCORE (0-100)
    # =========================================
    baseline_daily_avg = len(baseline_articles) / EVI_CONFIG['baseline_days']
    current_daily_equiv = len(current_articles) * (24 / EVI_CONFIG['current_hours'])
    
    if baseline_daily_avg > 0:
        volume_ratio = current_daily_equiv / baseline_daily_avg
    else:
        volume_ratio = 1.0
    
    # Scale: 0.5x = 0, 1x = 50, 2x = 100
    volume_score = min(100, max(0, (volume_ratio - 0.5) * 100 / 1.5))
    
    # =========================================
    # 2. VELOCITY SCORE (0-100)
    # =========================================
    hourly_counts = calculate_hourly_distribution(current_articles, hours=24)
    
    if sum(hourly_counts) > 0:
        # Calculate coefficient of variation (higher = burstier)
        mean_hourly = sum(hourly_counts) / len(hourly_counts)
        variance = sum((x - mean_hourly) ** 2 for x in hourly_counts) / len(hourly_counts)
        std_dev = variance ** 0.5
        
        if mean_hourly > 0:
            cv = std_dev / mean_hourly
        else:
            cv = 0
        
        # Scale CV to 0-100 (CV of 2 = very bursty = 100)
        velocity_score = min(100, cv * 50)
    else:
        velocity_score = 0
    
    # =========================================
    # 3. URGENCY SCORE (0-100)
    # =========================================
    total_urgency = sum(count_urgency_keywords(a) for a in current_articles)
    articles_with_urgency = sum(1 for a in current_articles if count_urgency_keywords(a) > 0)
    
    if len(current_articles) > 0:
        urgency_density = articles_with_urgency / len(current_articles)
    else:
        urgency_density = 0
    
    # Scale: 0% = 0, 20% = 50, 40%+ = 100
    urgency_score = min(100, urgency_density * 250)
    
    # =========================================
    # COMPOSITE EVI
    # =========================================
    evi_score = (
        volume_score * EVI_CONFIG['weight_volume'] +
        velocity_score * EVI_CONFIG['weight_velocity'] +
        urgency_score * EVI_CONFIG['weight_urgency']
    )
    
    evi_score = round(evi_score, 2)
    
    # Determine regime
    if evi_score >= 80:
        regime = "CRISIS"
    elif evi_score >= 50:
        regime = "ELEVATED"
    elif evi_score >= 20:
        regime = "NORMAL"
    else:
        regime = "QUIET"
    
    # Get interpretation
    interpretation = get_evi_interpretation(regime, volume_ratio, urgency_density)
    
    return {
        'date': datetime.utcnow().strftime('%Y-%m-%d'),
        'evi_score': evi_score,
        'regime': regime,
        'volume_score': round(volume_score, 1),
        'velocity_score': round(velocity_score, 1),
        'urgency_score': round(urgency_score, 1),
        'volume_ratio': round(volume_ratio, 2),
        'urgency_density': round(urgency_density, 3),
        'article_count': len(current_articles),
        'interpretation': interpretation
    }, None


def get_evi_interpretation(regime: str, volume_ratio: float, urgency_density: float) -> str:
    """Get human-readable interpretation"""
    if regime == "CRISIS":
        return f"Crisis-level news flow. Volume {volume_ratio:.1f}x normal, {urgency_density*100:.0f}% urgent. Expect volatility."
    elif regime == "ELEVATED":
        return f"Elevated event activity. Volume {volume_ratio:.1f}x baseline. Monitor for escalation."
    elif regime == "NORMAL":
        return "Normal news environment. No unusual event activity detected."
    else:
        return "Quiet news period. Below-average coverage. Watch for surprise catalysts."


# ============================================================
# SUPABASE OUTPUT
# ============================================================

def save_evi_to_supabase(supabase: Client, evi_data: dict):
    """Save EVI to Supabase daily_signals table."""
    today = datetime.utcnow().strftime('%Y-%m-%d')
    
    record = {
        'date': today,
        'evi_score': evi_data['evi_score'],
        'evi_regime': evi_data['regime'],
        'evi_volume': evi_data['volume_score'],
        'evi_velocity': evi_data['velocity_score'],
        'evi_urgency': evi_data['urgency_score'],
        'updated_at': datetime.utcnow().isoformat()
    }
    
    try:
        response = supabase.table('daily_signals') \
            .upsert(record, on_conflict='date') \
            .execute()
        
        print(f"Saved EVI to Supabase: {evi_data['evi_score']} ({evi_data['regime']})")
        return response
    except Exception as e:
        print(f"Error saving to Supabase: {e}")
        return None


# ============================================================
# MAIN PIPELINE FUNCTIONS
# ============================================================

def run_daily_evi(write_to_sheets: bool = True, write_to_supabase: bool = True) -> dict:
    """
    Main daily pipeline function.
    Call this after article processing completes.
    """
    print("=" * 50)
    print("EVENT VOLATILITY INDEX (EVI) CALCULATION")
    print(f"Time: {datetime.utcnow().isoformat()}")
    print("=" * 50)
    
    # Get Supabase client
    supabase = get_supabase_client()
    
    # Fetch article history
    print("\n1. Fetching article history...")
    articles = get_article_history(supabase, days=EVI_CONFIG['baseline_days'])
    print(f"   Retrieved {len(articles)} articles")
    
    if not articles:
        print("   ERROR: No articles found")
        return {'evi_score': 0, 'regime': 'INSUFFICIENT_DATA', 'article_count': 0}
    
    # Calculate EVI
    print("\n2. Calculating EVI components...")
    evi_data, error = calculate_evi(articles)
    
    if error:
        print(f"   ERROR: {error}")
        return {'evi_score': 0, 'regime': 'INSUFFICIENT_DATA', 'article_count': 0}
    
    print(f"   EVI Score: {evi_data['evi_score']}")
    print(f"   Regime: {evi_data['regime']}")
    print(f"   Volume Score: {evi_data['volume_score']} ({evi_data['volume_ratio']}x baseline)")
    print(f"   Velocity Score: {evi_data['velocity_score']}")
    print(f"   Urgency Score: {evi_data['urgency_score']} ({evi_data['urgency_density']*100:.1f}% urgent)")
    print(f"   Interpretation: {evi_data['interpretation']}")
    
    # Write to Google Sheets
    if write_to_sheets:
        print("\n3. Writing to Google Sheets...")
        write_evi_to_sheets(evi_data)
    
    # Save to Supabase
    if write_to_supabase:
        print("\n4. Saving to Supabase...")
        save_evi_to_supabase(supabase, evi_data)
    
    print("\n" + "=" * 50)
    print("EVI CALCULATION COMPLETE")
    print("=" * 50)
    
    return evi_data


# ============================================================
# TEST FUNCTION
# ============================================================

def test_evi():
    """Test EVI calculation with sample data"""
    print("=== EVI TEST ===")
    
    # Simulate articles - baseline (steady state)
    test_articles = []
    
    # Baseline: 10 articles per day for 13 days
    for day in range(13, 1, -1):
        for hour in range(0, 24, 2):  # Every 2 hours = 12 per day
            date = (datetime.utcnow() - timedelta(days=day, hours=hour)).isoformat()
            test_articles.append({
                'processed_at': date,
                'title': 'Regular AI industry update',
                'summary': 'Normal coverage of tech sector developments'
            })
    
    # Current 24h: Crisis scenario - 50 articles, many urgent
    for hour in range(24):
        date = (datetime.utcnow() - timedelta(hours=hour)).isoformat()
        # Normal articles
        test_articles.append({
            'processed_at': date,
            'title': 'Market analysis continues',
            'summary': 'Standard coverage'
        })
        # Urgent articles
        if hour % 2 == 0:
            test_articles.append({
                'processed_at': date,
                'title': 'BREAKING: Market crash fears emerge',
                'summary': 'Urgent alert as stocks plunge amid crisis'
            })
    
    print(f"Testing with {len(test_articles)} simulated articles")
    print("Simulating crisis scenario with elevated volume and urgency\n")
    
    evi_data, error = calculate_evi(test_articles)
    
    if error:
        print(f"Error: {error}")
        return
    
    print(f"EVI Score: {evi_data['evi_score']}")
    print(f"Regime: {evi_data['regime']}")
    print(f"Volume Score: {evi_data['volume_score']}")
    print(f"Velocity Score: {evi_data['velocity_score']}")
    print(f"Urgency Score: {evi_data['urgency_score']}")
    print(f"\nInterpretation: {evi_data['interpretation']}")
    
    return evi_data


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'test':
            test_evi()
        
        elif command == 'run':
            run_daily_evi()
        
        else:
            print("Usage:")
            print("  python evi_event_volatility_index.py test  - Run with sample data")
            print("  python evi_event_volatility_index.py run   - Run daily EVI pipeline")
    
    else:
        # Default: run test
        test_evi()