"""
NARRATIVE COHERENCE INDEX (NCI)
Measures how clustered/similar the day's articles are
High coherence = everyone talking about same risks = late-stage fragility signal

Scale: 0-100
0-30: Low coherence (scattered topics - healthy)
30-60: Moderate coherence (some clustering)
60-80: High coherence (narratives converging - caution)
80-100: Extreme coherence (echo chamber - warning)

Location: argus1/nci_narrative_coherence.py
"""

import os
import re
import json
import random
from collections import Counter
from typing import List, Dict, Any, Optional
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

# Google Sheets config
GOOGLE_SHEETS_CREDS_FILE = 'credentials.json'  # Path to your service account JSON
SPREADSHEET_NAME = 'Copy of Copy of Version 3.3 - Development Copy (Active)'  # Your spreadsheet name
NCI_SHEET_NAME = 'NCI_Dial'  # Sheet to write NCI results

# Stop words to ignore in title analysis
STOP_WORDS = {
    'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can',
    'had', 'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been',
    'could', 'would', 'about', 'which', 'when', 'make', 'like', 'into',
    'just', 'over', 'such', 'than', 'them', 'then', 'these', 'will',
    'with', 'this', 'that', 'from', 'they', 'what', 'says', 'said',
    'how', 'why', 'new', 'news'
}

# Domain-specific keywords to track
IMPORTANT_TERMS = [
    # Infrastructure
    'data center', 'datacenter', 'gpu', 'chip', 'semiconductor', 'fab',
    'power', 'energy', 'grid', 'electricity', 'water', 'cooling',
    # Policy/Constraints
    'tariff', 'regulation', 'export', 'ban', 'restriction', 'control',
    'china', 'taiwan', 'rare earth', 'supply chain',
    # Bubble/Market
    'bubble', 'overvalued', 'crash', 'correction', 'selloff', 'decline',
    'capex', 'spending', 'investment', 'valuation',
    # Companies
    'nvidia', 'oracle', 'microsoft', 'google', 'amazon', 'meta',
    'tsmc', 'intel', 'amd', 'broadcom', 'asml'
]


# ============================================================
# DATABASE FUNCTIONS
# ============================================================

def get_supabase_client() -> Client:
    """Get Supabase client"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables required")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_todays_articles(supabase: Client) -> List[Dict[str, Any]]:
    """Fetch today's processed articles from Supabase"""
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Adjust table/column names to match your schema
    response = supabase.table('processed_articles') \
        .select('title, y2ai_category, processed_at') \
        .gte('processed_at', today_start.isoformat()) \
        .execute()
    
    return response.data if response.data else []


def fetch_articles_by_date(supabase: Client, date: datetime) -> List[Dict[str, Any]]:
    """Fetch articles for a specific date"""
    date_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
    date_end = date_start + timedelta(days=1)
    
    response = supabase.table('processed_articles') \
        .select('title, y2ai_category, processed_at') \
        .gte('processed_at', date_start.isoformat()) \
        .lt('processed_at', date_end.isoformat()) \
        .execute()
    
    return response.data if response.data else []


def save_nci_to_supabase(supabase: Client, nci_result: Dict[str, Any]):
    """Save NCI result to Supabase daily_signals table"""
    today = datetime.utcnow().strftime('%Y-%m-%d')
    
    record = {
        'nci_score': nci_result['score'],
        'nci_regime': nci_result['regime'],
        'nci_top_category': nci_result['topClusters'][0]['category'] if nci_result.get('topClusters') else None,
        'nci_top_keyword': nci_result['topKeywords'][0]['keyword'] if nci_result.get('topKeywords') else None,
        'updated_at': datetime.utcnow().isoformat()
    }
    
    try:
        # Update existing row for today
        response = supabase.table('daily_signals') \
            .update(record) \
            .eq('date', today) \
            .execute()
        
        if response.data:
            print(f"Saved NCI to Supabase: {nci_result['score']} ({nci_result['regime']})")
        else:
            print(f"No row found for {today} - NCI not saved to Supabase")
        
        return response
    except Exception as e:
        print(f"Error saving to Supabase: {e}")
        return None

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


def ensure_nci_sheet_exists(client) -> gspread.Worksheet:
    """Create NCI_Dial sheet if it doesn't exist"""
    spreadsheet = client.open(SPREADSHEET_NAME)
    
    try:
        sheet = spreadsheet.worksheet(NCI_SHEET_NAME)
    except gspread.WorksheetNotFound:
        # Create the sheet
        sheet = spreadsheet.add_worksheet(title=NCI_SHEET_NAME, rows=1000, cols=10)
        
        # Add headers
        headers = [
            'Date',
            'NCI Score',
            'Regime',
            'Top Category',
            'Top Keyword',
            'Article Count',
            'Cat Concentration',
            'Keyword Coherence',
            'Title Similarity',
            'Interpretation'
        ]
        sheet.append_row(headers)
        
        # Format header row
        sheet.format('A1:J1', {
            'textFormat': {'bold': True},
            'backgroundColor': {'red': 0.1, 'green': 0.1, 'blue': 0.2}
        })
        
        print(f"Created {NCI_SHEET_NAME} sheet with headers")
    
    return sheet

def write_nci_to_sheets(nci_result: Dict[str, Any]):
    """Write NCI result to Google Sheets"""
    try:
        print("Getting sheets client...")
        client = get_sheets_client()
        
        print(f"Opening spreadsheet: {SPREADSHEET_NAME}")
        spreadsheet = client.open(SPREADSHEET_NAME)
        
        print("Looking for NCI_Dial sheet...")
        try:
            sheet = spreadsheet.worksheet(NCI_SHEET_NAME)
            print("Found existing NCI_Dial sheet")
        except gspread.WorksheetNotFound:
            print("Creating NCI_Dial sheet...")
            sheet = spreadsheet.add_worksheet(title=NCI_SHEET_NAME, rows=1000, cols=10)
            headers = ['Date', 'NCI Score', 'Regime', 'Top Category', 'Top Keyword', 
                      'Article Count', 'Cat Concentration', 'Keyword Coherence', 
                      'Title Similarity', 'Interpretation']
            sheet.append_row(headers)
            print("Created NCI_Dial with headers")
        
        # Prepare row data
        breakdown = nci_result.get('breakdown', {})
        row = [
            datetime.now().strftime('%Y-%m-%d'),
            nci_result['score'],
            nci_result['regime'],
            nci_result['topClusters'][0]['category'] if nci_result.get('topClusters') else '',
            nci_result['topKeywords'][0]['keyword'] if nci_result.get('topKeywords') else '',
            nci_result['articleCount'],
            round(breakdown.get('categoryConcentration', {}).get('score', 0), 1),
            round(breakdown.get('keywordCoherence', {}).get('score', 0), 1),
            round(breakdown.get('titleSimilarity', {}).get('score', 0), 1),
            get_nci_interpretation(nci_result['regime'])
        ]
        
        print("Appending row...")
        sheet.append_row(row)
        print(f"Wrote NCI to sheets: {nci_result['score']} ({nci_result['regime']})")
        return True
        
    except Exception as e:
        print(f"Error writing to sheets: {type(e).__name__}: {e}")
        return False

# ============================================================
# NCI CALCULATION FUNCTIONS
# ============================================================

def calculate_nci(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Main function to calculate NCI
    
    Args:
        articles: List of article dicts with 'title', 'y2ai_category', etc.
    
    Returns:
        Dict with score, regime, breakdown, topClusters
    """
    if not articles or len(articles) < 3:
        return {
            'score': 0,
            'regime': 'INSUFFICIENT_DATA',
            'breakdown': {},
            'topClusters': [],
            'topKeywords': [],
            'articleCount': len(articles) if articles else 0
        }
    
    # Component 1: Category concentration (40% weight)
    category_result = calculate_category_concentration(articles)
    
    # Component 2: Keyword coherence (30% weight)
    keyword_result = calculate_keyword_coherence(articles)
    
    # Component 3: Title similarity (30% weight)
    title_result = calculate_title_coherence(articles)
    
    # Weighted composite
    nci_score = (
        category_result['score'] * 0.4 +
        keyword_result['score'] * 0.3 +
        title_result['score'] * 0.3
    )
    
    # Determine regime
    if nci_score >= 80:
        regime = 'EXTREME'
    elif nci_score >= 60:
        regime = 'HIGH'
    elif nci_score >= 30:
        regime = 'MODERATE'
    else:
        regime = 'LOW'
    
    return {
        'score': round(nci_score, 1),
        'regime': regime,
        'breakdown': {
            'categoryConcentration': category_result,
            'keywordCoherence': keyword_result,
            'titleSimilarity': title_result
        },
        'topClusters': category_result['topCategories'],
        'topKeywords': keyword_result['topKeywords'],
        'articleCount': len(articles)
    }


def calculate_category_concentration(articles: List[Dict]) -> Dict[str, Any]:
    """
    Component 1: Category Concentration
    High score if articles cluster in few categories
    Uses Herfindahl-Hirschman Index (HHI)
    """
    category_counts = Counter()
    
    for article in articles:
        cat = article.get('y2ai_category') or article.get('category') or 'unknown'
        category_counts[cat] += 1
    
    total = len(articles)
    categories = list(category_counts.keys())
    
    # Calculate HHI (sum of squared market shares)
    hhi = sum((count / total) ** 2 for count in category_counts.values())
    
    # Normalize: HHI ranges from 1/n (equal) to 1 (single category)
    min_hhi = 1 / max(len(categories), 1)
    if hhi >= 1:
        normalized_hhi = 1
    else:
        normalized_hhi = (hhi - min_hhi) / (1 - min_hhi) if min_hhi < 1 else 0
    
    score = min(100, normalized_hhi * 100)
    
    # Top categories
    top_categories = [
        {
            'category': cat,
            'count': count,
            'percentage': f"{(count / total) * 100:.1f}%"
        }
        for cat, count in category_counts.most_common(3)
    ]
    
    return {
        'score': score,
        'hhi': round(hhi, 3),
        'categoryCount': len(categories),
        'topCategories': top_categories
    }


def calculate_keyword_coherence(articles: List[Dict]) -> Dict[str, Any]:
    """
    Component 2: Keyword Coherence
    High score if same keywords appear across multiple articles
    """
    all_keywords = []
    
    for article in articles:
        title = (article.get('title') or '').lower()
        
        for term in IMPORTANT_TERMS:
            if term in title:
                all_keywords.append(term)
    
    if not all_keywords:
        return {'score': 0, 'topKeywords': [], 'avgKeywordsPerArticle': 0}
    
    keyword_counts = Counter(all_keywords)
    unique_keywords = list(keyword_counts.keys())
    
    # Score based on shared keywords
    shared_keyword_score = 0
    for kw, count in keyword_counts.items():
        if count >= 2:
            shared_keyword_score += (count / len(articles)) * 100
    
    # Normalize
    score = min(100, shared_keyword_score / max(len(unique_keywords), 1) * 2)
    
    # Top keywords
    top_keywords = [
        {
            'keyword': kw,
            'count': count,
            'percentage': f"{(count / len(articles)) * 100:.1f}%"
        }
        for kw, count in keyword_counts.most_common(5)
    ]
    
    return {
        'score': score,
        'topKeywords': top_keywords,
        'avgKeywordsPerArticle': round(len(all_keywords) / len(articles), 1)
    }


def calculate_title_coherence(articles: List[Dict]) -> Dict[str, Any]:
    """
    Component 3: Title Similarity
    High score if article titles are semantically similar
    """
    if len(articles) < 2:
        return {'score': 0, 'avgSimilarity': 0, 'pairsAnalyzed': 0}
    
    # Clean titles
    cleaned_titles = [clean_title(a.get('title', '')) for a in articles]
    
    # Calculate pairwise similarities
    max_pairs = 100
    n = len(cleaned_titles)
    total_pairs = (n * (n - 1)) // 2
    
    similarities = []
    pairs_analyzed = 0
    
    if total_pairs <= max_pairs:
        # Do all pairs
        for i in range(n):
            for j in range(i + 1, n):
                sim = jaccard_similarity(cleaned_titles[i], cleaned_titles[j])
                similarities.append(sim)
                pairs_analyzed += 1
    else:
        # Random sampling
        seen_pairs = set()
        while pairs_analyzed < max_pairs:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i != j and (i, j) not in seen_pairs and (j, i) not in seen_pairs:
                sim = jaccard_similarity(cleaned_titles[i], cleaned_titles[j])
                similarities.append(sim)
                seen_pairs.add((i, j))
                pairs_analyzed += 1
    
    # Average similarity
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0
    
    # Scale to 0-100 (typical similarities are 0-0.5)
    score = min(100, avg_similarity * 200)
    
    return {
        'score': score,
        'avgSimilarity': round(avg_similarity, 3),
        'pairsAnalyzed': pairs_analyzed
    }


def clean_title(title: str) -> List[str]:
    """Clean title and return list of words"""
    # Remove HTML tags
    title = re.sub(r'<[^>]+>', '', title)
    # Remove special characters
    title = re.sub(r'[^\w\s]', ' ', title)
    # Normalize whitespace and lowercase
    words = title.lower().split()
    # Filter stop words and short words
    return [w for w in words if len(w) > 2 and w not in STOP_WORDS]


def jaccard_similarity(words1: List[str], words2: List[str]) -> float:
    """Calculate Jaccard similarity between two word lists"""
    if not words1 or not words2:
        return 0
    
    set1 = set(words1)
    set2 = set(words2)
    intersection = set1 & set2
    union = set1 | set2
    
    return len(intersection) / len(union) if union else 0


def get_nci_interpretation(regime: str) -> str:
    """Get human-readable interpretation"""
    interpretations = {
        'EXTREME': 'Echo chamber - virtually all coverage on same themes. Often precedes volatility.',
        'HIGH': 'High clustering around shared themes. Markets focused on same risks.',
        'MODERATE': 'Some thematic clustering but diverse coverage. Normal conditions.',
        'LOW': 'Scattered coverage across many topics. No dominant narrative.',
        'INSUFFICIENT_DATA': 'Not enough articles to calculate.'
    }
    return interpretations.get(regime, 'Unable to assess.')


# ============================================================
# MAIN PIPELINE FUNCTIONS
# ============================================================

def run_daily_nci(write_to_sheets: bool = True, write_to_supabase: bool = True) -> Dict[str, Any]:
    """
    Main daily pipeline function
    Call this after article processing completes
    """
    print("=== NCI DAILY RUN ===")
    print(f"Time: {datetime.now().isoformat()}")
    
    # Get Supabase client
    supabase = get_supabase_client()
    
    # Fetch today's articles
    articles = fetch_todays_articles(supabase)
    print(f"Fetched {len(articles)} articles")
    
    if len(articles) < 3:
        print("Insufficient articles for NCI calculation")
        return {'score': 0, 'regime': 'INSUFFICIENT_DATA', 'articleCount': len(articles)}
    
    # Calculate NCI
    nci_result = calculate_nci(articles)
    
    print(f"\nNCI Score: {nci_result['score']}")
    print(f"Regime: {nci_result['regime']}")
    print(f"Interpretation: {get_nci_interpretation(nci_result['regime'])}")
    
    if nci_result.get('topClusters'):
        print(f"Top Category: {nci_result['topClusters'][0]['category']}")
    if nci_result.get('topKeywords'):
        print(f"Top Keyword: {nci_result['topKeywords'][0]['keyword']}")
    
    # Write to Google Sheets
    if write_to_sheets:
        write_nci_to_sheets(nci_result)
    
    # Write to Supabase
    if write_to_supabase:
        save_nci_to_supabase(supabase, nci_result)
    
    print("\n=== NCI COMPLETE ===")
    return nci_result


def run_nci_for_date(date_str: str, write_to_sheets: bool = False) -> Dict[str, Any]:
    """
    Run NCI for a specific date (for backfilling)
    
    Args:
        date_str: Date in 'YYYY-MM-DD' format
    """
    date = datetime.strptime(date_str, '%Y-%m-%d')
    
    print(f"=== NCI FOR {date_str} ===")
    
    supabase = get_supabase_client()
    articles = fetch_articles_by_date(supabase, date)
    
    print(f"Fetched {len(articles)} articles")
    
    nci_result = calculate_nci(articles)
    
    print(f"NCI Score: {nci_result['score']} ({nci_result['regime']})")
    
    if write_to_sheets:
        write_nci_to_sheets(nci_result)
    
    return nci_result


# ============================================================
# TEST FUNCTION
# ============================================================

def test_nci():
    """Test with sample articles"""
    test_articles = [
        {'title': 'AI Data Center Boom Sparks Fears of Glut Amid Lending Frenzy', 'y2ai_category': 'skepticism'},
        {'title': 'Wall Street ends lower; fears of AI bubble and inflation', 'y2ai_category': 'skepticism'},
        {'title': "Oracle's $300B AI Bet Becomes Bubble Barometer", 'y2ai_category': 'skepticism'},
        {'title': 'China Weighs Up to $70 Billion Chip Push in Tech Fight', 'y2ai_category': 'china'},
        {'title': 'China Unleashes $70 Billion Semiconductor Gambit', 'y2ai_category': 'china'},
        {'title': 'Huawei-SMIC chip hits milestone, still lags TSMC', 'y2ai_category': 'china'},
        {'title': 'Environmental Groups Urge Congress to Pause Data Center Growth', 'y2ai_category': 'energy'},
        {'title': 'Michigan AG challenges energy rules amid data center boom', 'y2ai_category': 'energy'},
        {'title': 'DeKalb County considering regulations on data centers', 'y2ai_category': 'energy'},
        {'title': 'Thumbs down: Chandler kills data center in 7-0 vote', 'y2ai_category': 'constraints'},
        {'title': "Germany's Foreign Minister crawls to Beijing for rare earths", 'y2ai_category': 'constraints'},
        {'title': 'Energy Dominance vs Mineral Reality: Rare Earth Wall', 'y2ai_category': 'constraints'}
    ]
    
    print("=== NCI TEST ===")
    print(f"Testing with {len(test_articles)} articles\n")
    
    result = calculate_nci(test_articles)
    
    print(f"NCI Score: {result['score']}")
    print(f"Regime: {result['regime']}")
    print(f"\nInterpretation: {get_nci_interpretation(result['regime'])}")
    
    print(f"\n=== BREAKDOWN ===")
    print(f"Category Concentration: {result['breakdown']['categoryConcentration']['score']:.1f}")
    print(f"  Top categories: {result['topClusters']}")
    
    print(f"\nKeyword Coherence: {result['breakdown']['keywordCoherence']['score']:.1f}")
    print(f"  Top keywords: {result['topKeywords']}")
    
    print(f"\nTitle Similarity: {result['breakdown']['titleSimilarity']['score']:.1f}")
    print(f"  Avg similarity: {result['breakdown']['titleSimilarity']['avgSimilarity']}")
    
    return result


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'test':
            test_nci()
        
        elif command == 'run':
            # Run daily NCI
            run_daily_nci()
        
        elif command == 'date' and len(sys.argv) > 2:
            # Run for specific date: python nci_narrative_coherence.py date 2025-12-12
            run_nci_for_date(sys.argv[2], write_to_sheets=True)
        
        else:
            print("Usage:")
            print("  python nci_narrative_coherence.py test     - Run with sample data")
            print("  python nci_narrative_coherence.py run      - Run daily NCI pipeline")
            print("  python nci_narrative_coherence.py date YYYY-MM-DD  - Run for specific date")
    
    else:
        # Default: run test
        test_nci()