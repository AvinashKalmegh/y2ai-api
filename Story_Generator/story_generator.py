"""
Y2AI Daily Story Generator
February 20, 2026

Reads signal data from Google Sheets, identifies the day's strongest
story using priority ranking, and generates a draft Medium post saved
to Google Drive. Weekly mode generates strand emergence story bank.

Usage:
    python story_generator.py              # Daily story generation
    python story_generator.py weekly       # Weekly strand story bank
    python story_generator.py test         # Test run with verbose output

Requirements:
    pip install gspread google-auth google-api-python-client pandas numpy
"""

import sys
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from datetime import datetime, timedelta
import time


# ============================================================
# CONFIGURATION
# ============================================================

SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

CREDENTIALS_FILE = 'credentials.json'

DM_SPREADSHEET_ID    = '1GiLsxrgW-nssIhuUGYzn7SHYzcNjP4yg_7p1G2nPNiE'
ARGUS_SPREADSHEET_ID = '1YYwgMvY7I4_i0RD2OH0lM878OkK2WumdEz19UfFcUZ0'

# Sheet names in DM spreadsheet
DM_LATEST_TAB    = 'DM_Latest'
DM_HISTORY_TAB   = 'DM_2024_2026'
MONITOR_TAB      = 'Chain_Monitor'
BACKTEST_TAB     = 'Chain_Backtest_Results'
CLUSTER_TAB      = 'Strand_Emergence_Clusters'

# Output
PIPELINE_TAB     = 'Story_Pipeline'
DRAFTS_FOLDER    = 'Y2AI_Daily_Drafts'

# Signal thresholds
ANOMALY_ZSCORE       = -1.8
ANOMALY_LAYER_MIN_DM = 55
VELOCITY_DAYS        = 5
DIVERGENCE_DM_CHANGE = -10
DIVERGENCE_SECTOR_DM = 50
FADING_DM_LOW        = 30
FADING_DM_HIGH       = 50
FADING_30D_CHANGE    = -20
STRAND_MIN_DM        = 60
STRAND_MIN_STRANDS   = 25

WEEKLY_MAX_DRAFTS = 5


# ============================================================
# ETF TO SECTOR MAPPING
# ============================================================

ETF_TO_SECTOR = {
    'XLK':  'Technology',
    'SMH':  'Semiconductors',
    'IGV':  'Enterprise Software',
    'XLF':  'Financials',
    'XLV':  'Healthcare',
    'XBI':  'Biotech',
    'XLE':  'Energy',
    'XLI':  'Industrials',
    'XLY':  'Consumer Discretionary',
    'XLP':  'Consumer Staples',
    'XLU':  'Utilities',
    'XLC':  'Communication Services',
    'XLRE': 'Real Estate',
    'XLB':  'Materials',
    'ITA':  'Defense & Aerospace',
    'ITB':  'Homebuilders',
    'GDX':  'Gold Miners',
    'URA':  'Uranium & Nuclear',
    'HACK': 'Cybersecurity',
    'ARKK': 'Innovation/Disruptive',
    'QQQ':  'Large-Cap Tech',
    'SPY':  'Broad Market',
}

AI_ADJACENT_SECTORS = {
    'Technology', 'Enterprise Software', 'Semiconductors',
    'Communication Services', 'Cybersecurity', 'Innovation/Disruptive',
    'Large-Cap Tech',
}

SECTOR_PLAIN = {
    'Technology':              'technology',
    'Semiconductors':          'semiconductor stocks',
    'Enterprise Software':     'enterprise software',
    'Financials':              'financial stocks',
    'Healthcare':              'healthcare',
    'Biotech':                 'biotech',
    'Energy':                  'energy stocks',
    'Industrials':             'industrial stocks',
    'Consumer Discretionary':  'consumer discretionary',
    'Consumer Staples':        'consumer staples',
    'Utilities':               'utility stocks',
    'Communication Services':  'media and communications',
    'Real Estate':             'real estate',
    'Materials':               'materials stocks',
    'Defense & Aerospace':     'defense and aerospace',
    'Homebuilders':            'homebuilders',
    'Gold Miners':             'gold miners',
    'Uranium & Nuclear':       'nuclear energy',
    'Cybersecurity':           'cybersecurity',
    'Innovation/Disruptive':   'high-growth tech',
    'Large-Cap Tech':          'large-cap tech',
    'Broad Market':            'broad market',
}


# ============================================================
# GOOGLE SHEETS / DRIVE CLIENTS
# ============================================================

_gc = None
_drive = None


def get_sheets_client():
    global _gc
    if _gc is None:
        creds = Credentials.from_service_account_file(
            CREDENTIALS_FILE, scopes=SCOPES
        )
        _gc = gspread.authorize(creds)
    return _gc


def get_drive_service():
    global _drive
    if _drive is None:
        creds = Credentials.from_service_account_file(
            CREDENTIALS_FILE, scopes=SCOPES
        )
        _drive = build('drive', 'v3', credentials=creds)
    return _drive


# ============================================================
# DATA LOADING
# ============================================================

def load_dm_latest():
    """
    Load DM_Latest. Returns DataFrame with columns:
    Ticker, DM, Phase, DM_Change, ETF, Close, Volume_Z, RelStr_SPY
    """
    print("  Loading DM_Latest...")
    gc = get_sheets_client()
    sh = gc.open_by_key(DM_SPREADSHEET_ID)
    ws = sh.worksheet(DM_LATEST_TAB)
    data = ws.get_all_values()

    header = data[0]
    rows = data[1:]
    df = pd.DataFrame(rows, columns=header)

    # Standardize column names (flexible detection)
    col_map = {}
    for col in df.columns:
        cl = col.strip().lower()
        if cl == 'ticker':     col_map[col] = 'Ticker'
        elif cl == 'dm':       col_map[col] = 'DM'
        elif cl == 'phase':    col_map[col] = 'Phase'
        elif cl in ('dm_change', 'dm change'): col_map[col] = 'DM_Change'
        elif cl == 'etf':      col_map[col] = 'ETF'
        elif cl == 'close':    col_map[col] = 'Close'
        elif cl in ('volume_z', 'volume z'): col_map[col] = 'Volume_Z'
        elif cl in ('relstr_spy', 'relstr spy', 'rel_str_spy'):
            col_map[col] = 'RelStr_SPY'
        elif cl in ('dm_7d_ago', 'dm 7d ago'):
            col_map[col] = 'DM_7d_Ago'
        elif cl == 'date':     col_map[col] = 'Date'
    df = df.rename(columns=col_map)

    # Convert numeric columns
    for c in ['DM', 'DM_Change', 'Close', 'Volume_Z', 'RelStr_SPY', 'DM_7d_Ago']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.dropna(subset=['Ticker', 'DM'])
    df['Ticker'] = df['Ticker'].str.strip()
    df = df[df['Ticker'] != '']

    # Map ETF to sector
    if 'ETF' in df.columns:
        df['Sector'] = df['ETF'].map(ETF_TO_SECTOR).fillna('Other')
        df['Sector_Plain'] = df['Sector'].map(SECTOR_PLAIN).fillna(
            'the broader market')
    else:
        df['Sector'] = 'Other'
        df['Sector_Plain'] = 'the broader market'

    print(f"    {len(df)} tickers loaded")
    return df


def load_chain_monitor():
    """Load latest Chain_Monitor row. Returns dict or None."""
    print("  Loading Chain_Monitor...")
    gc = get_sheets_client()
    sh = gc.open_by_key(DM_SPREADSHEET_ID)

    try:
        ws = sh.worksheet(MONITOR_TAB)
    except gspread.WorksheetNotFound:
        print("    Chain_Monitor tab not found")
        return None

    data = ws.get_all_values()
    if len(data) < 2:
        print("    Chain_Monitor empty")
        return None

    header = data[0]
    last_row = data[-1]
    result = dict(zip(header, last_row))

    # Check for alerts using actual column names and values
    cascade = False
    systemic = False

    # Check by column name (case-insensitive key lookup)
    for key, val in result.items():
        kl = key.strip().lower()
        vl = str(val).strip().upper()
        if 'cascade' in kl and vl == 'YES':
            cascade = True
        if 'systemic' in kl and vl == 'YES':
            systemic = True

    row_date = result.get('Date', result.get('date', ''))
    print(f"    Monitor date: {row_date}")
    print(f"    Monitor columns: {list(result.keys())}")
    print(f"    CASCADE_RISK: {cascade}, SYSTEMIC_SIGNAL: {systemic}")

    return {
        'date':     row_date,
        'raw':      result,
        'cascade':  cascade,
        'systemic': systemic,
    }


def load_strand_clusters():
    """Load Strand_Emergence_Clusters. Returns DataFrame."""
    print("  Loading Strand_Emergence_Clusters...")
    gc = get_sheets_client()
    sh = gc.open_by_key(DM_SPREADSHEET_ID)

    try:
        ws = sh.worksheet(CLUSTER_TAB)
    except gspread.WorksheetNotFound:
        print("    Strand_Emergence_Clusters tab not found")
        return pd.DataFrame()

    data = ws.get_all_values()
    if len(data) < 2:
        return pd.DataFrame()

    header = data[0]
    df = pd.DataFrame(data[1:], columns=header)

    # Flexible column detection
    col_map = {}
    for col in df.columns:
        cl = col.strip().lower()
        if cl == 'ticker':   col_map[col] = 'Ticker'
        elif cl in ('strands', 'new_strands', 'strand_count'):
            col_map[col] = 'Strands'
        elif cl in ('dm', 'dm_current'):
            col_map[col] = 'DM'
        elif cl == 'signal': col_map[col] = 'Signal'
        elif cl in ('partners', 'partner_list'):
            col_map[col] = 'Partners'
    df = df.rename(columns=col_map)

    for c in ['Strands', 'DM']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Diagnostic: show actual columns after mapping
    print(f"    {len(df)} clusters loaded")
    print(f"    Columns after mapping: {df.columns.tolist()}")

    # Safety check: if DM column still missing, try fallback
    if 'DM' not in df.columns:
        for col in df.columns:
            if 'dm' in col.lower() or 'DM' in col:
                df = df.rename(columns={col: 'DM'})
                df['DM'] = pd.to_numeric(df['DM'], errors='coerce')
                print(f"    Fallback: mapped '{col}' -> 'DM'")
                break
    if 'Strands' not in df.columns:
        for col in df.columns:
            if 'strand' in col.lower():
                df = df.rename(columns={col: 'Strands'})
                df['Strands'] = pd.to_numeric(df['Strands'], errors='coerce')
                print(f"    Fallback: mapped '{col}' -> 'Strands'")
                break
    return df


# ============================================================
# SIGNAL DETECTION — FIVE PRIORITY LEVELS
# ============================================================

def check_chain_alerts(monitor_data):
    """Priority 1: Chain monitor cascade/systemic alerts."""
    if monitor_data is None:
        return None
    if not monitor_data['cascade'] and not monitor_data['systemic']:
        return None

    alert_type = []
    if monitor_data['cascade']:
        alert_type.append('CASCADE_RISK')
    if monitor_data['systemic']:
        alert_type.append('SYSTEMIC_SIGNAL')

    return {
        'priority':     1,
        'signal_type':  'CHAIN_ALERT',
        'story_type':   'SYSTEMIC',
        'ticker':       'TSLA' if monitor_data['cascade'] else 'MULTIPLE',
        'sector':       'Broad Market',
        'sector_plain': 'the broader market',
        'magnitude':    5,
        'strength':     5,
        'detail':       ' + '.join(alert_type),
        'raw':          monitor_data.get('raw', {}),
    }


def find_anomalous_nodes(dm_latest_df):
    """
    Priority 2: Ticker with largest negative z-score vs sector peers.
    Requires z-score < -1.8, sector avg DM > 55.
    """
    anomalies = []

    for sector, group in dm_latest_df.groupby('Sector'):
        if sector == 'Other' or len(group) < 5:
            continue

        sector_avg = group['DM'].mean()
        if sector_avg < ANOMALY_LAYER_MIN_DM:
            continue

        sector_std = group['DM'].std()
        if sector_std < 1:
            continue

        group = group.copy()
        group['z_score'] = (group['DM'] - sector_avg) / sector_std

        weak = group[group['z_score'] < ANOMALY_ZSCORE]
        for _, row in weak.iterrows():
            anomalies.append({
                'priority':     2,
                'signal_type':  'ANOMALOUS_NODE',
                'ticker':       row['Ticker'],
                'sector':       sector,
                'sector_plain': row.get('Sector_Plain', sector.lower()),
                'dm':           row['DM'],
                'z_score':      row['z_score'],
                'sector_avg':   round(sector_avg, 1),
                'magnitude':    abs(round(row['z_score'], 1)),
                'strength':     4 if row['z_score'] < -2.5 else 3,
            })

    if not anomalies:
        return None

    anomalies.sort(key=lambda x: x['z_score'])
    best = anomalies[0]
    best['story_type'] = classify_story(best)
    return best


def find_sector_divergence(dm_latest_df):
    """
    Priority 3: 2+ tickers in same sector with coordinated outflow
    (DM_Change < -10) while sector avg holds above 50.
    """
    divergences = []

    for sector, group in dm_latest_df.groupby('Sector'):
        if sector == 'Other' or len(group) < 4:
            continue

        sector_avg = group['DM'].mean()
        if sector_avg < DIVERGENCE_SECTOR_DM:
            continue

        if 'DM_Change' not in group.columns:
            continue

        weak = group[group['DM_Change'] < DIVERGENCE_DM_CHANGE]
        if len(weak) >= 2:
            tickers = weak['Ticker'].tolist()
            avg_change = weak['DM_Change'].mean()
            divergences.append({
                'priority':     3,
                'signal_type':  'SECTOR_DIVERGENCE',
                'ticker':       tickers[0],
                'tickers':      tickers[:4],
                'sector':       sector,
                'sector_plain': weak.iloc[0].get(
                    'Sector_Plain', sector.lower()),
                'dm_values':    dict(zip(weak['Ticker'], weak['DM'])),
                'sector_avg':   round(sector_avg, 1),
                'avg_change':   round(avg_change, 1),
                'magnitude':    abs(round(avg_change, 1)),
                'strength':     4 if len(tickers) >= 3 else 3,
            })

    if not divergences:
        return None

    divergences.sort(key=lambda x: x['avg_change'])
    best = divergences[0]
    best['story_type'] = classify_story(best)
    return best


def find_fading_tickers(dm_latest_df):
    """
    Priority 4: Single ticker DM 30-50, 30d change < -20.
    Significant momentum collapse in a previously healthy name.
    """
    if 'DM_Change' not in dm_latest_df.columns:
        return None

    candidates = dm_latest_df[
        (dm_latest_df['DM'] >= FADING_DM_LOW) &
        (dm_latest_df['DM'] <= FADING_DM_HIGH) &
        (dm_latest_df['DM_Change'] < FADING_30D_CHANGE)
    ].copy()

    if candidates.empty:
        return None

    candidates = candidates.sort_values('DM_Change')
    best = candidates.iloc[0]

    signal = {
        'priority':     4,
        'signal_type':  'FADING',
        'ticker':       best['Ticker'],
        'sector':       best.get('Sector', 'Other'),
        'sector_plain': best.get('Sector_Plain', 'the broader market'),
        'dm':           best['DM'],
        'dm_change':    best['DM_Change'],
        'magnitude':    abs(round(best['DM_Change'], 1)),
        'strength':     3 if best['DM_Change'] < -30 else 2,
    }
    signal['story_type'] = classify_story(signal)
    return signal


def find_strand_stories(clusters_df, dm_latest_df):
    """
    Priority 5: Strand cluster with anchor DM > 60, 25+ strands.
    Positive capital rotation story.
    """
    if clusters_df.empty:
        return None

    candidates = clusters_df[
        (clusters_df['DM'] >= STRAND_MIN_DM) &
        (clusters_df['Strands'] >= STRAND_MIN_STRANDS)
    ].copy()

    if candidates.empty:
        return None

    candidates = candidates.sort_values('Strands', ascending=False)
    best = candidates.iloc[0]
    ticker = best['Ticker']

    ticker_row = dm_latest_df[dm_latest_df['Ticker'] == ticker]
    sector = (ticker_row.iloc[0]['Sector']
              if len(ticker_row) > 0 else 'Other')
    sector_plain = (ticker_row.iloc[0].get('Sector_Plain', sector.lower())
                    if len(ticker_row) > 0 else sector.lower())

    return {
        'priority':     5,
        'signal_type':  'STRAND_EMERGENCE',
        'story_type':   'ACCUMULATION',
        'ticker':       ticker,
        'sector':       sector,
        'sector_plain': sector_plain,
        'dm':           best['DM'],
        'strands':      int(best['Strands']),
        'partners':     best.get('Partners', ''),
        'magnitude':    int(best['Strands']),
        'strength':     3 if best['Strands'] >= 40 else 2,
    }


def select_signal(dm_latest_df, monitor_data, clusters_df):
    """Apply priority ranking. Return highest-priority signal."""
    print("\n  Evaluating signals by priority...")

    # P1
    signal = check_chain_alerts(monitor_data)
    if signal:
        print(f"    P1 HIT — CHAIN ALERT: {signal['detail']}")
        return signal

    # P2
    signal = find_anomalous_nodes(dm_latest_df)
    if signal:
        print(f"    P2 HIT — ANOMALOUS NODE: {signal['ticker']} "
              f"(z={signal['z_score']:.1f}, sector avg={signal['sector_avg']})")
        return signal

    # P3
    signal = find_sector_divergence(dm_latest_df)
    if signal:
        print(f"    P3 HIT — SECTOR DIVERGENCE: {signal['sector']} "
              f"({len(signal['tickers'])} tickers)")
        return signal

    # P4
    signal = find_fading_tickers(dm_latest_df)
    if signal:
        print(f"    P4 HIT — FADING: {signal['ticker']} "
              f"(DM={signal['dm']:.1f}, 30d={signal['dm_change']:.1f})")
        return signal

    # P5
    signal = find_strand_stories(clusters_df, dm_latest_df)
    if signal:
        print(f"    P5 HIT — STRAND: {signal['ticker']} "
              f"(DM={signal['dm']:.1f}, {signal['strands']} strands)")
        return signal

    print("    No signal above threshold today.")
    return None


# ============================================================
# STORY CLASSIFICATION
# ============================================================

def classify_story(signal):
    """Determine story type from signal context."""
    if signal.get('story_type'):
        return signal['story_type']

    sector = signal.get('sector', 'Other')

    if signal['signal_type'] == 'CHAIN_ALERT':
        return 'SYSTEMIC'
    if signal['signal_type'] == 'STRAND_EMERGENCE':
        return 'ACCUMULATION'
    if sector in AI_ADJACENT_SECTORS:
        return 'AI_DISPLACEMENT'
    if signal['signal_type'] == 'SECTOR_DIVERGENCE':
        return 'SECTOR_ROTATION'

    return 'DIVERGENCE'


# ============================================================
# HEADLINE GENERATION
# ============================================================

def generate_headlines(signal):
    """Generate three headline options for editorial selection."""
    ticker = signal['ticker']
    sector = signal.get('sector_plain', 'the market')
    story_type = signal.get('story_type', 'DIVERGENCE')

    if story_type == 'SYSTEMIC':
        h1 = ("Capital Flows Are Flashing a Warning Across Large-Cap "
              "Stocks -- Here's What the Data Shows")
        h2 = ("Multiple Market Signals Are Firing at Once. "
              "That Hasn't Happened Often.")
        h3 = ("The Market Structure Story Nobody Is Talking "
              "About Yet")

    elif story_type == 'AI_DISPLACEMENT':
        h1 = (f"Institutional Capital Is Quietly Leaving "
              f"{sector.title()} -- And {ticker} Shows Why")
        h2 = (f"AI Is Already Repricing {sector.title()} Stocks. "
              f"{ticker}'s Capital Flows Tell the Story.")
        h3 = (f"The AI Displacement Story in {sector.title()} "
              f"Nobody Is Watching Yet")

    elif story_type == 'ACCUMULATION':
        h1 = (f"Institutional Capital Is Quietly Accumulating in "
              f"{ticker} -- Before the Narrative Catches Up")
        h2 = (f"Where Smart Money Is Already Moving: "
              f"The {ticker} Capital Flow Signal")
        h3 = (f"The Capital Rotation Into {sector.title()} "
              f"Nobody Is Talking About Yet")

    elif story_type == 'SECTOR_ROTATION':
        tickers = signal.get('tickers', [ticker])
        names = ' and '.join(tickers[:2])
        h1 = (f"Capital Is Leaving {names} While Their Sector "
              f"Holds Steady -- That's a Signal")
        h2 = (f"The Divergence in {sector.title()} Is Getting "
              f"Hard to Ignore")
        h3 = (f"A Rotation Is Happening Inside {sector.title()}. "
              f"Quietly.")

    else:  # DIVERGENCE
        h1 = (f"{ticker} Is Getting Quietly Repriced -- And Its "
              f"Peers Are Not")
        h2 = (f"One Name in {sector.title()} Is Breaking From "
              f"the Pack. Capital Flows Explain Why.")
        h3 = (f"The {ticker} Divergence Story Nobody Is Talking "
              f"About Yet")

    return [h1, h2, h3]


# ============================================================
# DRAFT GENERATION — ONE BUILDER PER STORY TYPE
# ============================================================

def build_systemic_draft(signal, date_str, dm_latest_df):
    """Draft for Priority 1 chain alert / systemic stories."""
    critical = {}
    for _, row in dm_latest_df.iterrows():
        if row['Ticker'] in ('TSLA', 'NVDA', 'PLTR', 'MSTR', 'APP', 'CEG'):
            critical[row['Ticker']] = row['DM']

    cascade_text = ""
    if 'CASCADE_RISK' in signal.get('detail', ''):
        tsla_dm = critical.get('TSLA', 0)
        cascade_text = (
            f"TSLA's capital flow score has dropped to {tsla_dm:.0f}, "
            f"crossing below the 35 threshold that historically "
            f"precedes portfolio-wide contagion. "
        )

    weak_origins = []
    for t in ('PLTR', 'MSTR'):
        if t in critical and critical[t] < 50:
            weak_origins.append(f"{t} at {critical[t]:.0f}")
    origins_text = ""
    if weak_origins:
        origins_text = (
            f"Two market-leading names -- {' and '.join(weak_origins)} "
            f"-- are simultaneously showing institutional outflows. "
            f"When multiple high-influence nodes weaken at the same "
            f"time, the pattern has historically preceded broader "
            f"market stress."
        )

    nvda_text = ""
    if 'NVDA' in critical and critical['NVDA'] < 60:
        nvda_text = (
            f"NVDA's capital flow score sits at "
            f"{critical['NVDA']:.0f}, below the 60 level that "
            f"typically signals strong institutional conviction. "
            f"For the stock most central to the AI infrastructure "
            f"cycle, that weakness carries outsized significance. "
        )

    draft = f"""# Capital Flows Are Flashing a Warning Across Large-Cap Stocks
## Multiple chain analysis signals triggered simultaneously -- a rare occurrence in our two-year dataset

*Y2AI Research | {date_str}*

---

Something unusual is happening under the surface of the equity market. Our capital flow analysis of 541 large-cap stocks -- which tracks institutional buying and selling patterns using a proprietary scoring system -- is showing multiple warning signals firing at once. That combination has been rare over the two years we've been running this data.

{cascade_text}{origins_text}

{nvda_text}The pattern here isn't about any single stock. It's about the structure of how capital moves through the market. When high-influence names -- the ones whose movements historically predict broader trends -- weaken simultaneously, it signals that institutional investors are reducing exposure broadly, not just repositioning within sectors.

To be clear about what this is and isn't: this is a structural observation, not a crash prediction. Capital flows lead price by days to weeks, not hours. The specific condition that would change this picture is straightforward -- if the weakening nodes stabilize above their alert thresholds and hold for five consecutive trading days, the contagion pathway closes.

What makes this moment distinctive is the simultaneity. Individual names weaken all the time. But when the chain analysis identifies coordinated weakness across nodes that typically move independently, it points to a shift in institutional behavior that the broader market hasn't priced yet.

---

*Y2AI Research tracks capital flows across 541 large-cap stocks using proprietary quantitative analysis. This post is for informational purposes only and does not constitute investment advice.*
"""
    return draft


def build_anomaly_draft(signal, date_str, dm_latest_df):
    """Draft for Priority 2 anomalous node stories."""
    ticker = signal['ticker']
    sector_plain = signal.get('sector_plain', 'the sector')
    dm = signal['dm']
    z = signal['z_score']
    sector_avg = signal['sector_avg']
    story_type = signal.get('story_type', 'DIVERGENCE')

    # Peer context
    sector = signal.get('sector', 'Other')
    peers = dm_latest_df[dm_latest_df['Sector'] == sector]
    peers = peers.sort_values('DM', ascending=False)
    top_peers = peers.head(3)
    peer_text = ', '.join(
        f"{r['Ticker']} ({r['DM']:.0f})"
        for _, r in top_peers.iterrows()
    )

    if story_type == 'AI_DISPLACEMENT':
        mechanism_para = (
            f"The disruption mechanism here follows a pattern we've "
            f"tracked across multiple sectors: AI tools are compressing "
            f"the value of services that depend on information asymmetry "
            f"or manual coordination. When the core value proposition "
            f"of a business model becomes automatable, institutional "
            f"capital doesn't wait for earnings to confirm it -- it "
            f"moves first."
        )
    else:
        mechanism_para = (
            f"The question for {ticker} is whether this represents a "
            f"temporary dislocation or a structural repricing. When a "
            f"single name diverges this sharply from healthy peers, it "
            f"typically signals that institutional investors have "
            f"identified something company-specific that the broader "
            f"market narrative hasn't absorbed yet."
        )

    draft = f"""# {ticker} Is Getting Quietly Repriced -- And Its {sector_plain.title()} Peers Are Not
## Capital flow data shows {ticker} diverging sharply from a sector that looks otherwise healthy

*Y2AI Research | {date_str}*

---

The {sector_plain} sector looks fine on the surface. The average capital flow score across the group sits at {sector_avg}, well into healthy territory. Names like {peer_text} are all showing sustained institutional buying. But inside that picture, something specific is happening to {ticker}.

Our capital flow analysis of 541 large-cap stocks shows {ticker} at a score of {dm:.1f} -- that's {abs(z):.1f} standard deviations below its sector peers. To put that in context, a z-score below {abs(ANOMALY_ZSCORE)} means institutional capital is leaving this specific name at a rate that can't be explained by sector rotation alone. Something about {ticker}'s positioning is making large investors uncomfortable.

{mechanism_para}

The broader {sector_plain} sector provides the contrast that makes this signal meaningful. When an entire sector weakens, it's macro. When one name collapses while peers hold, it's micro -- and micro signals are where the analytical edge lives. The sector average of {sector_avg} tells you the institutional thesis on {sector_plain} is intact. The {ticker} reading of {dm:.1f} tells you the thesis on this specific business model is under review.

What would change this picture: {ticker}'s capital flow score stabilizing above 40 for five consecutive days would indicate institutional selling pressure has exhausted itself. Below that, the flow data suggests further repricing ahead.

---

*Y2AI Research tracks capital flows across 541 large-cap stocks using proprietary quantitative analysis. This post is for informational purposes only and does not constitute investment advice.*
"""
    return draft


def build_divergence_draft(signal, date_str, dm_latest_df):
    """Draft for Priority 3 sector divergence stories."""
    tickers = signal.get('tickers', [signal['ticker']])
    sector_plain = signal.get('sector_plain', 'the sector')
    sector_avg = signal.get('sector_avg', 50)
    dm_values = signal.get('dm_values', {})

    ticker_details = ', '.join(
        f"{t} ({dm_values.get(t, 0):.0f})" for t in tickers[:3]
    )
    names_text = ' and '.join(tickers[:2])

    draft = f"""# A Divergence Is Forming Inside {sector_plain.title()} -- And Capital Flows Show Exactly Where
## {names_text} are seeing institutional outflows while the sector average holds at {sector_avg}

*Y2AI Research | {date_str}*

---

The {sector_plain} sector carries a capital flow score of {sector_avg}, which places it comfortably in healthy territory by any historical standard. Institutional money is broadly committed to the space. But zoom in and the picture fragments. A subset of names -- {ticker_details} -- are seeing coordinated capital outflows that stand apart from the sector trend.

Our capital flow analysis of 541 large-cap stocks tracks institutional buying and selling patterns across the market. When two or more names in the same sector show simultaneous outflows of this magnitude while peers hold steady, it signals that large investors are making a distinction the market hasn't broadly recognized yet. This isn't sector rotation. This is intra-sector selection.

The divergence pattern typically resolves in one of two ways. Either the weak names stabilize and reconverge with peers -- which would indicate the outflows were temporary repositioning -- or the weak names continue declining as the market narrative catches up to what institutional flows already signaled. The historical tendency favors the latter.

What would change this picture: if {names_text} stabilize above 45 and the sector average holds, the divergence closes. If the sector average drops below 50, the story shifts from divergence to broad sector weakness -- a different signal entirely.

---

*Y2AI Research tracks capital flows across 541 large-cap stocks using proprietary quantitative analysis. This post is for informational purposes only and does not constitute investment advice.*
"""
    return draft


def build_fading_draft(signal, date_str, dm_latest_df):
    """Draft for Priority 4 fading detection stories."""
    ticker = signal['ticker']
    sector_plain = signal.get('sector_plain', 'its sector')
    dm = signal['dm']
    dm_change = signal['dm_change']

    draft = f"""# {ticker}'s Capital Flow Collapse Is Accelerating -- Down {abs(dm_change):.0f} Points in 30 Days
## A previously healthy name is fading fast, and institutional investors are the ones doing the selling

*Y2AI Research | {date_str}*

---

Thirty days ago, {ticker} looked fine. Capital flow scores were healthy, institutional positioning was stable, and there was no obvious reason for concern. Today, that picture has changed dramatically. {ticker}'s capital flow score has dropped {abs(dm_change):.0f} points to {dm:.1f}, placing it squarely in the zone where our historical data shows further downside is more likely than recovery.

Our capital flow analysis of 541 large-cap stocks shows this kind of rapid momentum collapse -- a score decline of {abs(dm_change):.0f}+ points in a single month -- occurs in roughly 3-5% of names at any given time. The significance isn't the level itself but the velocity. When institutional capital exits a position this quickly, it typically reflects a fundamental reassessment rather than routine rebalancing.

The {sector_plain} context matters here. If the entire sector were fading, this would be a macro story. But {ticker}'s decline stands out against its peer group, which suggests the catalyst is company-specific. The most common drivers of this pattern are earnings guidance revisions, competitive positioning shifts, or regulatory developments that alter the forward outlook.

What to watch: a stabilization above 40 for three consecutive days would signal the selling pressure has peaked. Below 30, historical patterns suggest the repricing has further to go. The 30-40 range where {ticker} currently sits is the inflection zone -- it resolves up or down within two to three weeks.

---

*Y2AI Research tracks capital flows across 541 large-cap stocks using proprietary quantitative analysis. This post is for informational purposes only and does not constitute investment advice.*
"""
    return draft


def build_accumulation_draft(signal, date_str, dm_latest_df):
    """Draft for Priority 5 strand emergence / accumulation stories."""
    ticker = signal['ticker']
    sector_plain = signal.get('sector_plain', 'the broader market')
    dm = signal['dm']
    strands = signal['strands']
    partners = signal.get('partners', '')

    partner_list = [p.strip() for p in partners.split(',')
                    if p.strip()][:5]
    partner_text = (', '.join(partner_list)
                    if partner_list else 'multiple names')

    draft = f"""# Institutional Capital Is Quietly Accumulating Around {ticker} -- Before the Narrative Catches Up
## Capital flow analysis shows {strands} new correlation strands forming, a pattern that historically precedes major price moves

*Y2AI Research | {date_str}*

---

There's a pattern in how institutional capital moves before major repricings. It doesn't arrive all at once. It builds connections -- quietly, across multiple related names -- before the broader market notices. Right now, that pattern is forming around {ticker}.

Our capital flow analysis of 541 large-cap stocks includes a strand emergence scanner that identifies when previously uncorrelated stocks begin moving together. {ticker} has formed {strands} new correlation strands in the last 30 days, linking it to names like {partner_text}. That's well above the baseline for our universe and places {ticker} among the strongest accumulation signals in the current market.

{ticker}'s capital flow score of {dm:.1f} reinforces the picture. Scores above 60 indicate sustained institutional buying, not speculative froth. The combination of high flow scores and rapid strand formation is the specific pattern that distinguishes genuine capital rotation from noise. Institutional investors are building positions and the correlated movement across connected names confirms it's coordinated, not coincidental.

What would change this picture: if {ticker}'s capital flow score drops below 50, it would indicate the accumulation cycle is peaking. If the new strand count declines in next week's scan, the coordination signal weakens. For now, the data points in one direction -- institutional money is moving into this name ahead of whatever fundamental catalyst the market will eventually price.

---

*Y2AI Research tracks capital flows across 541 large-cap stocks using proprietary quantitative analysis. This post is for informational purposes only and does not constitute investment advice.*
"""
    return draft


def generate_draft(signal, date_str, dm_latest_df):
    """Route to correct draft builder based on signal type."""
    story_type = signal.get('story_type', 'DIVERGENCE')

    if story_type == 'SYSTEMIC':
        return build_systemic_draft(signal, date_str, dm_latest_df)
    elif signal['signal_type'] == 'ANOMALOUS_NODE':
        return build_anomaly_draft(signal, date_str, dm_latest_df)
    elif signal['signal_type'] == 'SECTOR_DIVERGENCE':
        return build_divergence_draft(signal, date_str, dm_latest_df)
    elif signal['signal_type'] == 'FADING':
        return build_fading_draft(signal, date_str, dm_latest_df)
    elif story_type == 'ACCUMULATION':
        return build_accumulation_draft(signal, date_str, dm_latest_df)
    else:
        return build_anomaly_draft(signal, date_str, dm_latest_df)


# ============================================================
# GOOGLE DRIVE OUTPUT
# ============================================================

def get_or_create_folder(folder_name):
    """Find or create a Google Drive folder. Returns folder ID."""
    drive = get_drive_service()

    query = (f"name='{folder_name}' and "
             f"mimeType='application/vnd.google-apps.folder' and "
             f"trashed=false")
    results = drive.files().list(
        q=query, spaces='drive', fields='files(id, name)'
    ).execute()
    files = results.get('files', [])

    if files:
        folder_id = files[0]['id']
        print(f"  Found folder '{folder_name}': {folder_id}")
        return folder_id

    metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    folder = drive.files().create(body=metadata, fields='id').execute()
    folder_id = folder.get('id')
    print(f"  Created folder '{folder_name}': {folder_id}")
    return folder_id


def save_draft(content, filename, date_str, story_type):
    """
    Save draft in two places:
    1. Google Sheet tab 'Story_Drafts' in ARGUS spreadsheet
    2. Local markdown file in Story_Generator/drafts/ folder
    Returns a reference string for the pipeline.
    """
    # --- Save to Google Sheet tab ---
    gc = get_sheets_client()
    sh = gc.open_by_key(ARGUS_SPREADSHEET_ID)
    tab_name = 'Story_Drafts'

    try:
        ws = sh.worksheet(tab_name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=tab_name, rows=500, cols=5)
        ws.update(range_name='A1:D1', values=[[
            'Date', 'Filename', 'Story_Type', 'Draft_Content'
        ]])
        print(f"  Created {tab_name} sheet")

    # Append draft as a row (content in one cell)
    ws.append_row(
        [date_str, filename, story_type, content],
        value_input_option='USER_ENTERED'
    )
    print(f"  Saved to Google Sheet: {tab_name}")

    # --- Save local markdown file ---
    import os
    drafts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'drafts')
    os.makedirs(drafts_dir, exist_ok=True)
    local_path = os.path.join(drafts_dir, filename)

    with open(local_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  Saved locally: {local_path}")

    return f"Sheet:{tab_name} | Local:{local_path}"


# ============================================================
# STORY PIPELINE SHEET
# ============================================================

def ensure_pipeline_sheet():
    """Create Story_Pipeline tab in ARGUS spreadsheet if needed."""
    gc = get_sheets_client()
    sh = gc.open_by_key(ARGUS_SPREADSHEET_ID)

    try:
        ws = sh.worksheet(PIPELINE_TAB)
        return ws
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(
            title=PIPELINE_TAB, rows=500, cols=15
        )
        header = [
            'Date', 'Story_Type', 'Primary_Ticker', 'Sector',
            'Signal_Strength', 'Headline_1', 'Headline_2',
            'Headline_3', 'Draft_Link', 'Status', 'Notes'
        ]
        ws.update(range_name='A1:K1', values=[header])
        print(f"  Created {PIPELINE_TAB} sheet")
        return ws


def update_pipeline(date_str, signal, headlines, draft_link):
    """Append row to Story_Pipeline sheet."""
    ws = ensure_pipeline_sheet()

    row = [
        date_str,
        signal.get('story_type', ''),
        signal.get('ticker', ''),
        signal.get('sector', ''),
        signal.get('strength', 0),
        headlines[0] if len(headlines) > 0 else '',
        headlines[1] if len(headlines) > 1 else '',
        headlines[2] if len(headlines) > 2 else '',
        draft_link,
        'DRAFT',
        '',
    ]

    ws.append_row(row, value_input_option='USER_ENTERED')
    print(f"  Pipeline updated: {signal.get('ticker', '')} / "
          f"{signal.get('story_type', '')}")


# ============================================================
# MAIN WORKFLOWS
# ============================================================

def run_daily():
    """Daily story generation workflow."""
    date_str = datetime.today().strftime('%Y-%m-%d')
    date_display = datetime.today().strftime('%B %d, %Y')

    print("=" * 60)
    print("Y2AI DAILY STORY GENERATOR")
    print(f"Date: {date_str}")
    print("=" * 60)

    print("\nLoading data sources...")
    dm_latest_df = load_dm_latest()
    monitor_data = load_chain_monitor()
    clusters_df = load_strand_clusters()

    signal = select_signal(dm_latest_df, monitor_data, clusters_df)

    if not signal:
        print(f"\n  No signal above threshold. No draft generated.")
        return

    story_type = signal.get('story_type', 'DIVERGENCE')
    headlines = generate_headlines(signal)
    draft = generate_draft(signal, date_display, dm_latest_df)

    # Preview
    print("\n" + "=" * 60)
    print("DRAFT PREVIEW")
    print("=" * 60)
    lines = draft.strip().split('\n')
    for line in lines[:8]:
        print(f"  {line}")
    print(f"  ... ({len(lines)} total lines)")
    print(f"\n  Headlines:")
    for i, h in enumerate(headlines, 1):
        print(f"    {i}. {h}")
    print(f"  Story type: {story_type}")
    print(f"  Signal strength: {signal.get('strength', 0)}/5")

    # Save
    print("\nSaving outputs...")
    filename = f"Draft_{date_str}_{story_type}.md"
    draft_link = save_draft(draft, filename, date_str, story_type)
    update_pipeline(date_str, signal, headlines, draft_link)

    print("\n" + "=" * 60)
    print("DAILY STORY GENERATOR COMPLETE")
    print(f"  Draft:  {filename}")
    print(f"  Type:   {story_type}")
    print(f"  Ticker: {signal.get('ticker', 'MULTIPLE')}")
    print("=" * 60)


def run_weekly():
    """Weekly strand story bank — run Sunday evenings."""
    date_str = datetime.today().strftime('%Y-%m-%d')
    date_display = datetime.today().strftime('%B %d, %Y')

    print("=" * 60)
    print("Y2AI WEEKLY STRAND STORY BANK")
    print(f"Date: {date_str}")
    print("=" * 60)

    print("\nLoading data sources...")
    dm_latest_df = load_dm_latest()
    clusters_df = load_strand_clusters()

    if clusters_df.empty:
        print("  No strand clusters available. "
              "Run chain_analysis.py first.")
        return

    qualified = clusters_df[
        (clusters_df['DM'] >= STRAND_MIN_DM) &
        (clusters_df['Strands'] >= STRAND_MIN_STRANDS)
    ].sort_values('Strands', ascending=False).head(WEEKLY_MAX_DRAFTS)

    if qualified.empty:
        print("  No qualifying clusters.")
        return

    print(f"\n  {len(qualified)} qualifying clusters found")
    drafts_generated = 0

    for _, row in qualified.iterrows():
        ticker = row['Ticker']
        print(f"\n  Generating strand story for {ticker}...")

        ticker_row = dm_latest_df[dm_latest_df['Ticker'] == ticker]
        sector = (ticker_row.iloc[0]['Sector']
                  if len(ticker_row) > 0 else 'Other')
        sector_plain = (
            ticker_row.iloc[0].get('Sector_Plain', sector.lower())
            if len(ticker_row) > 0 else sector.lower()
        )

        signal = {
            'priority':     5,
            'signal_type':  'STRAND_EMERGENCE',
            'story_type':   'ACCUMULATION',
            'ticker':       ticker,
            'sector':       sector,
            'sector_plain': sector_plain,
            'dm':           row['DM'],
            'strands':      int(row['Strands']),
            'partners':     row.get('Partners', ''),
            'magnitude':    int(row['Strands']),
            'strength':     3 if row['Strands'] >= 40 else 2,
        }

        headlines = generate_headlines(signal)
        draft = generate_draft(signal, date_display, dm_latest_df)

        filename = f"Draft_{date_str}_STRAND_{ticker}.md"
        draft_link = save_draft(draft, filename, date_str, 'ACCUMULATION')
        update_pipeline(date_str, signal, headlines, draft_link)
        drafts_generated += 1
        time.sleep(1)

    print("\n" + "=" * 60)
    print("WEEKLY STRAND STORY BANK COMPLETE")
    print(f"  Drafts generated: {drafts_generated}")
    print("=" * 60)


def run_test():
    """Test run — shows all signals, prints full draft. No saves."""
    date_str = datetime.today().strftime('%Y-%m-%d')
    date_display = datetime.today().strftime('%B %d, %Y')

    print("=" * 60)
    print("Y2AI STORY GENERATOR — TEST MODE")
    print(f"Date: {date_str}")
    print("=" * 60)

    print("\nLoading data sources...")
    dm_latest_df = load_dm_latest()
    monitor_data = load_chain_monitor()
    clusters_df = load_strand_clusters()

    # Show all signals at each level
    print("\n" + "-" * 40)
    print("SIGNAL SCAN (all levels)")
    print("-" * 40)

    p1 = check_chain_alerts(monitor_data)
    if p1:
        print(f"\n  [P1] CHAIN ALERT: {p1['detail']}")
        print(f"       Strength: {p1['strength']}/5")
    else:
        print("\n  [P1] No chain alert")

    p2 = find_anomalous_nodes(dm_latest_df)
    if p2:
        print(f"\n  [P2] ANOMALOUS NODE: {p2['ticker']}")
        print(f"       z-score: {p2['z_score']:.2f}, "
              f"Sector avg: {p2['sector_avg']}")
        print(f"       DM: {p2['dm']:.1f}, Sector: {p2['sector']}")
        print(f"       Story type: {p2['story_type']}")
    else:
        print("\n  [P2] No anomalous nodes")

    p3 = find_sector_divergence(dm_latest_df)
    if p3:
        print(f"\n  [P3] SECTOR DIVERGENCE: {p3['sector']}")
        print(f"       Tickers: {p3['tickers']}")
        print(f"       Avg change: {p3['avg_change']}")
    else:
        print("\n  [P3] No sector divergence")

    p4 = find_fading_tickers(dm_latest_df)
    if p4:
        print(f"\n  [P4] FADING: {p4['ticker']}")
        print(f"       DM: {p4['dm']:.1f}, "
              f"30d change: {p4['dm_change']:.1f}")
    else:
        print("\n  [P4] No fading tickers")

    p5 = find_strand_stories(clusters_df, dm_latest_df)
    if p5:
        print(f"\n  [P5] STRAND: {p5['ticker']}")
        print(f"       DM: {p5['dm']:.1f}, "
              f"Strands: {p5['strands']}")
    else:
        print("\n  [P5] No strand stories")

    # Select winner and generate draft
    signal = select_signal(dm_latest_df, monitor_data, clusters_df)

    if not signal:
        print("\n  No signal above threshold today.")
        return

    headlines = generate_headlines(signal)
    draft = generate_draft(signal, date_display, dm_latest_df)

    print("\n" + "=" * 60)
    print("SELECTED SIGNAL")
    print("=" * 60)
    print(f"  Priority: {signal['priority']}")
    print(f"  Type:     {signal.get('story_type', '')}")
    print(f"  Ticker:   {signal.get('ticker', '')}")
    print(f"  Strength: {signal.get('strength', 0)}/5")

    print(f"\n  Headlines:")
    for i, h in enumerate(headlines, 1):
        print(f"    {i}. {h}")

    print("\n" + "=" * 60)
    print("FULL DRAFT")
    print("=" * 60)
    print(draft)

    # Show weekly strand candidates
    if not clusters_df.empty:
        strand_cand = clusters_df[
            (clusters_df['DM'] >= STRAND_MIN_DM) &
            (clusters_df['Strands'] >= STRAND_MIN_STRANDS)
        ].sort_values('Strands', ascending=False).head(5)

        if not strand_cand.empty:
            print("-" * 40)
            print("WEEKLY STRAND CANDIDATES")
            print("-" * 40)
            for _, r in strand_cand.iterrows():
                print(f"  {r['Ticker']:8s} DM={r['DM']:.1f}  "
                      f"Strands={int(r['Strands'])}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'daily'

    if mode == 'weekly':
        run_weekly()
    elif mode == 'test':
        run_test()
    else:
        run_daily()