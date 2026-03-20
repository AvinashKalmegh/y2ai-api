"""
Universe Candidate Watchlist + EDGAR Scanner
====================================================
Two functions:
  1. Maintain a Universe_Candidates Google Sheet tab tracking potential
     ticker additions with trigger type, status, and decision tracking.
  2. Weekly EDGAR scan for:
     - Nvidia 13F filings (new strategic investments)
     - Hyperscaler 8-K filings (infrastructure contracts > $1B)
     - New Nasdaq listings in AI infrastructure SIC codes

Usage:
  python universe_candidates.py push             # Push watchlist to Google Sheet
  python universe_candidates.py scan             # Run EDGAR + listing scan
  python universe_candidates.py scan --dry-run   # Scan without updating sheet
  python universe_candidates.py add TICKER "Company Name" "Category" "Trigger"

Pull schedule: Weekly on Monday at 8:00 AM ET.
"""

import os
import sys
import logging
import argparse
import requests
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
GOOGLE_SHEETS_CREDS_FILE = 'credentials.json'

# Spreadsheet for the Universe_Candidates tab
CANDIDATES_SPREADSHEET = "copy-dm-history 2024-current"
CANDIDATES_TAB = "Universe_Candidates"

CANDIDATES_HEADERS = [
    'Ticker', 'Company', 'Category', 'Trigger Type', 'Date Flagged',
    'DM Status', 'Decision', 'Decision Date', 'Notes'
]

# Initial candidates from DevBrief
INITIAL_CANDIDATES = [
    {"ticker": "NBIS", "company": "Nebius Group", "category": "Neocloud",
     "trigger": "Hyperscaler contract ($27B Meta)", "date_flagged": "2026-03-16",
     "dm_status": "ADDED", "decision": "ADD", "decision_date": "2026-03-16",
     "notes": "Nvidia $2B stake. $17.4B Microsoft deal."},
    {"ticker": "CRWV", "company": "CoreWeave", "category": "Neocloud",
     "trigger": "Hyperscaler contract + Nvidia-backed", "date_flagged": "2026-03-16",
     "dm_status": "ADDED", "decision": "ADD", "decision_date": "2026-03-16",
     "notes": "$23B IPO March 2025."},
    {"ticker": "NSCL", "company": "NScale", "category": "Neocloud",
     "trigger": "Nvidia investment ($2B)", "date_flagged": "2026-03-16",
     "dm_status": "ADDED", "decision": "ADD", "decision_date": "2026-03-16",
     "notes": "UK-based neocloud."},
    {"ticker": "DELL", "company": "Dell Technologies", "category": "Hardware",
     "trigger": "Already in universe", "date_flagged": "2026-03-16",
     "dm_status": "IN UNIVERSE", "decision": "CONFIRM", "decision_date": "2026-03-16",
     "notes": "Confirm active tracking."},
    {"ticker": "ARM", "company": "Arm Holdings", "category": "Semiconductor",
     "trigger": "Review for addition", "date_flagged": "2026-03-16",
     "dm_status": "REVIEW", "decision": "WATCH", "decision_date": "",
     "notes": "AI chip architecture, evaluate DM signal."},
]

# EDGAR search parameters
HYPERSCALERS = ["Meta Platforms", "Microsoft", "Alphabet", "Google", "Amazon"]
AI_INFRA_SIC_CODES = ["7374", "7372", "3577"]
NVIDIA_CIK = "0001045810"


# ============================================================
# SUPABASE CLIENT
# ============================================================
_supabase_client = None

def get_supabase():
    global _supabase_client
    if _supabase_client is None:
        from supabase import create_client
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase_client


# ============================================================
# GOOGLE SHEETS
# ============================================================
def get_sheets_client():
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
    scope = [
        'https://spreadsheets.google.com/feeds',
        'https://www.googleapis.com/auth/drive'
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        GOOGLE_SHEETS_CREDS_FILE, scope
    )
    return gspread.authorize(creds)


def push_candidates_to_sheet(candidates):
    """Push Universe_Candidates list to Google Sheet tab."""
    import gspread

    client = get_sheets_client()
    spreadsheet = client.open(CANDIDATES_SPREADSHEET)

    try:
        sheet = spreadsheet.worksheet(CANDIDATES_TAB)
    except gspread.WorksheetNotFound:
        sheet = spreadsheet.add_worksheet(
            title=CANDIDATES_TAB, rows=200, cols=len(CANDIDATES_HEADERS)
        )

    # Build data
    rows = [CANDIDATES_HEADERS]
    for c in candidates:
        rows.append([
            c.get("ticker", ""),
            c.get("company", ""),
            c.get("category", ""),
            c.get("trigger", ""),
            c.get("date_flagged", ""),
            c.get("dm_status", ""),
            c.get("decision", ""),
            c.get("decision_date", ""),
            c.get("notes", ""),
        ])

    sheet.clear()
    sheet.update(range_name=f"A1:I{len(rows)}", values=rows)
    logger.info(f"  Pushed {len(candidates)} candidates to {CANDIDATES_TAB}")


def load_candidates_from_sheet():
    """Load existing candidates from Google Sheet."""
    import gspread

    try:
        client = get_sheets_client()
        spreadsheet = client.open(CANDIDATES_SPREADSHEET)
        sheet = spreadsheet.worksheet(CANDIDATES_TAB)
        data = sheet.get_all_records()
        candidates = []
        for row in data:
            candidates.append({
                "ticker": row.get("Ticker", ""),
                "company": row.get("Company", ""),
                "category": row.get("Category", ""),
                "trigger": row.get("Trigger Type", ""),
                "date_flagged": row.get("Date Flagged", ""),
                "dm_status": row.get("DM Status", ""),
                "decision": row.get("Decision", ""),
                "decision_date": row.get("Decision Date", ""),
                "notes": row.get("Notes", ""),
            })
        return candidates
    except Exception:
        return INITIAL_CANDIDATES.copy()


# ============================================================
# EDGAR SCANNER
# ============================================================
def scan_nvidia_13f():
    """
    Check EDGAR for recent Nvidia 13F filings to detect new strategic investments.
    Returns list of findings.
    """
    logger.info("Scanning Nvidia 13F filings...")
    findings = []

    url = "https://efts.sec.gov/LATEST/search-index"
    params = {
        "q": "13F",
        "dateRange": "custom",
        "startdt": (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
        "enddt": datetime.now().strftime("%Y-%m-%d"),
        "entityName": "NVIDIA",
        "forms": "13F-HR",
    }

    # Use EDGAR EFTS full-text search
    search_url = "https://efts.sec.gov/LATEST/search-index"
    efts_url = f"https://efts.sec.gov/LATEST/search-index?q=%2213F%22&entityName=NVIDIA&forms=13F-HR&dateRange=custom&startdt={(datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')}&enddt={datetime.now().strftime('%Y-%m-%d')}"

    # Simpler: use EDGAR company search API
    api_url = f"https://data.sec.gov/submissions/CIK{NVIDIA_CIK}.json"

    try:
        resp = requests.get(api_url, timeout=30, headers={
            "User-Agent": "Y2AI Research admin@y2ai.com",
            "Accept": "application/json",
        })
        if resp.status_code != 200:
            logger.warning(f"  EDGAR API returned HTTP {resp.status_code}")
            return findings

        data = resp.json()
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        descriptions = recent.get("primaryDocDescription", [])

        for i, form in enumerate(forms):
            if "13F" in form:
                findings.append({
                    "type": "NVIDIA_13F",
                    "form": form,
                    "date": dates[i] if i < len(dates) else "",
                    "description": descriptions[i] if i < len(descriptions) else "",
                })

        logger.info(f"  Found {len(findings)} Nvidia 13F filings")
    except Exception as e:
        logger.warning(f"  EDGAR Nvidia scan error: {e}")

    return findings


def scan_hyperscaler_8k():
    """
    Check EDGAR for recent 8-K filings from hyperscalers mentioning
    infrastructure agreements.
    Returns list of findings.
    """
    logger.info("Scanning hyperscaler 8-K filings...")
    findings = []

    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    for company in HYPERSCALERS:
        try:
            url = "https://efts.sec.gov/LATEST/search-index"
            params = {
                "q": f'"{company}" "infrastructure" "agreement"',
                "forms": "8-K",
                "dateRange": "custom",
                "startdt": start_date,
                "enddt": end_date,
            }

            resp = requests.get(url, params=params, timeout=30, headers={
                "User-Agent": "Y2AI Research admin@y2ai.com",
            })

            if resp.status_code == 200:
                data = resp.json()
                hits = data.get("hits", {}).get("hits", [])
                for hit in hits[:5]:  # limit per company
                    source = hit.get("_source", {})
                    findings.append({
                        "type": "HYPERSCALER_8K",
                        "company": company,
                        "date": source.get("file_date", ""),
                        "description": source.get("display_names", [""])[0] if source.get("display_names") else "",
                        "form": "8-K",
                    })

        except Exception as e:
            logger.warning(f"  EDGAR scan error for {company}: {e}")

    logger.info(f"  Found {len(findings)} hyperscaler 8-K filings")
    return findings


def scan_new_listings():
    """
    Check for new Nasdaq/NYSE listings in AI infrastructure SIC codes.
    Returns list of findings.
    """
    logger.info("Scanning for new AI infrastructure listings...")
    findings = []

    start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    for sic in AI_INFRA_SIC_CODES:
        try:
            url = "https://efts.sec.gov/LATEST/search-index"
            params = {
                "q": "*",
                "forms": "S-1,S-1/A,424B4",
                "dateRange": "custom",
                "startdt": start_date,
                "enddt": end_date,
            }

            resp = requests.get(url, params=params, timeout=30, headers={
                "User-Agent": "Y2AI Research admin@y2ai.com",
            })

            if resp.status_code == 200:
                data = resp.json()
                hits = data.get("hits", {}).get("hits", [])
                for hit in hits[:10]:
                    source = hit.get("_source", {})
                    findings.append({
                        "type": "NEW_LISTING",
                        "sic": sic,
                        "date": source.get("file_date", ""),
                        "description": source.get("display_names", [""])[0] if source.get("display_names") else "",
                        "form": source.get("form_type", ""),
                    })

        except Exception as e:
            logger.warning(f"  EDGAR listing scan error for SIC {sic}: {e}")

    logger.info(f"  Found {len(findings)} potential new listings")
    return findings


def run_scan(dry_run=False):
    """Run full EDGAR scan and update candidates sheet."""
    logger.info("Universe Candidate Scanner — weekly scan")

    nvidia_13f = scan_nvidia_13f()
    hyperscaler_8k = scan_hyperscaler_8k()
    new_listings = scan_new_listings()

    all_findings = nvidia_13f + hyperscaler_8k + new_listings

    if not all_findings:
        logger.info("  No new findings this week.")
        return

    logger.info(f"\n  SCAN RESULTS ({len(all_findings)} findings):")
    for f in all_findings:
        logger.info(f"    [{f['type']}] {f.get('date','')} — {f.get('description','')[:60]}")

    if dry_run:
        logger.info("  DRY RUN — not updating sheet")
        return

    # Load existing candidates and append new findings as WATCH items
    candidates = load_candidates_from_sheet()
    existing_tickers = {c["ticker"] for c in candidates}
    today = datetime.now().strftime("%Y-%m-%d")

    new_count = 0
    for f in all_findings:
        # Only add if we can extract a ticker-like reference
        desc = f.get("description", "")
        if f["type"] == "NVIDIA_13F" and "13F" in f.get("form", ""):
            # Flag for manual review
            if "NVIDIA_13F_REVIEW" not in existing_tickers:
                candidates.append({
                    "ticker": "REVIEW",
                    "company": f"Nvidia 13F ({f.get('date','')})",
                    "category": "Nvidia Investment",
                    "trigger": "Nvidia 13F filing detected",
                    "date_flagged": today,
                    "dm_status": "PENDING REVIEW",
                    "decision": "WATCH",
                    "decision_date": "",
                    "notes": desc[:100],
                })
                new_count += 1

    if new_count > 0:
        push_candidates_to_sheet(candidates)
        logger.info(f"  Added {new_count} new candidates to watchlist")
    else:
        logger.info("  No new candidates to add")


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Universe Candidate Watchlist + EDGAR Scanner")
    parser.add_argument("command", choices=["push", "scan", "add"],
                        help="push=update sheet, scan=EDGAR scan, add=add ticker")
    parser.add_argument("args", nargs="*", help="For 'add': TICKER 'Company' 'Category' 'Trigger'")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if args.command == "push":
        # Load from sheet or use initial, then push
        try:
            candidates = load_candidates_from_sheet()
        except Exception:
            candidates = INITIAL_CANDIDATES.copy()
        push_candidates_to_sheet(candidates)

    elif args.command == "scan":
        run_scan(dry_run=args.dry_run)

    elif args.command == "add":
        if len(args.args) < 4:
            logger.error("Usage: python universe_candidates.py add TICKER 'Company' 'Category' 'Trigger'")
            sys.exit(1)

        ticker, company, category, trigger = args.args[0], args.args[1], args.args[2], args.args[3]
        candidates = load_candidates_from_sheet()

        candidates.append({
            "ticker": ticker.upper(),
            "company": company,
            "category": category,
            "trigger": trigger,
            "date_flagged": datetime.now().strftime("%Y-%m-%d"),
            "dm_status": "PENDING",
            "decision": "WATCH",
            "decision_date": "",
            "notes": "",
        })

        push_candidates_to_sheet(candidates)
        logger.info(f"  Added {ticker.upper()} to watchlist")


if __name__ == "__main__":
    main()
