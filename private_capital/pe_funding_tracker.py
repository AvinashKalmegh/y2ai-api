"""
PE Funding Tracker
==================

Generic PE/VC funding tracker - separate from Private Capital (AI-specific).
Sectors emerge from data, not predefined.

Writes to:
- Supabase: pe_funding_entries, pe_funding_daily, pe_sector_daily
- Google Sheets: PE_Funding_Dial tab, PE_Funding_Entries tab
  - Copy of Copy of Version 3.3 - Development Copy (Active)
  - Vikram-Develop-This
  - NEW_FlowOS

Author: ARGUS-1 Development Team
Created: January 2026
"""

import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

GOOGLE_SHEETS_CREDS_FILE = 'credentials.json'
SPREADSHEET_NAME = 'Copy of Copy of Version 3.3 - Development Copy (Active)'
SPREADSHEET_NAME_2 = 'Vikram-Develop-This'
SPREADSHEET_NAME_3 = 'NEW_FlowOS'
PE_SHEET_NAME = 'PE_Funding_Dial'
PE_ENTRIES_SHEET = 'PE_Funding_Entries'

# Headers for Google Sheets
PE_DIAL_HEADERS = [
    'Date', 'Deals', 'Volume_M', 'Mega_Rounds', 'Series_A', 'Series_B',
    'Late_Stage', 'IPO_Pipeline', 'Fund_Launches', 'Vol_7D_M', 'Vol_30D_M',
    'Top_Sectors', 'Sector_Volumes'
]

PE_ENTRIES_HEADERS = [
    'Date', 'Company', 'Amount_M', 'Stage', 'Sectors', 'Keywords',
    'Investors', 'Source', 'URL'
]


# =============================================================================
# GOOGLE SHEETS FUNCTIONS
# =============================================================================

def get_sheets_client():
    """Get authenticated Google Sheets client."""
    try:
        import gspread
        from oauth2client.service_account import ServiceAccountCredentials
    except ImportError:
        logger.error("gspread not installed. Run: pip install gspread oauth2client")
        return None
    
    scope = [
        'https://spreadsheets.google.com/feeds',
        'https://www.googleapis.com/auth/drive'
    ]
    
    try:
        creds = ServiceAccountCredentials.from_json_keyfile_name(
            GOOGLE_SHEETS_CREDS_FILE, scope
        )
        return gspread.authorize(creds)
    except Exception as e:
        logger.error(f"Failed to authenticate with Google Sheets: {e}")
        return None


def write_to_sheet(client, spreadsheet_name: str, sheet_name: str, 
                   row_data: list, headers: list = None):
    """Write row to a sheet, creating tab if needed."""
    import gspread
    
    try:
        spreadsheet = client.open(spreadsheet_name)
        try:
            sheet = spreadsheet.worksheet(sheet_name)
        except gspread.WorksheetNotFound:
            logger.info(f"Creating {sheet_name} tab in {spreadsheet_name}...")
            sheet = spreadsheet.add_worksheet(title=sheet_name, rows=1000, cols=20)
            if headers:
                sheet.append_row(headers, value_input_option='USER_ENTERED')
        
        sheet.append_row(row_data, value_input_option='USER_ENTERED')
        logger.info(f"Wrote to {spreadsheet_name}/{sheet_name}")
        return True
    except Exception as e:
        logger.error(f"Error writing to {spreadsheet_name}/{sheet_name}: {e}")
        return False


def write_batch_to_sheet(client, spreadsheet_name: str, sheet_name: str, 
                         rows: list, headers: list = None):
    """Write multiple rows to a sheet at once (batch write)."""
    import gspread
    import time
    
    if not rows:
        return True
    
    try:
        spreadsheet = client.open(spreadsheet_name)
        try:
            sheet = spreadsheet.worksheet(sheet_name)
        except gspread.WorksheetNotFound:
            logger.info(f"Creating {sheet_name} tab in {spreadsheet_name}...")
            sheet = spreadsheet.add_worksheet(title=sheet_name, rows=2000, cols=20)
            if headers:
                sheet.append_row(headers, value_input_option='USER_ENTERED')
                time.sleep(1)  # Brief pause after creating sheet
        
        # Batch write all rows at once
        sheet.append_rows(rows, value_input_option='USER_ENTERED')
        logger.info(f"Batch wrote {len(rows)} rows to {spreadsheet_name}/{sheet_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error batch writing to {spreadsheet_name}/{sheet_name}: {e}")
        return False


def write_to_both_sheets(client, sheet_name: str, row_data: list, headers: list = None):
    """Write row to all three spreadsheets."""
    write_to_sheet(client, SPREADSHEET_NAME, sheet_name, row_data, headers)
    write_to_sheet(client, SPREADSHEET_NAME_2, sheet_name, row_data, headers)
    write_to_sheet(client, SPREADSHEET_NAME_3, sheet_name, row_data, headers)


def write_batch_to_both_sheets(client, sheet_name: str, rows: list, headers: list = None):
    """Batch write rows to all three spreadsheets."""
    import time
    
    write_batch_to_sheet(client, SPREADSHEET_NAME, sheet_name, rows, headers)
    time.sleep(2)  # Pause between sheets to avoid rate limits
    write_batch_to_sheet(client, SPREADSHEET_NAME_2, sheet_name, rows, headers)
    time.sleep(2)
    write_batch_to_sheet(client, SPREADSHEET_NAME_3, sheet_name, rows, headers)


# =============================================================================
# PE FUNDING TRACKER
# =============================================================================

class PEFundingTracker:
    """
    Tracks generic PE/VC funding from Google Alerts.
    Writes to Supabase and Google Sheets.
    """
    
    def __init__(self, supabase_client=None):
        self.supabase = supabase_client
        self.sheets_client = None
    
    def _get_sheets_client(self):
        """Lazy load sheets client."""
        if not self.sheets_client:
            self.sheets_client = get_sheets_client()
        return self.sheets_client
    
    # -------------------------------------------------------------------------
    # SUPABASE OPERATIONS
    # -------------------------------------------------------------------------
    
    def store_entries_supabase(self, entries: List[Dict[str, Any]]) -> int:
        """Store parsed funding entries to pe_funding_entries table."""
        if not self.supabase or not entries:
            return 0
        
        stored = 0
        for entry in entries:
            try:
                row = {
                    'date': entry.get('date'),
                    'company': entry.get('company', 'Unknown'),
                    'amount_m': entry.get('amount_m'),
                    'stage': entry.get('stage'),
                    'raw_text': entry.get('raw_text'),
                    'source': entry.get('source'),
                    'url': entry.get('url'),
                    'sectors': entry.get('sectors', []),
                    'keywords': entry.get('keywords', []),
                    'investors': entry.get('investors', []),
                }
                
                self.supabase.table('pe_funding_entries').upsert(
                    row,
                    on_conflict='date,company,source'
                ).execute()
                stored += 1
                
            except Exception as e:
                logger.error(f"Error storing entry {entry.get('company')}: {e}")
        
        logger.info(f"Supabase: Stored {stored}/{len(entries)} PE funding entries")
        return stored
    
    def calculate_daily_supabase(self, date: str = None) -> Dict[str, Any]:
        """Calculate daily aggregated metrics and store to pe_funding_daily."""
        if not self.supabase:
            return {}
        
        date = date or datetime.utcnow().strftime('%Y-%m-%d')
        
        # Fetch today's entries
        response = self.supabase.table('pe_funding_entries')\
            .select('*')\
            .eq('date', date)\
            .execute()
        
        entries = response.data or []
        
        # Calculate metrics
        metrics = {
            'date': date,
            'deal_count': len(entries),
            'total_volume_m': sum(e.get('amount_m') or 0 for e in entries),
            'mega_rounds': sum(1 for e in entries if e.get('stage') == 'Mega_Round'),
            'series_a': sum(1 for e in entries if e.get('stage') == 'Series_A'),
            'series_b': sum(1 for e in entries if e.get('stage') == 'Series_B'),
            'late_stage': sum(1 for e in entries if e.get('stage') in 
                            ['Series_C', 'Series_D', 'Series_E+', 'Late_Stage', 'Growth']),
            'ipo_pipeline': sum(1 for e in entries if e.get('stage') == 'IPO'),
            'fund_launches': sum(1 for e in entries if e.get('stage') == 'Fund_Launch'),
        }
        
        # Sector breakdown
        sector_volumes = defaultdict(float)
        for entry in entries:
            amount = entry.get('amount_m') or 0
            for sector in (entry.get('sectors') or ['general']):
                sector_volumes[sector] += amount
        
        metrics['sector_volumes'] = dict(sector_volumes)
        metrics['top_sectors'] = sorted(
            sector_volumes.keys(), 
            key=lambda s: sector_volumes[s], 
            reverse=True
        )[:5]
        
        # Rolling metrics
        metrics['vol_7d_m'], metrics['deals_7d'] = self._get_rolling_metrics(date, 7)
        metrics['vol_30d_m'], metrics['deals_30d'] = self._get_rolling_metrics(date, 30)
        
        # Store to Supabase
        try:
            self.supabase.table('pe_funding_daily').upsert(
                metrics,
                on_conflict='date'
            ).execute()
            logger.info(f"Supabase: Stored PE daily metrics for {date}")
        except Exception as e:
            logger.error(f"Error storing PE daily to Supabase: {e}")
        
        return metrics
    
    def calculate_sector_daily_supabase(self, date: str = None) -> List[Dict]:
        """Calculate per-sector daily metrics and store to pe_sector_daily."""
        if not self.supabase:
            return []
        
        date = date or datetime.utcnow().strftime('%Y-%m-%d')
        
        # Fetch today's entries
        response = self.supabase.table('pe_funding_entries')\
            .select('*')\
            .eq('date', date)\
            .execute()
        
        entries = response.data or []
        
        # Aggregate by sector
        sector_data = defaultdict(lambda: {
            'deal_count': 0,
            'total_volume_m': 0,
            'companies': [],
            'stages': defaultdict(int)
        })
        
        for entry in entries:
            for sector in (entry.get('sectors') or ['general']):
                sector_data[sector]['deal_count'] += 1
                sector_data[sector]['total_volume_m'] += entry.get('amount_m') or 0
                sector_data[sector]['companies'].append(entry.get('company'))
                stage = entry.get('stage') or 'Unknown'
                sector_data[sector]['stages'][stage] += 1
        
        # Store each sector
        results = []
        for sector, data in sector_data.items():
            row = {
                'date': date,
                'sector': sector,
                'deal_count': data['deal_count'],
                'total_volume_m': data['total_volume_m'],
                'avg_deal_m': data['total_volume_m'] / max(data['deal_count'], 1),
                'top_companies': list(set(data['companies']))[:10],
                'stages': dict(data['stages']),
            }
            
            try:
                self.supabase.table('pe_sector_daily').upsert(
                    row,
                    on_conflict='date,sector'
                ).execute()
                results.append(row)
            except Exception as e:
                logger.error(f"Error storing sector {sector} to Supabase: {e}")
        
        logger.info(f"Supabase: Stored {len(results)} sector records for {date}")
        return results
    
    def _get_rolling_metrics(self, date: str, days: int) -> tuple:
        """Get rolling volume and deal count."""
        try:
            cutoff = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=days)).strftime('%Y-%m-%d')
            
            response = self.supabase.table('pe_funding_entries')\
                .select('amount_m')\
                .gte('date', cutoff)\
                .lte('date', date)\
                .execute()
            
            entries = response.data or []
            volume = sum(e.get('amount_m') or 0 for e in entries)
            return volume, len(entries)
            
        except Exception as e:
            logger.error(f"Error getting rolling metrics: {e}")
            return 0, 0
    
    # -------------------------------------------------------------------------
    # GOOGLE SHEETS OPERATIONS
    # -------------------------------------------------------------------------
    
    def write_entries_sheets(self, entries: List[Dict[str, Any]]) -> int:
        """Write funding entries to PE_Funding_Entries sheet (batch write)."""
        client = self._get_sheets_client()
        if not client or not entries:
            return 0
        
        # Build all rows first
        rows = []
        for entry in entries:
            row = [
                entry.get('date', ''),
                entry.get('company', ''),
                entry.get('amount_m', ''),
                entry.get('stage', ''),
                ', '.join(entry.get('sectors', [])),
                ', '.join(entry.get('keywords', [])),
                ', '.join(entry.get('investors', [])),
                entry.get('source', ''),
                entry.get('url', ''),
            ]
            rows.append(row)
        
        # Batch write all rows at once
        try:
            write_batch_to_both_sheets(client, PE_ENTRIES_SHEET, rows, PE_ENTRIES_HEADERS)
            logger.info(f"Google Sheets: Batch wrote {len(rows)} entries")
            return len(rows)
        except Exception as e:
            logger.error(f"Error batch writing entries to sheets: {e}")
            return 0
    
    def write_daily_sheets(self, metrics: Dict[str, Any]) -> bool:
        """Write daily metrics to PE_Funding_Dial sheet."""
        client = self._get_sheets_client()
        if not client or not metrics:
            return False
        
        # Prepare row
        row = [
            metrics.get('date', ''),
            metrics.get('deal_count', 0),
            metrics.get('total_volume_m', 0),
            metrics.get('mega_rounds', 0),
            metrics.get('series_a', 0),
            metrics.get('series_b', 0),
            metrics.get('late_stage', 0),
            metrics.get('ipo_pipeline', 0),
            metrics.get('fund_launches', 0),
            metrics.get('vol_7d_m', 0),
            metrics.get('vol_30d_m', 0),
            ', '.join(metrics.get('top_sectors', [])),
            json.dumps(metrics.get('sector_volumes', {})),
        ]
        
        try:
            write_to_both_sheets(client, PE_SHEET_NAME, row, PE_DIAL_HEADERS)
            logger.info(f"Google Sheets: Wrote daily metrics for {metrics.get('date')}")
            return True
        except Exception as e:
            logger.error(f"Error writing daily to sheets: {e}")
            return False
    
    # -------------------------------------------------------------------------
    # MAIN UPDATE FUNCTION
    # -------------------------------------------------------------------------
    
    def run_update(self, entries: List[Dict] = None, write_sheets: bool = True) -> Dict:
        """
        Run full update cycle:
        1. Store new entries to Supabase
        2. Write entries to Google Sheets
        3. Calculate and store daily metrics (Supabase)
        4. Write daily metrics to Google Sheets
        5. Calculate and store sector metrics (Supabase)
        
        Args:
            entries: List of parsed funding entry dicts
            write_sheets: Whether to write to Google Sheets (default True)
        
        Returns:
            Summary dict
        """
        date = datetime.utcnow().strftime('%Y-%m-%d')
        
        result = {
            'date': date,
            'entries_stored_supabase': 0,
            'entries_written_sheets': 0,
            'daily_metrics': {},
            'sectors_updated': 0,
            'sheets_daily_written': False,
        }
        
        # 1. Store entries to Supabase
        if entries:
            result['entries_stored_supabase'] = self.store_entries_supabase(entries)
        
        # 2. Write entries to Google Sheets
        if write_sheets and entries:
            result['entries_written_sheets'] = self.write_entries_sheets(entries)
        
        # 3. Calculate daily metrics (Supabase)
        result['daily_metrics'] = self.calculate_daily_supabase(date)
        
        # 4. Write daily to Google Sheets
        if write_sheets and result['daily_metrics']:
            result['sheets_daily_written'] = self.write_daily_sheets(result['daily_metrics'])
        
        # 5. Calculate sector metrics (Supabase)
        sector_results = self.calculate_sector_daily_supabase(date)
        result['sectors_updated'] = len(sector_results)
        
        logger.info(f"PE Funding update complete: "
                    f"{result['entries_stored_supabase']} entries (Supabase), "
                    f"{result['entries_written_sheets']} entries (Sheets), "
                    f"{result['sectors_updated']} sectors")
        
        return result


# =============================================================================
# WORKFLOW INTEGRATION
# =============================================================================

def run_pe_funding_update(supabase_client=None, hours_back: int = 24, write_sheets: bool = True):
    """
    Main entry point for PE funding update.
    Called by orchestrator.
    
    Args:
        supabase_client: Supabase client instance
        hours_back: How far back to fetch alerts (default 24 hours)
        write_sheets: Whether to write to Google Sheets
    
    Returns:
        Result dict with counts and metrics
    """
    # Import PE feeds from google_alerts_adapter
    from .google_alerts_adapter import PE_GOOGLE_ALERT_FEEDS, GoogleAlertsAdapter
    
    # Create adapter configured for PE feeds
    adapter = GoogleAlertsAdapter()
    adapter.feeds = PE_GOOGLE_ALERT_FEEDS  # Use PE feeds instead of AI feeds
    
    # Collect and parse alerts
    logger.info(f"Fetching PE alerts from last {hours_back} hours...")
    alerts = adapter.fetch_alerts(hours_back=hours_back)
    
    if not alerts:
        logger.warning("No PE funding entries found from alerts")
        return {'status': 'no_data', 'entries': 0}
    
    logger.info(f"Found {len(alerts)} raw alert entries")
    
    # Parse alerts into funding entries
    entries = []
    for alert in alerts:
        from .pe_funding_parser import parse_pe_alert
        entry = parse_pe_alert(alert)
        if entry:
            entries.append(entry)
    
    if not entries:
        logger.warning("No valid funding entries after parsing")
        return {'status': 'no_data', 'entries': 0}
    
    logger.info(f"Parsed {len(entries)} funding entries")
    
    # Store and calculate
    tracker = PEFundingTracker(supabase_client=supabase_client)
    result = tracker.run_update(entries=entries, write_sheets=write_sheets)
    
    result['status'] = 'success'
    return result


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse
    import os
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='PE Funding Tracker')
    parser.add_argument('--update', action='store_true', help='Run full update (Supabase + Sheets)')
    parser.add_argument('--supabase-only', action='store_true', help='Update Supabase only')
    parser.add_argument('--daily', action='store_true', help='Calculate daily metrics only')
    parser.add_argument('--hours', type=int, default=24, help='Hours back to fetch')
    parser.add_argument('--test-sheets', action='store_true', help='Test Google Sheets connection')
    
    args = parser.parse_args()
    
    # Initialize Supabase
    supabase_url = os.environ.get('SUPABASE_URL')
    supabase_key = os.environ.get('SUPABASE_KEY')
    
    client = None
    if supabase_url and supabase_key:
        from supabase import create_client
        client = create_client(supabase_url, supabase_key)
    else:
        logger.warning("SUPABASE_URL/KEY not set - Supabase operations will be skipped")
    
    if args.update:
        result = run_pe_funding_update(
            supabase_client=client, 
            hours_back=args.hours,
            write_sheets=True
        )
        print(f"\n=== PE Funding Update Complete ===")
        print(f"  Status: {result.get('status')}")
        print(f"  Entries (Supabase): {result.get('entries_stored_supabase', 0)}")
        print(f"  Entries (Sheets): {result.get('entries_written_sheets', 0)}")
        print(f"  Sectors: {result.get('sectors_updated', 0)}")
        if result.get('daily_metrics'):
            dm = result['daily_metrics']
            print(f"  30D Volume: ${dm.get('vol_30d_m', 0):.1f}M")
            print(f"  Top Sectors: {', '.join(dm.get('top_sectors', []))}")
    
    elif args.supabase_only:
        result = run_pe_funding_update(
            supabase_client=client,
            hours_back=args.hours,
            write_sheets=False
        )
        print(f"\nSupabase update complete: {result.get('entries_stored_supabase', 0)} entries")
    
    elif args.daily:
        tracker = PEFundingTracker(supabase_client=client)
        metrics = tracker.calculate_daily_supabase()
        print(f"\nDaily metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    
    elif args.test_sheets:
        print("\nTesting Google Sheets connection...")
        client = get_sheets_client()
        if client:
            print("✓ Google Sheets connection successful")
            try:
                ss = client.open(SPREADSHEET_NAME)
                print(f"✓ Found spreadsheet: {SPREADSHEET_NAME}")
            except Exception as e:
                print(f"✗ Could not open spreadsheet: {e}")
        else:
            print("✗ Google Sheets connection failed")
    
    else:
        parser.print_help()