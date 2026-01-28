"""
Private Capital Google Sheets Writer
=====================================

Writes Private Capital data to Google Sheets Dashboard.

Author: ARGUS-1 Development Team
Created: January 2026
"""

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Configuration
GOOGLE_SHEETS_CREDS_FILE = 'credentials.json'
SPREADSHEET_NAME = 'Copy of Copy of Version 3.3 - Development Copy (Active)'
SPREADSHEET_NAME_2 = 'Vikram-Develop-This'
SHEET_NAME = 'Private_Capital_Dial'


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


def write_to_both_sheets(client, sheet_name: str, row_data: list, headers: list = None):
    """Write row to both spreadsheets, creating tab if needed."""
    import gspread
    
    # Primary sheet
    try:
        spreadsheet1 = client.open(SPREADSHEET_NAME)
        try:
            sheet1 = spreadsheet1.worksheet(sheet_name)
        except gspread.WorksheetNotFound:
            logger.info(f"Creating {sheet_name} tab in primary sheet...")
            sheet1 = spreadsheet1.add_worksheet(title=sheet_name, rows=1000, cols=15)
            if headers:
                sheet1.append_row(headers, value_input_option='USER_ENTERED')
        
        sheet1.append_row(row_data, value_input_option='USER_ENTERED')
        logger.info(f"Wrote to {SPREADSHEET_NAME}")
    except Exception as e:
        logger.error(f"Error writing to primary sheet: {e}")
    
    # Secondary sheet (Vikram-Develop-This)
    try:
        spreadsheet2 = client.open(SPREADSHEET_NAME_2)
        try:
            sheet2 = spreadsheet2.worksheet(sheet_name)
        except gspread.WorksheetNotFound:
            logger.info(f"Creating {sheet_name} tab in secondary sheet...")
            sheet2 = spreadsheet2.add_worksheet(title=sheet_name, rows=1000, cols=15)
            if headers:
                sheet2.append_row(headers, value_input_option='USER_ENTERED')
        
        sheet2.append_row(row_data, value_input_option='USER_ENTERED')
        logger.info(f"Wrote to {SPREADSHEET_NAME_2}")
    except Exception as e:
        logger.warning(f"Error writing to secondary sheet: {e}")


def write_private_capital_to_sheets(result: dict):
    """
    Write Private Capital result to Google Sheets.
    
    Args:
        result: Output from update_private_capital() containing intensity data
    """
    try:
        logger.info("Writing Private Capital to Google Sheets...")
        client = get_sheets_client()
        
        if not client:
            logger.error("Could not get Google Sheets client")
            return False
        
        # Extract data
        intensity = result['intensity']
        interpretation = result.get('interpretation', {})
        
        # Define headers
        headers = [
            'Date', 'Score', 'Regime', '30D_Volume_M', '90D_Volume_M',
            'Megarounds', 'Fund_Launches', 'Categories', 'Detail',
            'Bubble_Index', 'Signal', 'Infra_Bias'
        ]
        
        # Prepare row
        row = [
            datetime.utcnow().strftime('%Y-%m-%d'),
            intensity.score,
            intensity.regime,
            intensity.vol_30d_m,
            intensity.vol_90d_m,
            intensity.megarounds_30d,
            intensity.fund_launches_30d,
            ', '.join(intensity.categories) if intensity.categories else '',
            intensity.detail,
            interpretation.get('bubble_index', ''),
            interpretation.get('signal', ''),
            interpretation.get('infra_bias', '')
        ]
        
        # Write to both sheets (auto-creates tab if needed)
        write_to_both_sheets(client, SHEET_NAME, row, headers=headers)
        
        logger.info(f"✅ Private Capital written to Google Sheets: {intensity.score} ({intensity.regime})")
        return True
        
    except Exception as e:
        logger.error(f"Error writing to Google Sheets: {e}")
        return False


def update_dashboard_cell(client, cell_name: str, value):
    """
    Update a named cell on the Dashboard sheet.
    
    Args:
        client: gspread client
        cell_name: Name of the cell/label to find
        value: Value to write
    """
    try:
        spreadsheet = client.open(SPREADSHEET_NAME)
        dashboard = spreadsheet.worksheet('Dashboard')
        
        # Find the cell by label and update the adjacent cell
        cell = dashboard.find(cell_name)
        if cell:
            # Write to the cell to the right of the label
            dashboard.update_cell(cell.row, cell.col + 1, value)
            logger.info(f"Updated Dashboard {cell_name}: {value}")
            return True
    except Exception as e:
        logger.warning(f"Could not update Dashboard cell {cell_name}: {e}")
    
    return False


def update_dashboard_private_capital(result: dict):
    """
    Update Dashboard sheet with Private Capital summary values.
    
    Updates these cells (if they exist):
    - Private_Capital: score
    - Private_Capital_Regime: regime
    - PC_Signal: cross-reference signal
    """
    try:
        client = get_sheets_client()
        if not client:
            return False
        
        intensity = result['intensity']
        interpretation = result.get('interpretation', {})
        
        # Update dashboard cells
        update_dashboard_cell(client, 'Private_Capital', intensity.score)
        update_dashboard_cell(client, 'Private_Capital_Regime', intensity.regime)
        update_dashboard_cell(client, 'PC_30D_Volume', f"${intensity.vol_30d_m/1000:.1f}B")
        update_dashboard_cell(client, 'PC_Megarounds', intensity.megarounds_30d)
        
        if interpretation:
            update_dashboard_cell(client, 'PC_Signal', interpretation.get('signal', ''))
            update_dashboard_cell(client, 'PC_Infra_Bias', interpretation.get('infra_bias', ''))
        
        logger.info("✅ Dashboard updated with Private Capital data")
        return True
        
    except Exception as e:
        logger.error(f"Error updating Dashboard: {e}")
        return False


if __name__ == '__main__':
    """Test Google Sheets writing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # Create mock result for testing
    from private_capital import IntensityResult
    
    mock_result = {
        'intensity': IntensityResult(
            score=80,
            regime='STRONG',
            vol_30d_m=76000,
            vol_90d_m=76000,
            megarounds_30d=3,
            fund_launches_30d=0,
            categories=['foundation_model', 'energy', 'application'],
            detail='30D: $76.0B, Megarounds: 3, Categories: 3'
        ),
        'interpretation': {
            'bubble_index': 44,
            'signal': 'EARLY_CYCLE',
            'infra_bias': 0.5
        }
    }
    
    print("\n=== Testing Google Sheets Writer ===\n")
    write_private_capital_to_sheets(mock_result)