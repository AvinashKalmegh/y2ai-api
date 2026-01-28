"""
Sync NCI, NPD, Burst, EVI data from source sheet to Vikram-Develop-This
and update Python files to write to both sheets.
"""

import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

# Config
SOURCE_SHEET = 'Copy of Copy of Version 3.3 - Development Copy (Active)'
TARGET_SHEET = 'Vikram-Develop-This'
CREDS_FILE = 'credentials.json'

SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

def get_client():
    """Get authenticated gspread client."""
    creds = Credentials.from_service_account_file(CREDS_FILE, scopes=SCOPES)
    return gspread.authorize(creds)

def sync_dial_data(client, dial_name: str, start_date: str = '2026-01-07'):
    """
    Copy data from source to target sheet for a specific dial.
    
    Args:
        client: gspread client
        dial_name: NCI_Dial, NPD_Dial, Burst_Dial, or EVI_Dial
        start_date: Copy data after this date
    """
    print(f"\n=== Syncing {dial_name} ===")
    
    # Open both sheets
    source = client.open(SOURCE_SHEET).worksheet(dial_name)
    target = client.open(TARGET_SHEET).worksheet(dial_name)
    
    # Get all data from source
    source_data = source.get_all_values()
    headers = source_data[0]
    
    # Find date column (usually column A)
    date_col = 0
    
    # Filter rows after start_date
    new_rows = []
    for row in source_data[1:]:
        if row[date_col]:
            try:
                # Handle various date formats
                date_str = row[date_col]
                if isinstance(date_str, str) and len(date_str) >= 10:
                    row_date = date_str[:10]
                    if row_date >= start_date:
                        new_rows.append(row)
            except:
                continue
    
    if not new_rows:
        print(f"  No new rows found after {start_date}")
        return 0
    
    print(f"  Found {len(new_rows)} rows to sync")
    
    # Get existing data from target to find where to append
    target_data = target.get_all_values()
    next_row = len(target_data) + 1
    
    # Append new rows
    if new_rows:
        # Use batch update for efficiency
        target.append_rows(new_rows, value_input_option='USER_ENTERED')
        print(f"  Appended {len(new_rows)} rows to {dial_name}")
    
    return len(new_rows)

def update_python_files():
    """Update Python files to write to both sheets."""
    
    files = [
        'nci_narrative_coherence.py',
        'npd_narrative_polarity_drift.py', 
        'burst_keyword_detection.py',
        'evi_event_volatility_index.py'
    ]
    
    old_line = "SPREADSHEET_NAME = 'Copy of Copy of Version 3.3 - Development Copy (Active)'"
    
    # New config with both sheets
    new_config = '''SPREADSHEET_NAME = 'Copy of Copy of Version 3.3 - Development Copy (Active)'
SPREADSHEET_NAME_2 = 'Vikram-Develop-This'  # Secondary sheet for backup'''
    
    for filename in files:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if 'SPREADSHEET_NAME_2' not in content:
                content = content.replace(old_line, new_config)
                
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"Updated {filename} with dual sheet config")
            else:
                print(f"{filename} already has dual sheet config")
                
        except Exception as e:
            print(f"Error updating {filename}: {e}")

def add_dual_write_function():
    """
    Print the function that needs to be added to each file
    for writing to both sheets.
    """
    print("""
=== ADD THIS HELPER FUNCTION TO EACH FILE ===

def write_to_sheets(data_row, sheet_name):
    \"\"\"Write data to both spreadsheets.\"\"\"
    try:
        gc = get_google_sheets_client()
        
        # Write to primary sheet
        sheet1 = gc.open(SPREADSHEET_NAME).worksheet(sheet_name)
        sheet1.append_row(data_row, value_input_option='USER_ENTERED')
        
        # Write to secondary sheet
        try:
            sheet2 = gc.open(SPREADSHEET_NAME_2).worksheet(sheet_name)
            sheet2.append_row(data_row, value_input_option='USER_ENTERED')
        except Exception as e:
            logger.warning(f"Failed to write to secondary sheet: {e}")
            
    except Exception as e:
        logger.error(f"Failed to write to sheets: {e}")

=== Then replace existing sheet.append_row() calls with write_to_sheets() ===
""")

def main():
    print("=" * 50)
    print("NCI/NPD/Burst/EVI Sheet Sync Tool")
    print("=" * 50)
    
    # Step 1: Sync data
    print("\n[Step 1] Syncing data from source to Vikram-Develop-This...")
    
    try:
        client = get_client()
        
        dials = ['NCI_Dial', 'NPD_Dial', 'Burst_Dial', 'EVI_Dial']
        total_synced = 0
        
        for dial in dials:
            try:
                count = sync_dial_data(client, dial, start_date='2026-01-07')
                total_synced += count
            except Exception as e:
                print(f"  Error syncing {dial}: {e}")
        
        print(f"\nTotal rows synced: {total_synced}")
        
    except Exception as e:
        print(f"Error connecting to Google Sheets: {e}")
        print("Make sure credentials.json exists and has access to both sheets")
    
    # Step 2: Update Python files
    print("\n[Step 2] Updating Python files for dual-sheet writing...")
    update_python_files()
    
    # Step 3: Show manual steps needed
    add_dual_write_function()

if __name__ == "__main__":
    main()