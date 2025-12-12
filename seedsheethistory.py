"""
seed_metrics_history.py

Export your Trends_History sheet from Google Sheets as CSV, then run this script
to bulk upload it to your Railway API.

Usage:
    1. In Google Sheets, go to Trends_History tab
    2. File > Download > Comma Separated Values (.csv)
    3. Save as trends_history.csv
    4. Run: python seed_metrics_history.py trends_history.csv

The CSV should have headers matching your Trends_History columns:
Date, AMRI, AMRI_Regime, Enhanced_AMRI, Break_Prob, MCI, MCI_Regime, 
Bubble_Index, Clusters, Corr_20D, Breadth_20D, VIX, Credit_Spreads, 
Infra_Breadth, Ent_Breadth, Thesis_Balance
"""

import csv
import json
import sys
import requests
from datetime import datetime

# Your Railway API endpoint
API_BASE = "https://y2ai-api-production.up.railway.app"

def parse_date(date_str):
    """Parse various date formats from Google Sheets"""
    if not date_str or date_str.strip() == '':
        return None
    
    formats = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m/%d/%y",
        "%d/%m/%Y",
        "%Y/%m/%d",
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    
    print(f"Warning: Could not parse date '{date_str}'")
    return None

def parse_float(val):
    """Safely parse float values"""
    if val is None or val == '' or val == 'N/A':
        return None
    try:
        # Handle percentage strings like "41.9%"
        if isinstance(val, str) and val.endswith('%'):
            return float(val.rstrip('%')) / 100
        return float(val)
    except ValueError:
        return None

def parse_int(val):
    """Safely parse integer values"""
    if val is None or val == '' or val == 'N/A':
        return None
    try:
        return int(float(val))
    except ValueError:
        return None

def load_csv(filepath):
    """Load and parse CSV file"""
    rows = []
    
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        
        # Print detected columns for debugging
        print(f"Detected columns: {reader.fieldnames}")
        
        for row in reader:
            # Map column names (handle variations)
            date = parse_date(
                row.get('Date') or row.get('date') or row.get('DATE')
            )
            
            if not date:
                print(f"Skipping row with invalid date: {row}")
                continue
            
            parsed = {
                "date": date,
                "amri": parse_float(row.get('AMRI') or row.get('amri')),
                "amri_regime": row.get('AMRI_Regime') or row.get('amri_regime') or row.get('AMRI Regime'),
                "enhanced_amri": parse_float(row.get('Enhanced_AMRI') or row.get('enhanced_amri')),
                "break_prob": parse_float(row.get('Break_Prob') or row.get('break_prob') or row.get('Break Prob')),
                "mci": parse_float(row.get('MCI') or row.get('mci')),
                "mci_regime": row.get('MCI_Regime') or row.get('mci_regime') or row.get('MCI Regime'),
                "bubble_index": parse_float(row.get('Bubble_Index') or row.get('bubble_index') or row.get('Bubble Index')),
                "clusters": parse_int(row.get('Clusters') or row.get('clusters')),
                "corr_20d": parse_float(row.get('Corr_20D') or row.get('corr_20d') or row.get('Corr 20D')),
                "breadth_20d": parse_float(row.get('Breadth_20D') or row.get('breadth_20d') or row.get('Breadth 20D')),
                "vix": parse_float(row.get('VIX') or row.get('vix')),
                "credit_spreads": parse_float(row.get('Credit_Spreads') or row.get('credit_spreads') or row.get('Credit Spreads')),
                "infra_breadth": parse_float(row.get('Infra_Breadth') or row.get('infra_breadth') or row.get('Infra Breadth')),
                "ent_breadth": parse_float(row.get('Ent_Breadth') or row.get('ent_breadth') or row.get('Ent Breadth')),
                "thesis_balance": parse_float(row.get('Thesis_Balance') or row.get('thesis_balance') or row.get('Thesis Balance')),
            }
            
            rows.append(parsed)
    
    return rows

def upload_to_api(rows):
    """Upload rows to the API bulk endpoint"""
    url = f"{API_BASE}/metrics-history/bulk"
    
    print(f"\nUploading {len(rows)} rows to {url}...")
    
    try:
        response = requests.post(url, json=rows, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success! {result.get('rows_processed', len(rows))} rows processed.")
            return True
        else:
            print(f"Error {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python seed_metrics_history.py <csv_file>")
        print("\nExample:")
        print("  python seed_metrics_history.py trends_history.csv")
        sys.exit(1)
    
    filepath = sys.argv[1]
    print(f"Loading CSV from {filepath}...")
    
    rows = load_csv(filepath)
    print(f"Parsed {len(rows)} valid rows")
    
    if rows:
        # Show sample of what we're uploading
        print("\nSample row:")
        print(json.dumps(rows[0], indent=2))
        
        # Confirm before uploading
        confirm = input("\nProceed with upload? (y/n): ")
        if confirm.lower() == 'y':
            upload_to_api(rows)
        else:
            print("Upload cancelled.")
    else:
        print("No valid rows found in CSV.")

if __name__ == "__main__":
    main()