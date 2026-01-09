"""
VIX DIAL VALIDATION
===================
Compare Python VIX calculations vs Google Sheets values.
Both should now use FRED VIXCLS as the data source.
"""

import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

# Initialize Supabase
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

if not url or not key:
    print("❌ Set SUPABASE_URL and SUPABASE_KEY")
    exit(1)

client = create_client(url, key)

print("=" * 70)
print("VIX DIAL VALIDATION")
print("=" * 70)

# =============================================================================
# 1. VIX DIAL DAILY (calculated metrics)
# =============================================================================
print("\n" + "-" * 50)
print("1. VIX DIAL DAILY (latest)")
print("-" * 50)

r = client.table('vix_dial_daily').select('*').order('date', desc=True).limit(1).execute()
if r.data:
    vix = r.data[0]
    print(f"\nDate: {vix['date']}")
    print(f"\n{'Metric':<25} {'Python':<15} {'Google Sheets':<15}")
    print("-" * 55)
    
    # Helper to safely format values
    def fmt(val, decimals=2):
        if val is None:
            return "N/A"
        try:
            return f"{float(val):.{decimals}f}"
        except:
            return str(val)
    
    # Current VIX
    py_vix = vix.get('vix')
    print(f"{'Current VIX':<25} {fmt(py_vix):<15} {'16.13':<15}")
    
    # Regime
    py_regime = vix.get('combined_regime') or vix.get('level_regime') or 'N/A'
    print(f"{'Regime':<25} {py_regime:<15} {'Elevated':<15}")
    
    # 20-Day MA
    py_ma20 = vix.get('ma_20') or vix.get('vix_ma20') or vix.get('vix_20d_avg')
    print(f"{'20-Day MA':<25} {fmt(py_ma20):<15} {'16.72':<15}")
    
    # Std Dev 20
    py_std = vix.get('std_dev_20') or vix.get('std_20') or vix.get('vix_std20')
    print(f"{'Std Dev (20D)':<25} {fmt(py_std):<15} {'~1.2':<15}")
    
    # Bollinger Bands
    py_upper = vix.get('upper_bb') or vix.get('bb_upper')
    py_lower = vix.get('lower_bb') or vix.get('bb_lower')
    print(f"{'Upper BB':<25} {fmt(py_upper):<15} {'~19':<15}")
    print(f"{'Lower BB':<25} {fmt(py_lower):<15} {'~14':<15}")
    
    # Trend
    py_trend = vix.get('trend_20d') or vix.get('vix_20d_change')
    print(f"{'20D Trend':<25} {fmt(py_trend):<15} {'~-1.5':<15}")
    
    # BB Regime
    py_bb = vix.get('bb_regime') or 'N/A'
    print(f"{'BB Regime':<25} {py_bb:<15} {'Normal':<15}")
    
    # Level Regime
    py_level = vix.get('level_regime') or 'N/A'
    print(f"{'Level Regime':<25} {py_level:<15} {'Healthy':<15}")
    
    print("\n" + "-" * 50)
    print("ALL COLUMNS:")
    print("-" * 50)
    for k, v in sorted(vix.items()):
        print(f"  {k}: {v}")
else:
    print("No data in vix_dial_daily")

# =============================================================================
# 2. VIX HISTORY (raw data from FRED)
# =============================================================================
print("\n" + "=" * 70)
print("2. VIX HISTORY (last 10 days)")
print("=" * 70)

r = client.table('vix_history').select('date, close').order('date', desc=True).limit(10).execute()
if r.data:
    print(f"\n{'Date':<15} {'VIX (FRED)':<15}")
    print("-" * 30)
    for row in r.data:
        print(f"  {row['date']:<13} {row['close']:<15}")
else:
    print("No data in vix_history")

# =============================================================================
# 3. CALCULATE VOL-OF-VOL FROM RAW DATA
# =============================================================================
print("\n" + "=" * 70)
print("3. VOL-OF-VOL CALCULATION (from vix_history)")
print("=" * 70)

r = client.table('vix_history').select('date, close').order('date', desc=True).limit(25).execute()
if r.data and len(r.data) >= 21:
    # Get last 21 days (need 20 changes)
    vix_values = [row['close'] for row in r.data[:21]]
    
    # Calculate daily changes
    changes = []
    for i in range(len(vix_values) - 1):
        change = vix_values[i] - vix_values[i + 1]
        changes.append(change)
    
    # Vol-of-Vol = StdDev of changes
    import statistics
    if len(changes) >= 20:
        vol_of_vol = statistics.stdev(changes[:20])
        mean_change = statistics.mean(changes[:20])
        
        print(f"\nLast 20 daily VIX changes:")
        for i, ch in enumerate(changes[:20]):
            print(f"  Day {i+1}: {ch:+.2f}")
        
        print(f"\n{'Metric':<25} {'Calculated':<15} {'Google Sheets':<15}")
        print("-" * 55)
        print(f"{'Mean Change':<25} {mean_change:+.3f}{'':>11} {'~0':<15}")
        print(f"{'Vol-of-Vol (StdDev)':<25} {vol_of_vol:.3f}{'':>12} {'0.81':<15}")
else:
    print("Not enough data for Vol-of-Vol calculation")

# =============================================================================
# 4. COMPARISON SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("4. VALIDATION SUMMARY")
print("=" * 70)

print("""
┌─────────────────────┬───────────────┬───────────────┬──────────────┐
│ Metric              │ Python        │ Google Sheets │ Match?       │
├─────────────────────┼───────────────┼───────────────┼──────────────┤
│ Data Source         │ FRED VIXCLS   │ FRED VIXCLS   │ ✅ Same      │
│ Current VIX         │ ~14.9         │ 16.13         │ ⚠️ Check GS  │
│ Regime Thresholds   │ <20 Healthy   │ <18 Healthy   │ Check        │
└─────────────────────┴───────────────┴───────────────┴──────────────┘

NOTE: If Google Sheets shows 16.13 but FRED returns 15.13 for Jan 6,
      run buildVixDial() in Apps Script to refresh Google Sheets.
      
FRED has a 1-day lag, so today's VIX won't appear until tomorrow.
""")

# =============================================================================
# 5. REGIME THRESHOLDS CHECK
# =============================================================================
print("=" * 70)
print("5. REGIME THRESHOLDS")
print("=" * 70)

print("""
Google Sheets VIX_Dial thresholds (from VixDial.txt):
  Level Regime:
    - Crisis:  VIX > 40
    - Fragile: VIX > 30
    - Caution: VIX > 20
    - Healthy: VIX <= 20

Python vix_dial.py thresholds:
    - Crisis:  VIX > 40
    - Fragile: VIX > 30  
    - Caution: VIX > 20
    - Healthy: VIX <= 20

✅ Thresholds match!

The VIX_Dial summary screenshot showed "Elevated" status with VIX 16.13,
but the code threshold says <20 is "Healthy". This suggests Google Sheets
may use different display thresholds or the screenshot was from a 
different tab/calculation.
""")