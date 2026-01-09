"""
ARGUS-1 DIAGNOSTIC SCRIPT
==========================
Checks data availability and module configuration
"""

import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

# Check Supabase connection
print("="*70)
print("ARGUS-1 DIAGNOSTIC")
print("="*70)

try:
    from supabase import create_client
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    
    if not url or not key:
        print("❌ SUPABASE_URL or SUPABASE_KEY not set")
    else:
        client = create_client(url, key)
        print("✅ Supabase connected")
except Exception as e:
    print(f"❌ Supabase error: {e}")
    client = None

# =========================================================================
# 1. CHECK TABLE DATA
# =========================================================================
print("\n" + "-"*50)
print("1. TABLE DATA CHECK")
print("-"*50)

tables_to_check = [
    ("breadth_daily", "breadth_20d"),
    ("vix_history", "close"),
    ("credit_spread_daily", "hy_spread"),
    ("pillar_index_daily", "infra_5d"),
    ("bubble_index_daily", "bubble_index"),
    ("amri_daily", "amri_score"),
]

if client:
    for table, column in tables_to_check:
        try:
            result = client.table(table).select("*").order("date", desc=True).limit(5).execute()
            if result.data:
                print(f"\n✅ {table}: {len(result.data)} rows")
                row = result.data[0]
                print(f"   Latest date: {row.get('date', 'N/A')}")
                print(f"   {column}: {row.get(column, 'N/A')}")
                # Show all columns
                print(f"   Columns: {list(row.keys())[:8]}...")
            else:
                print(f"❌ {table}: No data")
        except Exception as e:
            print(f"❌ {table}: Error - {e}")

# =========================================================================
# 2. CHECK MCI DATA AVAILABILITY
# =========================================================================
print("\n" + "-"*50)
print("2. MCI DATA CHECK (needs 6+ breadth, 11+ VIX, 11+ credit rows)")
print("-"*50)

if client:
    # Breadth - needs 6 rows (5 lookback + 1 current)
    try:
        result = client.table("breadth_daily").select("date, breadth_20d").order("date", desc=True).limit(10).execute()
        print(f"\nbreadth_daily: {len(result.data)} rows available")
        if result.data and len(result.data) >= 6:
            current = result.data[0]
            previous = result.data[5]
            print(f"  Current ({current['date']}): {current.get('breadth_20d', 'N/A')}")
            print(f"  5D ago ({previous['date']}): {previous.get('breadth_20d', 'N/A')}")
            if current.get('breadth_20d') and previous.get('breadth_20d'):
                change = (current['breadth_20d'] - previous['breadth_20d']) * 100
                print(f"  5D Change: {change:+.2f}%")
        else:
            print(f"  ❌ Need 6+ rows, only have {len(result.data)}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # VIX - needs 11 rows (10 lookback + 1 current)
    try:
        result = client.table("vix_history").select("date, close").order("date", desc=True).limit(15).execute()
        print(f"\nvix_history: {len(result.data)} rows available")
        if result.data and len(result.data) >= 11:
            current = result.data[0]
            previous = result.data[10]
            print(f"  Current ({current['date']}): {current.get('close', 'N/A')}")
            print(f"  10D ago ({previous['date']}): {previous.get('close', 'N/A')}")
            if current.get('close') and previous.get('close'):
                change = current['close'] - previous['close']
                print(f"  10D Change: {change:+.2f} pts")
        else:
            print(f"  ❌ Need 11+ rows, only have {len(result.data)}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Credit - needs 11 rows
    try:
        result = client.table("credit_spread_daily").select("date, hy_spread").order("date", desc=True).limit(15).execute()
        print(f"\ncredit_spread_daily: {len(result.data)} rows available")
        if result.data and len(result.data) >= 11:
            current = result.data[0]
            previous = result.data[10]
            print(f"  Current ({current['date']}): {current.get('hy_spread', 'N/A')}")
            print(f"  10D ago ({previous['date']}): {previous.get('hy_spread', 'N/A')}")
            if current.get('hy_spread') and previous.get('hy_spread'):
                change = (current['hy_spread'] - previous['hy_spread']) * 100
                print(f"  10D Change: {change:+.0f} bps")
        else:
            print(f"  ❌ Need 11+ rows, only have {len(result.data)}")
    except Exception as e:
        print(f"  Error: {e}")

# =========================================================================
# 3. CHECK MODULE CONFIGURATION
# =========================================================================
print("\n" + "-"*50)
print("3. MODULE CONFIGURATION CHECK")
print("-"*50)

# Check if corrected modules are in place
try:
    from dials.mci import MCI_CONFIG
    print(f"\nMCI Config:")
    print(f"  VIX_THRESHOLD: {MCI_CONFIG.get('VIX_THRESHOLD', 'NOT SET')}")
    print(f"  CREDIT_THRESHOLD: {MCI_CONFIG.get('CREDIT_THRESHOLD', 'NOT SET')}")
    if MCI_CONFIG.get('VIX_THRESHOLD') == 6:
        print("  ✅ VIX threshold correct (6)")
    else:
        print("  ❌ VIX threshold wrong (should be 6)")
    if MCI_CONFIG.get('CREDIT_THRESHOLD') == 30:
        print("  ✅ Credit threshold correct (30)")
    else:
        print("  ❌ Credit threshold wrong (should be 30)")
except Exception as e:
    print(f"❌ MCI Config error: {e}")

# Check bubble index
try:
    from bubble_index import BubbleIndexCalculator
    calc = BubbleIndexCalculator()
    # Check if it has the 6-component method
    if hasattr(calc, 'calculate_valuation_extreme'):
        print("\n✅ Bubble Index: 6-component formula installed")
    else:
        print("\n❌ Bubble Index: Still using old CAPE-only formula")
except Exception as e:
    print(f"\n❌ Bubble Index error: {e}")

# Check AMRI
try:
    from analytical.amri import AMRICalculator, AMRI_WEIGHTS
    print(f"\nAMRI Weights: {AMRI_WEIGHTS}")
    if len(AMRI_WEIGHTS) == 4:
        print("  ✅ AMRI: 4-component formula")
    else:
        print(f"  ❌ AMRI: Wrong number of components ({len(AMRI_WEIGHTS)})")
except Exception as e:
    print(f"\n❌ AMRI error: {e}")

# =========================================================================
# 4. CALCULATE WITH CURRENT DATA
# =========================================================================
print("\n" + "-"*50)
print("4. MANUAL CALCULATION TEST")
print("-"*50)

if client:
    try:
        # Get actual data and calculate MCI manually
        breadth = client.table("breadth_daily").select("date, breadth_20d").order("date", desc=True).limit(10).execute()
        vix = client.table("vix_history").select("date, close").order("date", desc=True).limit(15).execute()
        credit = client.table("credit_spread_daily").select("date, hy_spread").order("date", desc=True).limit(15).execute()
        pillar = client.table("pillar_index_daily").select("*").order("date", desc=True).limit(1).execute()
        
        print("\nManual MCI Calculation:")
        
        # Breadth component
        if breadth.data and len(breadth.data) >= 6:
            b_curr = breadth.data[0].get('breadth_20d', 0) or 0
            b_prev = breadth.data[5].get('breadth_20d', 0) or 0
            b_change = (b_curr - b_prev) * 100
            b_score = max(-25, min(25, (b_change / 10) * 25))
            print(f"  Breadth: {b_change:+.1f}% → {b_score:+.1f}")
        else:
            b_score = 0
            print(f"  Breadth: No data → 0")
        
        # VIX component
        if vix.data and len(vix.data) >= 11:
            v_curr = vix.data[0].get('close', 0) or 0
            v_prev = vix.data[10].get('close', 0) or 0
            v_change = v_curr - v_prev
            v_score = max(-25, min(25, (-v_change / 6) * 25))
            print(f"  VIX: {v_change:+.2f} pts → {v_score:+.1f}")
        else:
            v_score = 0
            print(f"  VIX: No data → 0")
        
        # Credit component
        if credit.data and len(credit.data) >= 11:
            c_curr = credit.data[0].get('hy_spread', 0) or 0
            c_prev = credit.data[10].get('hy_spread', 0) or 0
            c_change_bps = (c_curr - c_prev) * 100
            c_score = max(-25, min(25, (-c_change_bps / 30) * 25))
            print(f"  Credit: {c_change_bps:+.0f} bps → {c_score:+.1f}")
        else:
            c_score = 0
            print(f"  Credit: No data → 0")
        
        # Pillar component
        if pillar.data:
            p = pillar.data[0]
            momenta = [
                p.get('infra_5d', 0) or 0,
                p.get('enterprise_5d', 0) or 0,
                p.get('macro_5d', 0) or 0,
                p.get('financial_5d', 0) or 0,
                p.get('productivity_5d', 0) or 0,
                p.get('demand_5d', 0) or 0,
            ]
            avg = sum(momenta) / len(momenta) * 100
            p_score = max(-25, min(25, (avg / 3) * 25))
            print(f"  Pillar: {avg:+.2f}% → {p_score:+.1f}")
        else:
            p_score = 0
            print(f"  Pillar: No data → 0")
        
        total = b_score + v_score + c_score + p_score
        print(f"\n  TOTAL MCI: {total:+.1f}")
        
    except Exception as e:
        print(f"Error: {e}")

print("\n" + "="*70)
print("END DIAGNOSTIC")
print("="*70)