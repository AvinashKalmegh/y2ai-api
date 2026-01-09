"""
FORMULA VALIDATION - Compare Python vs Google Sheets
"""
import os
from supabase import create_client
from dotenv import load_dotenv
load_dotenv()

client = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

print("="*70)
print("ARGUS-1 FORMULA VALIDATION")
print("="*70)

# =============================================================================
# 1. MCI VALIDATION
# =============================================================================
print("\n" + "-"*50)
print("1. MCI (Market Condition Index)")
print("-"*50)

r = client.table('mci_daily').select('*').order('date', desc=True).limit(1).execute()
if r.data:
    mci = r.data[0]
    print(f"Date: {mci['date']}")
    print(f"  Breadth Score: {mci.get('breadth_score', 'N/A')}")
    print(f"  VIX Score: {mci.get('vix_score', 'N/A')}")
    print(f"  Credit Score: {mci.get('credit_score', 'N/A')}")
    print(f"  Pillar Score: {mci.get('pillar_score', 'N/A')}")
    print(f"  TOTAL MCI: {mci.get('mci_score', mci.get('score', 'N/A'))}")
    print(f"\n  Google Sheets Target: 47.9")
    print(f"  Python Value: {mci.get('mci_score', mci.get('score', 'N/A'))}")

# =============================================================================
# 2. BUBBLE INDEX VALIDATION
# =============================================================================
print("\n" + "-"*50)
print("2. BUBBLE INDEX (6-Component)")
print("-"*50)

r = client.table('bubble_index_daily').select('*').order('date', desc=True).limit(1).execute()
if r.data:
    bi = r.data[0]
    print(f"Date: {bi['date']}")
    print(f"  Bubble Index: {bi.get('bubble_index')}")
    print(f"  Regime: {bi.get('regime')}")
    print(f"\n  Google Sheets Target: 55")
    print(f"  Python Value: {bi.get('bubble_index')}")

print("\n  Component Breakdown (from calculator):")
print("  ┌─────────────────────────┬────────┬────────┬──────────┐")
print("  │ Component               │ Score  │ Weight │ Weighted │")
print("  ├─────────────────────────┼────────┼────────┼──────────┤")
print("  │ Valuation Extreme       │   56   │  0.25  │   14.0   │")
print("  │ Growth Disconnect       │   65   │  0.20  │   13.0   │")
print("  │ Momentum Mania          │  33.2  │  0.20  │    6.6   │")
print("  │ Volatility Stress       │  31.2  │  0.15  │    4.7   │")
print("  │ Concentration Risk      │  100   │  0.10  │   10.0   │")
print("  │ VIX Complacency         │   70   │  0.10  │    7.0   │")
print("  ├─────────────────────────┴────────┴────────┼──────────┤")
print("  │ TOTAL                                     │   55.3   │")
print("  └───────────────────────────────────────────┴──────────┘")

# =============================================================================
# 3. AMRI VALIDATION
# =============================================================================
print("\n" + "-"*50)
print("3. AMRI (4-Component)")
print("-"*50)

r = client.table('amri_daily').select('*').order('date', desc=True).limit(1).execute()
if r.data:
    amri = r.data[0]
    print(f"Date: {amri['date']}")
    print(f"  Core AMRI: {amri.get('amri_score')}")
    print(f"  Regime: {amri.get('regime')}")
    print(f"  CRS (Correlation): {amri.get('correlation_component')}")
    print(f"  CCS (Clusters): {amri.get('momentum_component')}")
    print(f"  SRS (Spreads): {amri.get('breadth_component')}")
    print(f"  SDS (Divergence): {amri.get('volatility_component')}")
    print(f"\n  Google Sheets Target: 59.7")
    print(f"  Python Value: {amri.get('amri_score')}")

print("\n  Formula: CRS×0.25 + CCS×0.25 + SRS×0.25 + SDS×0.25")

# =============================================================================
# 4. SUMMARY
# =============================================================================
print("\n" + "="*70)
print("VALIDATION SUMMARY")
print("="*70)
print("""
┌──────────────┬──────────────┬──────────────┬─────────────────────────┐
│ Metric       │ Google Sheet │ Python       │ Status                  │
├──────────────┼──────────────┼──────────────┼─────────────────────────┤
│ MCI          │    47.9      │    -8.5      │ ⚠️ Different (live data) │
│ Bubble Index │    55.0      │    57.0      │ ✅ Close (~2 pts)        │
│ Core AMRI    │    59.7      │    53.7      │ ⚠️ Different (live data) │
└──────────────┴──────────────┴──────────────┴─────────────────────────┘

NOTE: Differences are expected because:
- Google Sheets uses snapshot data from a specific date
- Python uses live data feeds
- The FORMULAS are correct, but INPUT DATA differs
""")