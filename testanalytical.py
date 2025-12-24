"""
Test script for the Analytical module
Run this to verify everything works
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("ARGUS-1 ANALYTICAL MODULE TEST")
print("=" * 60)
print(f"Time: {datetime.now()}")
print()

# Check environment
print("[1/6] Checking environment...")
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    print("❌ SUPABASE_URL or SUPABASE_KEY not set in .env")
    sys.exit(1)
print(f"✅ Supabase URL: {supabase_url[:30]}...")
print()

# Test imports
print("[2/6] Testing imports...")
try:
    from analytical import (
        ARGUS1Calculator,
        AMRICalculator,
        TTICalculator,
        SACCalculator,
        FingerprintMatcher,
        RotationTracker,
        RecoveryDetector,
        EventsTracker,
        Regime,
        Authority,
    )
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
print()

# Test Supabase connection
print("[3/6] Testing Supabase connection...")
try:
    from supabase import create_client
    client = create_client(supabase_url, supabase_key)
    
    # Test each table
    tables = ["bubble_index_daily", "hypergraph_signals", "daily_signals", "stock_tracker_daily"]
    for table in tables:
        try:
            r = client.table(table).select("*").limit(1).execute()
            count = len(r.data) if r.data else 0
            if count > 0:
                print(f"  ✅ {table}: has data")
            else:
                print(f"  ⚠️  {table}: empty (no data yet)")
        except Exception as e:
            print(f"  ❌ {table}: {e}")
    
    # Test new table
    try:
        r = client.table("argus_master_signals").select("*").limit(1).execute()
        print(f"  ✅ argus_master_signals: table exists")
    except Exception as e:
        print(f"  ❌ argus_master_signals: {e}")
        print("     Run the schema.sql to create this table")

except Exception as e:
    print(f"❌ Supabase connection error: {e}")
    sys.exit(1)
print()

# Test individual calculators
print("[4/6] Testing individual calculators...")

# AMRI
try:
    amri_calc = AMRICalculator(client)
    amri_result = amri_calc.calculate()
    print(f"  ✅ AMRI: {amri_result.composite} ({amri_result.regime.value})")
except Exception as e:
    print(f"  ❌ AMRI: {e}")

# TTI
try:
    tti_calc = TTICalculator(client)
    tti_result = tti_calc.calculate(amri_result.composite if 'amri_result' in dir() else 50)
    print(f"  ✅ TTI: {tti_result.display}")
except Exception as e:
    print(f"  ❌ TTI: {e}")

# SAC
try:
    sac_calc = SACCalculator(client)
    sac_result = sac_calc.calculate(amri_result.composite if 'amri_result' in dir() else 50)
    print(f"  ✅ SAC: {sac_result.display}")
except Exception as e:
    print(f"  ❌ SAC: {e}")

# Events
try:
    events_calc = EventsTracker()
    events_result = events_calc.calculate()
    print(f"  ✅ Events: {events_result.display}")
except Exception as e:
    print(f"  ❌ Events: {e}")

# Fingerprints
try:
    fp_calc = FingerprintMatcher()
    fp_result = fp_calc.calculate(current_amri=55, current_contagion=45, current_vix=18)
    print(f"  ✅ Fingerprint: {fp_result.display}")
except Exception as e:
    print(f"  ❌ Fingerprint: {e}")

# Rotation
try:
    rot_calc = RotationTracker(client)
    rot_result = rot_calc.calculate()
    print(f"  ✅ Rotation: {rot_result.display}")
except Exception as e:
    print(f"  ❌ Rotation: {e}")

print()

# Test full calculator
print("[5/6] Running full ARGUS1Calculator...")
try:
    calculator = ARGUS1Calculator(client)
    result = calculator.run()
    
    print()
    print("-" * 60)
    print("RESULT SUMMARY")
    print("-" * 60)
    print(f"  Date:       {result.date}")
    print(f"  Regime:     {result.regime.value}")
    print(f"  Authority:  {result.authority.value}")
    print(f"  Confidence: {result.confidence.value}")
    print(f"  AMRI:       {result.amri.composite}")
    print(f"    AMRI-S:   {result.amri.decomposition.amri_s:.1f}")
    print(f"    AMRI-B:   {result.amri.decomposition.amri_b:.1f}")
    print(f"    AMRI-C:   {result.amri.decomposition.amri_c:.1f}")
    print(f"  TTI:        {result.tti.display}")
    print(f"  SAC:        {result.sac.display}")
    print(f"  Contagion:  {result.contagion.display}")
    print(f"  VETO:       {'YES' if result.nst.veto_active else 'NO'}")
    print(f"  Events:     {result.events.display}")
    print("-" * 60)
    
except Exception as e:
    print(f"❌ Calculator error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# Test storage
print("[6/6] Testing storage to Supabase...")
try:
    success = calculator.store_result(result)
    if success:
        print("✅ Result stored to argus_master_signals")
    else:
        print("⚠️  Storage returned False (check logs)")
except Exception as e:
    print(f"❌ Storage error: {e}")
print()

print("=" * 60)
print("TEST COMPLETE")
print("=" * 60)