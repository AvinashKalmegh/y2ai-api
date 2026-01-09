"""
RUN BUBBLE INDEX AND AMRI (FIXED)
"""

import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

print("="*60)
print("BUBBLE INDEX & AMRI CALCULATOR")
print("="*60)

# 1. BUBBLE INDEX
print("\n" + "-"*50)
print("1. BUBBLE INDEX")
print("-"*50)

try:
    from bubble_index import BubbleIndexCalculator
    
    calc = BubbleIndexCalculator()
    data = calc.calculate()
    
    # Access as attribute, not dict
    print(f"\nBubble Index: {data.bubble_index}")
    print(f"Regime: {data.regime}")
    
    calc.save_to_supabase(data)
    print("✅ Bubble Index saved")
    
except Exception as e:
    print(f"❌ Bubble Index error: {e}")
    import traceback
    traceback.print_exc()

# 2. AMRI
print("\n" + "-"*50)
print("2. AMRI")
print("-"*50)

try:
    from analytical.amri import AMRICalculator
    
    calc = AMRICalculator()
    data = calc.calculate()
    print(f"\nAMRI: {data}")
    
    calc.save_to_supabase(data)
    print("✅ AMRI saved")
    
except Exception as e:
    print(f"❌ AMRI error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("DONE")
print("="*60)