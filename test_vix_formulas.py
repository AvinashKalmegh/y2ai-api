"""
VIX FORMULA VALIDATION
Test with Google Sheets values
"""
import os
from supabase import create_client
from dotenv import load_dotenv
load_dotenv()
client = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

print("="*70)
print("VIX FORMULA TEST WITH GOOGLE SHEETS VALUES")
print("="*70)

# Get last 20 days of VIX from vix_history
r = client.table('vix_history').select('date, close').order('date', desc=True).limit(25).execute()
vix_data = r.data if r.data else []

if len(vix_data) >= 20:
    closes = [float(row['close']) for row in vix_data[:20]]
    current_vix = closes[0]
    
    # Calculate metrics
    ma_10d = sum(closes[:10]) / 10
    ma_20d = sum(closes[:20]) / 20
    
    # Standard deviation for z-score
    mean_20 = ma_20d
    variance = sum((x - mean_20) ** 2 for x in closes[:20]) / 20
    std_20 = variance ** 0.5
    
    # Z-score
    zscore = (current_vix - mean_20) / std_20 if std_20 > 0 else 0
    
    # Changes
    change_1d = current_vix - closes[1] if len(closes) > 1 else 0
    change_5d = current_vix - closes[5] if len(closes) > 5 else 0
    change_10d = current_vix - closes[10] if len(closes) > 10 else 0
    
    # Vol-of-Vol (standard deviation of daily changes)
    daily_changes = [closes[i] - closes[i+1] for i in range(min(20, len(closes)-1))]
    if daily_changes:
        vov_mean = sum(daily_changes) / len(daily_changes)
        vov_variance = sum((x - vov_mean) ** 2 for x in daily_changes) / len(daily_changes)
        vol_of_vol = vov_variance ** 0.5
    else:
        vol_of_vol = 0
    
    # Regime
    if current_vix < 12:
        regime = "Complacent"
    elif current_vix < 18:
        regime = "Healthy"
    elif current_vix < 25:
        regime = "Elevated"
    elif current_vix < 35:
        regime = "Stressed"
    else:
        regime = "Crisis"
    
    print(f"\nUsing VIX history from Python (latest: {vix_data[0]['date']})")
    print(f"\n{'Metric':<20} {'Python Calc':<15} {'Google Sheets':<15} {'Match?'}")
    print("-"*65)
    print(f"{'Current VIX':<20} {current_vix:<15.2f} {16.13:<15} {'⚠️ Data source'}")
    print(f"{'Regime':<20} {regime:<15} {'Elevated':<15} {'✅' if regime == 'Elevated' else '❌'}")
    print(f"{'10-Day MA':<20} {ma_10d:<15.2f} {15.48:<15}")
    print(f"{'20-Day MA':<20} {ma_20d:<15.2f} {16.72:<15}")
    print(f"{'1D Change':<20} {change_1d:<15.2f} {1.00:<15}")
    print(f"{'5D Change':<20} {change_5d:<15.2f} {1.62:<15}")
    print(f"{'Z-Score':<20} {zscore:<15.2f} {-0.79:<15}")
    print(f"{'Vol-of-Vol':<20} {vol_of_vol:<15.2f} {0.81:<15}")
    
    print("\n" + "-"*65)
    print("RAW VIX DATA (last 10 days):")
    for row in vix_data[:10]:
        print(f"  {row['date']}: {row['close']}")

    # Now test with EXACT Google Sheets VIX values
    print("\n" + "="*70)
    print("TEST WITH EXACT GOOGLE SHEETS VIX HISTORY")
    print("="*70)
    
    # From Google Sheets history (visible in screenshot)
    gs_vix = [16.13, 15.13, 14.95, 14.51, 14.95, 14.33, 14.2, 13.6, 13.47, 14.0, 
              14.08, 14.77, 15.72, 17.22, 17.74, 17.35, 15.88, 15.95, 14.52, 13.36]
    
    gs_current = gs_vix[0]
    gs_ma10 = sum(gs_vix[:10]) / 10
    gs_ma20 = sum(gs_vix[:20]) / 20
    gs_std20 = (sum((x - gs_ma20) ** 2 for x in gs_vix[:20]) / 20) ** 0.5
    gs_zscore = (gs_current - gs_ma20) / gs_std20
    
    print(f"\n{'Metric':<20} {'Python Calc':<15} {'Google Sheets':<15} {'Match?'}")
    print("-"*65)
    print(f"{'Current VIX':<20} {gs_current:<15.2f} {16.13:<15} ✅")
    print(f"{'10-Day MA':<20} {gs_ma10:<15.2f} {15.48:<15} {'✅' if abs(gs_ma10 - 15.48) < 0.1 else '❌'}")
    print(f"{'20-Day MA':<20} {gs_ma20:<15.2f} {16.72:<15} {'✅' if abs(gs_ma20 - 16.72) < 0.1 else '❌'}")
    print(f"{'Z-Score':<20} {gs_zscore:<15.2f} {-0.79:<15} {'✅' if abs(gs_zscore - (-0.79)) < 0.1 else '❌'}")

else:
    print("Not enough VIX history data")