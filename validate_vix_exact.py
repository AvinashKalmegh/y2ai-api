"""
VIX FORMULA VALIDATION - Using exact Google Sheets history
"""

# Last 25 days from Google Sheets VIX History (ending Dec 4, 2025)
gs_history = [
    (2025, 12, 4, 15.78, -0.30, 1.643),
    (2025, 12, 3, 16.08, -0.51, 1.683),
    (2025, 12, 2, 16.59, -0.65, 1.692),
    (2025, 12, 1, 17.24, 0.89, 1.739),
    (2025, 11, 28, 16.35, -0.86, 1.728),
    (2025, 11, 27, 17.21, 0.02, 1.722),
    (2025, 11, 26, 17.19, -1.37, 1.722),
    (2025, 11, 25, 18.56, -1.96, 1.695),
    (2025, 11, 24, 20.52, -2.91, 1.630),
    (2025, 11, 21, 23.43, -2.99, 1.477),
    (2025, 11, 20, 26.42, 2.76, 1.301),
    (2025, 11, 19, 23.66, -1.03, 1.241),
    (2025, 11, 18, 24.69, 2.31, 1.209),
    (2025, 11, 17, 22.38, 2.55, 1.129),
    (2025, 11, 14, 19.83, -0.17, 1.147),
    (2025, 11, 13, 20.00, 2.49, 1.507),
    (2025, 11, 12, 17.51, 0.23, 1.760),
    (2025, 11, 11, 17.28, -0.32, 1.758),
    (2025, 11, 10, 17.60, -1.48, 1.808),
    (2025, 11, 7, 19.08, -0.42, 1.869),
    (2025, 11, 6, 19.50, 1.49, 2.201),
    (2025, 11, 5, 18.01, -0.99, 2.180),
    (2025, 11, 4, 19.00, 1.83, 2.179),
    (2025, 11, 3, 17.17, -0.27, 2.150),
    (2025, 10, 31, 17.44, 0.53, 2.150),
]

# Extract just VIX values and changes
vix_values = [row[3] for row in gs_history]
vix_changes = [row[4] for row in gs_history]
gs_vol_of_vol = [row[5] for row in gs_history]

current_vix = vix_values[0]  # 15.78

print("="*70)
print("VIX FORMULA VALIDATION - Google Sheets Dec 4, 2025")
print("="*70)

# 1. Moving Averages
ma_10d = sum(vix_values[:10]) / 10
ma_20d = sum(vix_values[:20]) / 20

print(f"\nCurrent VIX: {current_vix}")
print(f"10-Day MA: {ma_10d:.2f}")
print(f"20-Day MA: {ma_20d:.2f}")

# 2. VIX Change calculations
change_1d = vix_changes[0]  # Already calculated in sheet
change_5d = current_vix - vix_values[5]  # vs 5 days ago
change_10d = current_vix - vix_values[10]  # vs 10 days ago

print(f"\n1D Change: {change_1d:.2f}")
print(f"5D Change: {change_5d:.2f}")
print(f"10D Change: {change_10d:.2f}")

# 3. Z-Score (VIX vs 20-day mean, using 20-day std)
std_20 = (sum((x - ma_20d) ** 2 for x in vix_values[:20]) / 20) ** 0.5
zscore = (current_vix - ma_20d) / std_20

print(f"\n20D Std Dev: {std_20:.2f}")
print(f"Z-Score: {zscore:.2f}")

# 4. Vol-of-Vol (std dev of daily VIX changes over 20 days)
changes_20d = vix_changes[:20]
vov_mean = sum(changes_20d) / 20
vov_variance = sum((x - vov_mean) ** 2 for x in changes_20d) / 20
vol_of_vol_calc = vov_variance ** 0.5

print(f"\nVol-of-Vol Calculation:")
print(f"  Mean of 20D changes: {vov_mean:.3f}")
print(f"  Variance: {vov_variance:.3f}")
print(f"  Python Vol-of-Vol: {vol_of_vol_calc:.3f}")
print(f"  Google Sheets Vol-of-Vol: {gs_vol_of_vol[0]:.3f}")
print(f"  Match: {'✅' if abs(vol_of_vol_calc - gs_vol_of_vol[0]) < 0.01 else '❌'}")

# 5. Regime determination
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

print(f"\nRegime: {regime}")

# 6. Bollinger Bands (if using 20-day)
bb_upper = ma_20d + (2 * std_20)
bb_lower = ma_20d - (2 * std_20)

print(f"\nBollinger Bands (2 std):")
print(f"  Upper: {bb_upper:.2f}")
print(f"  Middle (MA20): {ma_20d:.2f}")
print(f"  Lower: {bb_lower:.2f}")

print("\n" + "="*70)
print("FORMULA SUMMARY")
print("="*70)
print("""
Vol-of-Vol = StdDev of daily VIX changes over 20 days
           = sqrt(sum((change_i - mean_change)^2) / 20)

Z-Score = (Current VIX - 20D MA) / 20D StdDev

Regimes:
  < 12:  Complacent
  12-18: Healthy
  18-25: Elevated
  25-35: Stressed
  > 35:  Crisis
""")