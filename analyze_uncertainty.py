import pandas as pd
import numpy as np

print("=== Analyzing NASA Power Data ===")
nasa_df = pd.read_csv('weather_data/nasa_power/nasa_power_merged_2025.csv')
nasa_df['Date'] = pd.to_datetime(nasa_df['Date'])

# Group by date and calculate daily average
nasa_daily = nasa_df.groupby('Date')['ALLSKY_SFC_SW_DWN'].agg(['mean', 'std', 'count']).reset_index()
nasa_daily.columns = ['Date', 'ghi_mean', 'ghi_std_across_locations', 'n_locations']

print(f"NASA Power Data:")
print(f"  Daily GHI mean: {nasa_daily['ghi_mean'].mean():.2f} MJ/m2/day")
print(f"  Daily GHI std:  {nasa_daily['ghi_mean'].std():.2f} MJ/m2/day")
print(f"  Coefficient of Variation (CV): {nasa_daily['ghi_mean'].std() / nasa_daily['ghi_mean'].mean():.3f}")

print("\n=== Analyzing GovData ===")
gov_df = pd.read_csv('weather_data/govdata/govdata_merged_2025.csv')
gov_df['Date'] = pd.to_datetime(gov_df['Date'])

print(f"GovData:")
print(f"  Daily GSR mean: {gov_df['KP_GSR'].mean():.2f} MJ/m2/day")
print(f"  Daily GSR std:  {gov_df['KP_GSR'].std():.2f} MJ/m2/day")
print(f"  Coefficient of Variation (CV): {gov_df['KP_GSR'].std() / gov_df['KP_GSR'].mean():.3f}")

print("\n=== Uncertainty Parameter ===")
nasa_cv = nasa_daily['ghi_mean'].std() / nasa_daily['ghi_mean'].mean()
gov_cv = gov_df['KP_GSR'].std() / gov_df['KP_GSR'].mean()

print(f"NASA CV:  {nasa_cv:.3f} (represents temporal variability)")
print(f"GovData CV: {gov_cv:.3f}")
print(f"\nRecommended MC load_std: {nasa_cv:.3f}")
print(f"(This is the relative uncertainty to apply during Monte Carlo)")
