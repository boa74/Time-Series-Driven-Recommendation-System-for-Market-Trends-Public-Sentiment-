# ============================================================================
# MERGE ALL DATASETS FOR TIME-SERIES ANALYSIS
# ============================================================================
# This script merges:
# 1. stock_data_wiki.csv (stock prices with company info)
# 2. sp500.csv (S&P 500 index data)
# 3. ccnews_depression_daily_count_final.csv (depression mentions in news)
# 4. depression_index.csv (depression index)
# 5. rainfall.csv (averaged across all states)
#
# OUTPUT: merged_analysis_data.csv
# ============================================================================

import pandas as pd
import numpy as np

print("=" * 80)
print("MERGING ALL DATASETS")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD ALL DATASETS
# ============================================================================
print("\n1. Loading datasets...")

# Stock data with company info
print("  Loading stock_data_wiki.csv...")
stock_data = pd.read_csv('cleaned_data/stock_data_wiki.csv')
stock_data['date'] = pd.to_datetime(stock_data['date'])
print(f"    ✓ {len(stock_data):,} rows, date range: {stock_data['date'].min().date()} to {stock_data['date'].max().date()}")

# S&P 500 index
print("  Loading sp500.csv...")
sp500 = pd.read_csv('raw_data/sp500.csv')
sp500 = sp500.rename(columns={'Date': 'date'})
sp500['date'] = pd.to_datetime(sp500['date'])
print(f"    ✓ {len(sp500):,} rows, date range: {sp500['date'].min().date()} to {sp500['date'].max().date()}")

# Depression news mentions
print("  Loading ccnews_depression_daily_count_final.csv...")
depression_news = pd.read_csv('cleaned_data/ccnews_depression_daily_count_final.csv')
depression_news['date'] = pd.to_datetime(depression_news['date'])
print(f"    ✓ {len(depression_news):,} rows, date range: {depression_news['date'].min().date()} to {depression_news['date'].max().date()}")

# Depression index
print("  Loading depression_index.csv...")
depression_idx = pd.read_csv('raw_data/depression_index.csv')
depression_idx['date'] = pd.to_datetime(depression_idx['date'])
print(f"    ✓ {len(depression_idx):,} rows, date range: {depression_idx['date'].min().date()} to {depression_idx['date'].max().date()}")

# Rainfall (need to average across all states)
print("  Loading rainfall.csv...")
rainfall = pd.read_csv('raw_data/rainfall.csv')
rainfall = rainfall.rename(columns={'Date': 'date'})
rainfall['date'] = pd.to_datetime(rainfall['date'])

# Calculate average rainfall across all states
state_columns = [col for col in rainfall.columns if col != 'date']
rainfall['avg_rainfall'] = rainfall[state_columns].mean(axis=1)
rainfall_avg = rainfall[['date', 'avg_rainfall']]
print(f"    ✓ {len(rainfall_avg):,} rows, averaged across {len(state_columns)} states")
print(f"      Date range: {rainfall_avg['date'].min().date()} to {rainfall_avg['date'].max().date()}")

# ============================================================================
# STEP 2: AGGREGATE STOCK DATA BY DATE (AVERAGE ACROSS ALL TICKERS)
# ============================================================================
print("\n2. Aggregating stock data by date...")
print("   (Calculating daily averages across all tickers)")

stock_daily = stock_data.groupby('date').agg({
    'open': 'mean',
    'high': 'mean',
    'low': 'mean',
    'close': 'mean',
    'volume': 'sum'
}).reset_index()

stock_daily = stock_daily.rename(columns={
    'open': 'avg_stock_open',
    'high': 'avg_stock_high',
    'low': 'avg_stock_low',
    'close': 'avg_stock_close',
    'volume': 'total_stock_volume'
})

print(f"   ✓ Aggregated to {len(stock_daily):,} unique dates")

# ============================================================================
# STEP 3: MERGE ALL DATASETS ON DATE
# ============================================================================
print("\n3. Merging all datasets on date...")

# Start with stock data as base (has the most comprehensive date range)
merged = stock_daily.copy()
print(f"   Starting with stock data: {len(merged):,} rows")

# Merge S&P 500 index
merged = merged.merge(
    sp500[['date', 'Close_^GSPC', 'Return', 'Volatility_7']],
    on='date',
    how='left'
)
merged = merged.rename(columns={
    'Close_^GSPC': 'sp500_close',
    'Return': 'sp500_return',
    'Volatility_7': 'sp500_volatility_7d'
})
print(f"   ✓ Merged S&P 500: {len(merged):,} rows")

# Merge rainfall
merged = merged.merge(
    rainfall_avg,
    on='date',
    how='left'
)
print(f"   ✓ Merged rainfall: {len(merged):,} rows")

# Merge depression index
merged = merged.merge(
    depression_idx,
    on='date',
    how='left'
)
print(f"   ✓ Merged depression index: {len(merged):,} rows")

# Merge depression news
merged = merged.merge(
    depression_news,
    on='date',
    how='left'
)
print(f"   ✓ Merged depression news: {len(merged):,} rows")

# ============================================================================
# STEP 4: DATA QUALITY CHECK
# ============================================================================
print("\n4. Data quality check...")

print(f"\n   Date range: {merged['date'].min().date()} to {merged['date'].max().date()}")
print(f"   Total rows: {len(merged):,}")
print(f"   Total columns: {len(merged.columns)}")

print("\n   Missing values by column:")
missing = merged.isnull().sum()
for col in merged.columns:
    if missing[col] > 0:
        pct = missing[col] / len(merged) * 100
        print(f"      {col:30s}: {missing[col]:6,} ({pct:5.1f}%)")

# ============================================================================
# STEP 5: ADD ADDITIONAL FEATURES
# ============================================================================
print("\n5. Adding additional features...")

# Extract date components
merged['year'] = merged['date'].dt.year
merged['month'] = merged['date'].dt.month
merged['day_of_week'] = merged['date'].dt.dayofweek
merged['quarter'] = merged['date'].dt.quarter

print("   ✓ Added: year, month, day_of_week, quarter")

# ============================================================================
# STEP 6: SAVE MERGED DATASET
# ============================================================================
print("\n6. Saving merged dataset...")

output_file = 'cleaned_data/merged_analysis_data.csv'
merged.to_csv(output_file, index=False)

print(f"   ✓ Saved to: {output_file}")
print(f"   ✓ Rows: {len(merged):,}")
print(f"   ✓ Columns: {len(merged.columns)}")

# ============================================================================
# STEP 7: DISPLAY SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("MERGE COMPLETE!")
print("=" * 80)

print(f"\nOutput file: {output_file}")
print(f"\nColumns in merged dataset:")
for i, col in enumerate(merged.columns, 1):
    print(f"  {i:2d}. {col}")

print(f"\nSample data (first 5 rows):")
print(merged.head().to_string())

print("\n" + "=" * 80)
print("✅ READY FOR ANALYSIS!")
print("=" * 80)
