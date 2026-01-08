#!/usr/bin/env python3
"""
Clean and prepare all raw datasets for analysis
No MongoDB required - works directly with CSV files
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("DATA CLEANING PIPELINE")
print("=" * 80)

# ============================================================================
# 1. CLEAN STOCK DATA
# ============================================================================
print("\n1. Cleaning stock_data.csv...")
stock = pd.read_csv('csv_exports/stock_data.csv')
print(f"   Raw: {len(stock):,} rows")

# Convert date and remove invalid rows
stock['date'] = pd.to_datetime(stock['date'], errors='coerce')
stock = stock.dropna(subset=['date', 'ticker', 'close'])
stock = stock[stock['close'] > 0]  # Remove invalid prices

# Merge with company info
company = pd.read_csv('company_info.csv')
stock_clean = stock.merge(company, on='ticker', how='left')

print(f"   Clean: {len(stock_clean):,} rows")
stock_clean.to_csv('csv_exports/stock_data_clean.csv', index=False)
print("   ✓ Saved: stock_data_clean.csv")

# ============================================================================
# 2. CLEAN SP500 DATA
# ============================================================================
print("\n2. Cleaning sp500.csv...")
sp500 = pd.read_csv('csv_exports/sp500.csv')
sp500['Date'] = pd.to_datetime(sp500['Date'], errors='coerce')
sp500 = sp500.dropna(subset=['Date', 'Close_^GSPC'])
sp500 = sp500.rename(columns={'Date': 'date'})

print(f"   Clean: {len(sp500):,} rows")
sp500.to_csv('csv_exports/sp500_clean.csv', index=False)
print("   ✓ Saved: sp500_clean.csv")

# ============================================================================
# 3. CLEAN DEPRESSION NEWS DATA
# ============================================================================
print("\n3. Cleaning ccnews_depression_daily_count_final.csv...")
depression_news = pd.read_csv('csv_exports/ccnews_depression_daily_count_final.csv')
depression_news['date'] = pd.to_datetime(depression_news['date'], errors='coerce')
depression_news = depression_news.dropna(subset=['date'])

print(f"   Clean: {len(depression_news):,} rows")
print(f"   Date range: {depression_news['date'].min().date()} to {depression_news['date'].max().date()}")
print("   ✓ Already clean")

# ============================================================================
# 4. CLEAN DEPRESSION INDEX DATA
# ============================================================================
print("\n4. Cleaning depression_index.csv...")
depression_idx = pd.read_csv('csv_exports/depression_index.csv')
depression_idx['date'] = pd.to_datetime(depression_idx['date'], errors='coerce')
depression_idx = depression_idx.dropna(subset=['date', 'depression_index'])

print(f"   Clean: {len(depression_idx):,} rows")
depression_idx.to_csv('csv_exports/depression_index_clean.csv', index=False)
print("   ✓ Saved: depression_index_clean.csv")

# ============================================================================
# 5. CLEAN AND AGGREGATE RAINFALL DATA
# ============================================================================
print("\n5. Cleaning rainfall.csv...")
rainfall = pd.read_csv('csv_exports/rainfall.csv')
rainfall['Date'] = pd.to_datetime(rainfall['Date'], errors='coerce')
rainfall = rainfall.dropna(subset=['Date'])

# Average across all states
state_cols = [col for col in rainfall.columns if col != 'Date']
rainfall['avg_rainfall'] = rainfall[state_cols].mean(axis=1)
rainfall_clean = rainfall[['Date', 'avg_rainfall']].rename(columns={'Date': 'date'})

print(f"   Clean: {len(rainfall_clean):,} rows")
print(f"   Averaged across {len(state_cols)} states")
rainfall_clean.to_csv('csv_exports/rainfall_clean.csv', index=False)
print("   ✓ Saved: rainfall_clean.csv")

# ============================================================================
# 6. CREATE MERGED DATASET
# ============================================================================
print("\n6. Creating merged dataset...")

# Aggregate stock by date
stock_daily = stock_clean.groupby('date').agg({
    'open': 'mean',
    'high': 'mean', 
    'low': 'mean',
    'close': 'mean',
    'volume': 'sum'
}).reset_index()

# Merge all datasets
merged = stock_daily.copy()
merged = merged.merge(sp500[['date', 'Close_^GSPC', 'Return', 'Volatility_7']], on='date', how='left')
merged = merged.merge(rainfall_clean, on='date', how='left')
merged = merged.merge(depression_idx, on='date', how='left')
merged = merged.merge(depression_news, on='date', how='left')

# Add date features
merged['year'] = merged['date'].dt.year
merged['month'] = merged['date'].dt.month
merged['day_of_week'] = merged['date'].dt.dayofweek
merged['quarter'] = merged['date'].dt.quarter

# Rename columns
merged = merged.rename(columns={
    'open': 'avg_stock_open',
    'high': 'avg_stock_high',
    'low': 'avg_stock_low',
    'close': 'avg_stock_close',
    'volume': 'total_stock_volume',
    'Close_^GSPC': 'sp500_close',
    'Return': 'sp500_return',
    'Volatility_7': 'sp500_volatility_7d'
})

merged.to_csv('csv_exports/merged_clean.csv', index=False)
print(f"   ✓ Saved: merged_clean.csv ({len(merged):,} rows, {len(merged.columns)} columns)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ CLEANING COMPLETE!")
print("=" * 80)
print("\nCleaned files created:")
print("  1. stock_data_clean.csv")
print("  2. sp500_clean.csv")
print("  3. depression_index_clean.csv")
print("  4. rainfall_clean.csv")
print("  5. merged_clean.csv (all datasets combined)")
print("\nAll files saved in: csv_exports/")
