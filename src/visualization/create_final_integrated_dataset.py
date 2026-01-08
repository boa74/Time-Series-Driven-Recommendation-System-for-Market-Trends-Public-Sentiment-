#!/usr/bin/env python3
"""
Create Final Integrated Dataset with Industry Information
==========================================================
This script creates a comprehensive dataset that includes:
1. Daily aggregated stock data (averages across all stocks)
2. Industry-level data (averages and statistics by industry)
3. S&P 500 index data
4. Depression metrics (index and word count)
5. Rainfall data
6. Time-based features

OUTPUT FILES:
- final_integrated_daily_data.csv (daily aggregates with all metrics)
- final_integrated_industry_data.csv (daily data by industry)
- final_integrated_stock_data.csv (individual stock data with all metrics)
"""

import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 80)
print("CREATING FINAL INTEGRATED DATASET WITH INDUSTRY INFORMATION")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD ALL DATASETS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: LOADING DATA")
print("=" * 80)

# Stock data with company info and industry
print("\n  Loading stock_data_wiki_clean.csv (with industry info)...")
stock_data = pd.read_csv('cleaned_data/stock_data_wiki_clean.csv')
stock_data['date'] = pd.to_datetime(stock_data['date'])
print(f"    ✓ {len(stock_data):,} rows")
print(f"    ✓ Date range: {stock_data['date'].min().date()} to {stock_data['date'].max().date()}")
print(f"    ✓ Unique tickers: {stock_data['ticker'].nunique()}")
print(f"    ✓ Unique industries: {stock_data['industry'].nunique()}")

# S&P 500 index
print("\n  Loading sp500.csv...")
sp500 = pd.read_csv('raw_data/sp500.csv')
sp500 = sp500.rename(columns={'Date': 'date'})
sp500['date'] = pd.to_datetime(sp500['date'])
print(f"    ✓ {len(sp500):,} rows")
print(f"    ✓ Date range: {sp500['date'].min().date()} to {sp500['date'].max().date()}")

# Depression news mentions
print("\n  Loading ccnews_depression_daily_count_final.csv...")
depression_news = pd.read_csv('cleaned_data/ccnews_depression_daily_count_final.csv')
depression_news['date'] = pd.to_datetime(depression_news['date'])
print(f"    ✓ {len(depression_news):,} rows")
print(f"    ✓ Date range: {depression_news['date'].min().date()} to {depression_news['date'].max().date()}")

# Depression index
print("\n  Loading depression_index.csv...")
depression_idx = pd.read_csv('raw_data/depression_index.csv')
depression_idx = depression_idx.rename(columns={'Week': 'date', 'depression': 'depression_index'})
depression_idx['date'] = pd.to_datetime(depression_idx['date'])
print(f"    ✓ {len(depression_idx):,} rows (weekly data)")
print(f"    ✓ Date range: {depression_idx['date'].min().date()} to {depression_idx['date'].max().date()}")

# Rainfall (average across all states)
print("\n  Loading rainfall.csv...")
rainfall = pd.read_csv('raw_data/rainfall.csv')
rainfall = rainfall.rename(columns={'Date': 'date'})
rainfall['date'] = pd.to_datetime(rainfall['date'])

# Calculate average rainfall across all states
state_columns = [col for col in rainfall.columns if col != 'date']
rainfall['rainfall'] = rainfall[state_columns].mean(axis=1)
rainfall_clean = rainfall[['date', 'rainfall']]
print(f"    ✓ {len(rainfall_clean):,} rows")
print(f"    ✓ Averaged across {len(state_columns)} states")

# ============================================================================
# STEP 2: PREPARE S&P 500 DATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: PREPARING S&P 500 DATA")
print("=" * 80)

sp500_clean = sp500[['date', 'Close_^GSPC', 'Return', 'Volatility_7']].copy()
sp500_clean = sp500_clean.rename(columns={
    'Close_^GSPC': 'sp500_close',
    'Return': 'sp500_return',
    'Volatility_7': 'sp500_volatility'
})

print(f"  ✓ Prepared {len(sp500_clean):,} S&P 500 records")

# ============================================================================
# STEP 3: FORWARD-FILL DEPRESSION INDEX (WEEKLY TO DAILY)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: CONVERTING DEPRESSION INDEX FROM WEEKLY TO DAILY")
print("=" * 80)

# Get date range from stock data
date_range = pd.date_range(
    start=stock_data['date'].min(),
    end=stock_data['date'].max(),
    freq='D'
)

# Create daily date frame
daily_dates = pd.DataFrame({'date': date_range})

# Merge and forward fill depression index
depression_daily = daily_dates.merge(depression_idx, on='date', how='left')
depression_daily['depression_index'] = depression_daily['depression_index'].fillna(method='ffill')

print(f"  ✓ Converted {len(depression_idx):,} weekly records to {len(depression_daily):,} daily records")
print(f"  ✓ Forward-filled missing values")

# ============================================================================
# STEP 4: CALCULATE DAILY AGGREGATES (ACROSS ALL STOCKS)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: CALCULATING DAILY AGGREGATES")
print("=" * 80)

print("\n  Aggregating stock data by date (averages across all stocks)...")
daily_agg = stock_data.groupby('date').agg({
    'open': 'mean',
    'high': 'mean',
    'low': 'mean',
    'close': 'mean',
    'volume': 'mean',
    'ticker': 'count'  # Number of stocks traded
}).reset_index()

daily_agg = daily_agg.rename(columns={
    'open': 'avg_stock_open',
    'high': 'avg_stock_high',
    'low': 'avg_stock_low',
    'close': 'avg_stock_close',
    'volume': 'avg_stock_volume',
    'ticker': 'num_stocks_traded'
})

# Calculate price range (volatility proxy)
daily_agg['price_range'] = daily_agg['avg_stock_high'] - daily_agg['avg_stock_low']

print(f"  ✓ Created {len(daily_agg):,} daily aggregate records")

# ============================================================================
# STEP 5: CALCULATE INDUSTRY AGGREGATES (BY INDUSTRY AND DATE)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: CALCULATING INDUSTRY-LEVEL AGGREGATES")
print("=" * 80)

print("\n  Aggregating stock data by industry and date...")
industry_agg = stock_data.groupby(['date', 'industry']).agg({
    'open': 'mean',
    'high': 'mean',
    'low': 'mean',
    'close': 'mean',
    'volume': 'mean',
    'ticker': 'count'  # Number of stocks in this industry
}).reset_index()

industry_agg = industry_agg.rename(columns={
    'open': 'industry_avg_open',
    'high': 'industry_avg_high',
    'low': 'industry_avg_low',
    'close': 'industry_avg_close',
    'volume': 'industry_avg_volume',
    'ticker': 'num_stocks_in_industry'
})

# Calculate industry-level volatility
industry_agg['industry_price_range'] = (
    industry_agg['industry_avg_high'] - industry_agg['industry_avg_low']
)

print(f"  ✓ Created {len(industry_agg):,} industry-level records")
print(f"  ✓ Industries: {industry_agg['industry'].nunique()}")
print(f"  ✓ Date range: {industry_agg['date'].min().date()} to {industry_agg['date'].max().date()}")

# ============================================================================
# STEP 6: MERGE ALL DATA - DAILY LEVEL
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: CREATING FINAL DAILY INTEGRATED DATASET")
print("=" * 80)

print("\n  Merging all data sources by date...")

# Start with daily stock aggregates
final_daily = daily_agg.copy()
print(f"  Starting with: {len(final_daily):,} rows")

# Merge S&P 500
final_daily = final_daily.merge(sp500_clean, on='date', how='left')
print(f"  ✓ Merged S&P 500: {len(final_daily):,} rows")

# Merge depression index
final_daily = final_daily.merge(depression_daily[['date', 'depression_index']], on='date', how='left')
print(f"  ✓ Merged depression index: {len(final_daily):,} rows")

# Merge depression word count
final_daily = final_daily.merge(depression_news, on='date', how='left')
print(f"  ✓ Merged depression word count: {len(final_daily):,} rows")

# Merge rainfall
final_daily = final_daily.merge(rainfall_clean, on='date', how='left')
print(f"  ✓ Merged rainfall: {len(final_daily):,} rows")

# Add time features
final_daily['year'] = final_daily['date'].dt.year
final_daily['month'] = final_daily['date'].dt.month
final_daily['day_of_week'] = final_daily['date'].dt.dayofweek
final_daily['quarter'] = final_daily['date'].dt.quarter
print(f"  ✓ Added time features")

# Sort by date
final_daily = final_daily.sort_values('date').reset_index(drop=True)

# ============================================================================
# STEP 7: MERGE ALL DATA - INDUSTRY LEVEL
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: CREATING FINAL INDUSTRY-LEVEL INTEGRATED DATASET")
print("=" * 80)

print("\n  Merging industry data with all metrics...")

# Start with industry aggregates
final_industry = industry_agg.copy()
print(f"  Starting with: {len(final_industry):,} rows")

# Merge S&P 500
final_industry = final_industry.merge(sp500_clean, on='date', how='left')
print(f"  ✓ Merged S&P 500: {len(final_industry):,} rows")

# Merge depression index
final_industry = final_industry.merge(depression_daily[['date', 'depression_index']], on='date', how='left')
print(f"  ✓ Merged depression index: {len(final_industry):,} rows")

# Merge depression word count
final_industry = final_industry.merge(depression_news, on='date', how='left')
print(f"  ✓ Merged depression word count: {len(final_industry):,} rows")

# Merge rainfall
final_industry = final_industry.merge(rainfall_clean, on='date', how='left')
print(f"  ✓ Merged rainfall: {len(final_industry):,} rows")

# Add time features
final_industry['year'] = final_industry['date'].dt.year
final_industry['month'] = final_industry['date'].dt.month
final_industry['day_of_week'] = final_industry['date'].dt.dayofweek
final_industry['quarter'] = final_industry['date'].dt.quarter
print(f"  ✓ Added time features")

# Sort by date and industry
final_industry = final_industry.sort_values(['date', 'industry']).reset_index(drop=True)

# ============================================================================
# STEP 8: CREATE STOCK-LEVEL DATASET (INDIVIDUAL STOCKS WITH ALL METRICS)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: CREATING FINAL STOCK-LEVEL INTEGRATED DATASET")
print("=" * 80)

print("\n  Merging individual stock data with all metrics...")

# Start with individual stock data
final_stock = stock_data.copy()
print(f"  Starting with: {len(final_stock):,} rows")

# Merge S&P 500
final_stock = final_stock.merge(sp500_clean, on='date', how='left')
print(f"  ✓ Merged S&P 500: {len(final_stock):,} rows")

# Merge depression index
final_stock = final_stock.merge(depression_daily[['date', 'depression_index']], on='date', how='left')
print(f"  ✓ Merged depression index: {len(final_stock):,} rows")

# Merge depression word count
final_stock = final_stock.merge(depression_news, on='date', how='left')
print(f"  ✓ Merged depression word count: {len(final_stock):,} rows")

# Merge rainfall
final_stock = final_stock.merge(rainfall_clean, on='date', how='left')
print(f"  ✓ Merged rainfall: {len(final_stock):,} rows")

# Add stock-level price range
final_stock['stock_price_range'] = final_stock['high'] - final_stock['low']

# Add time features
final_stock['year'] = final_stock['date'].dt.year
final_stock['month'] = final_stock['date'].dt.month
final_stock['day_of_week'] = final_stock['date'].dt.dayofweek
final_stock['quarter'] = final_stock['date'].dt.quarter
print(f"  ✓ Added features")

# Sort by date and ticker
final_stock = final_stock.sort_values(['date', 'ticker']).reset_index(drop=True)

# ============================================================================
# STEP 9: SAVE ALL DATASETS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 9: SAVING FINAL DATASETS")
print("=" * 80)

# Save daily aggregates
daily_file = 'final_integrated_daily_data.csv'
final_daily.to_csv(daily_file, index=False)
print(f"\n  ✓ Saved: {daily_file}")
print(f"    - {len(final_daily):,} rows")
print(f"    - {len(final_daily.columns)} columns")
print(f"    - Date range: {final_daily['date'].min().date()} to {final_daily['date'].max().date()}")

# Save industry-level data
industry_file = 'final_integrated_industry_data.csv'
final_industry.to_csv(industry_file, index=False)
print(f"\n  ✓ Saved: {industry_file}")
print(f"    - {len(final_industry):,} rows")
print(f"    - {len(final_industry.columns)} columns")
print(f"    - {final_industry['industry'].nunique()} industries")
print(f"    - Date range: {final_industry['date'].min().date()} to {final_industry['date'].max().date()}")

# Save stock-level data
stock_file = 'final_integrated_stock_data.csv'
final_stock.to_csv(stock_file, index=False)
print(f"\n  ✓ Saved: {stock_file}")
print(f"    - {len(final_stock):,} rows")
print(f"    - {len(final_stock.columns)} columns")
print(f"    - {final_stock['ticker'].nunique()} unique tickers")
print(f"    - {final_stock['industry'].nunique()} industries")
print(f"    - Date range: {final_stock['date'].min().date()} to {final_stock['date'].max().date()}")

# ============================================================================
# STEP 10: DATA QUALITY SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("STEP 10: DATA QUALITY SUMMARY")
print("=" * 80)

print("\n1. DAILY AGGREGATES (final_integrated_daily_data.csv)")
print("   " + "-" * 76)
print(f"   Columns: {', '.join(final_daily.columns.tolist())}")
print("\n   Missing values:")
missing_daily = final_daily.isnull().sum()
for col in final_daily.columns:
    if missing_daily[col] > 0:
        pct = missing_daily[col] / len(final_daily) * 100
        print(f"      {col:30s}: {missing_daily[col]:6,} ({pct:5.1f}%)")
    
print("\n2. INDUSTRY-LEVEL DATA (final_integrated_industry_data.csv)")
print("   " + "-" * 76)
print(f"   Top 5 industries by number of records:")
top_industries = final_industry['industry'].value_counts().head(5)
for industry, count in top_industries.items():
    print(f"      {industry:40s}: {count:6,} records")

print("\n3. STOCK-LEVEL DATA (final_integrated_stock_data.csv)")
print("   " + "-" * 76)
print(f"   Top 5 most traded stocks (by number of records):")
top_stocks = final_stock['ticker'].value_counts().head(5)
for ticker, count in top_stocks.items():
    company = final_stock[final_stock['ticker'] == ticker]['company'].iloc[0]
    print(f"      {ticker:6s} - {company:40s}: {count:4,} records")

# ============================================================================
# STEP 11: DISPLAY SAMPLE DATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 11: SAMPLE DATA PREVIEW")
print("=" * 80)

print("\n1. Daily Aggregates (first 5 rows):")
print(final_daily.head().to_string())

print("\n2. Industry-Level (first 5 rows):")
print(final_industry.head().to_string())

print("\n3. Stock-Level (first 5 rows):")
print(final_stock[['date', 'ticker', 'company', 'industry', 'close', 'depression_index', 'sp500_close']].head().to_string())

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ FINAL INTEGRATED DATASETS CREATED SUCCESSFULLY!")
print("=" * 80)

print("\nOUTPUT FILES:")
print(f"  1. {daily_file}")
print(f"     - Daily aggregates across all stocks")
print(f"     - Use for: Overall market analysis, time series analysis")
print(f"     - {len(final_daily):,} rows × {len(final_daily.columns)} columns")

print(f"\n  2. {industry_file}")
print(f"     - Daily aggregates by industry")
print(f"     - Use for: Industry-specific analysis, sector comparisons")
print(f"     - {len(final_industry):,} rows × {len(final_industry.columns)} columns")
print(f"     - {final_industry['industry'].nunique()} unique industries")

print(f"\n  3. {stock_file}")
print(f"     - Individual stock data with all metrics")
print(f"     - Use for: Stock-specific analysis, correlations")
print(f"     - {len(final_stock):,} rows × {len(final_stock.columns)} columns")
print(f"     - {final_stock['ticker'].nunique()} unique stocks")

print("\nKEY FEATURES INCLUDED:")
print("  ✓ Stock prices (open, high, low, close, volume)")
print("  ✓ S&P 500 index (close, return, volatility)")
print("  ✓ Depression metrics (Google Trends index, news word count)")
print("  ✓ Rainfall data (averaged across states)")
print("  ✓ Industry information (for sector analysis)")
print("  ✓ Time features (year, month, day_of_week, quarter)")
print("  ✓ Volatility metrics (price range)")

print("\n" + "=" * 80)
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
