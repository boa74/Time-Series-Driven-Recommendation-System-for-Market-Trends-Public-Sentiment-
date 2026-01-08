-- ============================================================================
-- Load Data into PostgreSQL Tables
-- ============================================================================
-- This script loads data from CSV files into the database
-- Run AFTER creating the schema with create_timeseries_postgres_schema.sql
-- ============================================================================

-- Set client encoding
SET client_encoding = 'UTF8';

-- ============================================================================
-- METHOD 1: Load from CSV files (if CSV files are in same directory)
-- ============================================================================

-- Load stock data from stock_data_wiki_clean.csv
-- Adjust path to your CSV file location
\COPY stock_data(date, ticker, industry, open_price, high_price, low_price, close_price, volume) 
FROM '/Users/boahkim/Documents/GitHub/Fundamentals-of-Data-Engineering_bk/csv_exports/stock_data_wiki_clean.csv' 
DELIMITER ',' 
CSV HEADER;

-- Load S&P 500 data
\COPY sp500_data(date, sp500_open, sp500_high, sp500_low, sp500_close, sp500_volume, sp500_return, sp500_volatility) 
FROM '/Users/boahkim/Documents/GitHub/Fundamentals-of-Data-Engineering_bk/csv_exports/sp500.csv' 
DELIMITER ',' 
CSV HEADER;

-- Load depression index (Google Trends - weekly)
\COPY depression_index(week_start_date, depression_index) 
FROM '/Users/boahkim/Documents/GitHub/Fundamentals-of-Data-Engineering_bk/csv_exports/depression_index.csv' 
DELIMITER ',' 
CSV HEADER;

-- Load depression word count (daily news mentions)
\COPY depression_word_count(date, word_count) 
FROM '/Users/boahkim/Documents/GitHub/Fundamentals-of-Data-Engineering_bk/csv_exports/ccnews_depression_daily_count_final.csv' 
DELIMITER ',' 
CSV HEADER;

-- Load rainfall data
\COPY rainfall_data(date, rainfall_mm) 
FROM '/Users/boahkim/Documents/GitHub/Fundamentals-of-Data-Engineering_bk/csv_exports/rainfall.csv' 
DELIMITER ',' 
CSV HEADER;

-- Load merged daily data (main analysis table)
\COPY daily_merged_data(date, avg_open, avg_high, avg_low, avg_close, avg_volume, price_range, sp500_close, sp500_return, sp500_volatility, depression_index, depression_word_count, rainfall_mm, num_stocks_traded) 
FROM '/Users/boahkim/Documents/GitHub/Fundamentals-of-Data-Engineering_bk/csv_exports/merged_analysis_data.csv' 
DELIMITER ',' 
CSV HEADER;

-- Load correlation statistics
\COPY correlation_statistics(variable_x, variable_y, correlation, p_value, r_squared, ci_lower, ci_upper, sample_size, significance_level, effect_size, bonferroni_significant, importance_score) 
FROM '/Users/boahkim/Documents/GitHub/Fundamentals-of-Data-Engineering_bk/csv_exports/correlation_statistics_full.csv' 
DELIMITER ',' 
CSV HEADER;

-- ============================================================================
-- Verify data loaded successfully
-- ============================================================================

-- Check row counts
DO $$
DECLARE
    stock_count INTEGER;
    sp500_count INTEGER;
    depression_idx_count INTEGER;
    depression_wc_count INTEGER;
    rainfall_count INTEGER;
    merged_count INTEGER;
    corr_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO stock_count FROM stock_data;
    SELECT COUNT(*) INTO sp500_count FROM sp500_data;
    SELECT COUNT(*) INTO depression_idx_count FROM depression_index;
    SELECT COUNT(*) INTO depression_wc_count FROM depression_word_count;
    SELECT COUNT(*) INTO rainfall_count FROM rainfall_data;
    SELECT COUNT(*) INTO merged_count FROM daily_merged_data;
    SELECT COUNT(*) INTO corr_count FROM correlation_statistics;
    
    RAISE NOTICE '========================================';
    RAISE NOTICE 'DATA LOAD VERIFICATION';
    RAISE NOTICE '========================================';
    RAISE NOTICE 'stock_data: % rows', stock_count;
    RAISE NOTICE 'sp500_data: % rows', sp500_count;
    RAISE NOTICE 'depression_index: % rows', depression_idx_count;
    RAISE NOTICE 'depression_word_count: % rows', depression_wc_count;
    RAISE NOTICE 'rainfall_data: % rows', rainfall_count;
    RAISE NOTICE 'daily_merged_data: % rows', merged_count;
    RAISE NOTICE 'correlation_statistics: % rows', corr_count;
    RAISE NOTICE '========================================';
END $$;

-- Show date ranges
SELECT 'stock_data' as table_name, MIN(date) as min_date, MAX(date) as max_date, COUNT(DISTINCT date) as distinct_dates FROM stock_data
UNION ALL
SELECT 'sp500_data', MIN(date), MAX(date), COUNT(DISTINCT date) FROM sp500_data
UNION ALL
SELECT 'depression_index', MIN(week_start_date), MAX(week_start_date), COUNT(DISTINCT week_start_date) FROM depression_index
UNION ALL
SELECT 'depression_word_count', MIN(date), MAX(date), COUNT(DISTINCT date) FROM depression_word_count
UNION ALL
SELECT 'rainfall_data', MIN(date), MAX(date), COUNT(DISTINCT date) FROM rainfall_data
UNION ALL
SELECT 'daily_merged_data', MIN(date), MAX(date), COUNT(DISTINCT date) FROM daily_merged_data;

-- Show sample data from merged table
SELECT 
    'Sample from daily_merged_data:' as info,
    date,
    avg_close,
    sp500_close,
    depression_index,
    depression_word_count
FROM daily_merged_data
ORDER BY date
LIMIT 5;

-- Show top correlations
SELECT 
    'Top 5 correlations by importance:' as info,
    variable_x,
    variable_y,
    correlation,
    p_value,
    importance_score,
    bonferroni_significant
FROM correlation_statistics
ORDER BY importance_score DESC, ABS(correlation) DESC
LIMIT 5;
