-- ============================================================================
-- PostgreSQL Query Examples for Time Series Analysis
-- ============================================================================
-- Useful queries for analyzing depression-stock correlation data
-- Copy and paste these into psql or your PostgreSQL client
-- ============================================================================

-- ============================================================================
-- 1. BASIC DATA EXPLORATION
-- ============================================================================

-- Get row counts for all tables
SELECT 
    'stock_data' as table_name, COUNT(*) as rows FROM stock_data
UNION ALL SELECT 'sp500_data', COUNT(*) FROM sp500_data
UNION ALL SELECT 'depression_index', COUNT(*) FROM depression_index
UNION ALL SELECT 'depression_word_count', COUNT(*) FROM depression_word_count
UNION ALL SELECT 'rainfall_data', COUNT(*) FROM rainfall_data
UNION ALL SELECT 'daily_merged_data', COUNT(*) FROM daily_merged_data
UNION ALL SELECT 'correlation_statistics', COUNT(*) FROM correlation_statistics;

-- Get date ranges
SELECT 
    'stock_data' as table_name,
    MIN(date) as start_date,
    MAX(date) as end_date,
    MAX(date)::date - MIN(date)::date as days,
    COUNT(DISTINCT date) as distinct_dates
FROM stock_data
UNION ALL
SELECT 'sp500_data', MIN(date), MAX(date), 
       MAX(date)::date - MIN(date)::date, COUNT(DISTINCT date)
FROM sp500_data
UNION ALL
SELECT 'daily_merged_data', MIN(date), MAX(date),
       MAX(date)::date - MIN(date)::date, COUNT(DISTINCT date)
FROM daily_merged_data;

-- Sample data from merged table
SELECT * FROM daily_merged_data
ORDER BY date
LIMIT 10;

-- ============================================================================
-- 2. CORRELATION ANALYSIS RESULTS
-- ============================================================================

-- Get all significant correlations (sorted by importance)
SELECT 
    variable_x,
    variable_y,
    ROUND(correlation::numeric, 4) as r,
    p_value,
    ROUND((r_squared * 100)::numeric, 2) || '%' as variance_explained,
    significance_level,
    effect_size,
    CASE WHEN bonferroni_significant THEN '✓' ELSE '✗' END as bonferroni,
    importance_score
FROM correlation_statistics
WHERE p_value < 0.05
ORDER BY importance_score DESC, ABS(correlation) DESC;

-- Get ONLY the robust correlations (survive Bonferroni correction)
SELECT 
    variable_x || ' ↔ ' || variable_y as relationship,
    ROUND(correlation::numeric, 4) as correlation,
    p_value,
    ROUND((r_squared * 100)::numeric, 2) || '%' as variance_explained,
    '[' || ROUND(ci_lower::numeric, 3) || ', ' || 
           ROUND(ci_upper::numeric, 3) || ']' as confidence_interval_95,
    sample_size as n
FROM correlation_statistics
WHERE bonferroni_significant = TRUE
ORDER BY ABS(correlation) DESC;

-- Compare effect sizes
SELECT 
    effect_size,
    COUNT(*) as num_correlations,
    ROUND(AVG(ABS(correlation))::numeric, 3) as avg_abs_correlation,
    ROUND(MIN(ABS(correlation))::numeric, 3) as min_abs_correlation,
    ROUND(MAX(ABS(correlation))::numeric, 3) as max_abs_correlation
FROM correlation_statistics
GROUP BY effect_size
ORDER BY 
    CASE effect_size
        WHEN 'Large' THEN 1
        WHEN 'Medium' THEN 2
        WHEN 'Small' THEN 3
        WHEN 'Negligible' THEN 4
    END;

-- ============================================================================
-- 3. TIME SERIES ANALYSIS
-- ============================================================================

-- Daily statistics with depression context
SELECT 
    date,
    ROUND(avg_close::numeric, 2) as avg_stock_close,
    ROUND(sp500_close::numeric, 2) as sp500,
    depression_index,
    depression_word_count,
    CASE 
        WHEN depression_index >= 85 THEN 'High'
        WHEN depression_index >= 75 THEN 'Medium'
        WHEN depression_index >= 65 THEN 'Low'
        ELSE 'Very Low'
    END as depression_level,
    ROUND(price_range::numeric, 2) as volatility
FROM daily_merged_data
WHERE depression_index IS NOT NULL
ORDER BY date;

-- Monthly aggregates
SELECT 
    DATE_TRUNC('month', date) as month,
    ROUND(AVG(avg_close)::numeric, 2) as avg_stock_price,
    ROUND(AVG(sp500_close)::numeric, 2) as avg_sp500,
    ROUND(AVG(depression_index)::numeric, 1) as avg_depression_index,
    ROUND(AVG(depression_word_count)::numeric, 1) as avg_word_count,
    ROUND(AVG(price_range)::numeric, 2) as avg_volatility,
    COUNT(*) as trading_days
FROM daily_merged_data
WHERE date BETWEEN '2017-01-01' AND '2018-07-05'
GROUP BY DATE_TRUNC('month', date)
ORDER BY month;

-- High depression days (index > 85)
SELECT 
    date,
    depression_index,
    ROUND(avg_close::numeric, 2) as stock_price,
    ROUND(sp500_close::numeric, 2) as sp500,
    ROUND(price_range::numeric, 2) as volatility,
    ROUND(sp500_volatility::numeric, 4) as sp500_volatility
FROM daily_merged_data
WHERE depression_index > 85
ORDER BY depression_index DESC, date;

-- ============================================================================
-- 4. VOLATILITY ANALYSIS
-- ============================================================================

-- Volatility vs Depression correlation by month
SELECT 
    DATE_TRUNC('month', date) as month,
    ROUND(CORR(price_range, depression_index)::numeric, 4) as monthly_correlation,
    ROUND(AVG(price_range)::numeric, 2) as avg_volatility,
    ROUND(AVG(depression_index)::numeric, 1) as avg_depression,
    COUNT(*) as num_days
FROM daily_merged_data
WHERE price_range IS NOT NULL AND depression_index IS NOT NULL
GROUP BY DATE_TRUNC('month', date)
ORDER BY month;

-- Top 10 most volatile days
SELECT 
    date,
    ROUND(price_range::numeric, 2) as volatility,
    ROUND(sp500_volatility::numeric, 4) as sp500_volatility,
    depression_index,
    depression_word_count,
    ROUND(sp500_return::numeric, 4) as sp500_return
FROM daily_merged_data
ORDER BY price_range DESC NULLS LAST
LIMIT 10;

-- Volatility quintiles with depression
SELECT 
    volatility_quintile,
    ROUND(AVG(depression_index)::numeric, 2) as avg_depression,
    ROUND(AVG(volatility)::numeric, 2) as avg_volatility,
    COUNT(*) as days
FROM (
    SELECT 
        NTILE(5) OVER (ORDER BY price_range) as volatility_quintile,
        price_range as volatility,
        depression_index
    FROM daily_merged_data
    WHERE price_range IS NOT NULL AND depression_index IS NOT NULL
) subquery
GROUP BY volatility_quintile
ORDER BY volatility_quintile;

-- ============================================================================
-- 5. DEPRESSION METRICS COMPARISON
-- ============================================================================

-- Depression index vs word count correlation
SELECT 
    ROUND(CORR(depression_index, depression_word_count)::numeric, 6) as correlation,
    ROUND(AVG(depression_index)::numeric, 2) as avg_index,
    ROUND(STDDEV(depression_index)::numeric, 2) as stddev_index,
    ROUND(AVG(depression_word_count)::numeric, 2) as avg_word_count,
    ROUND(STDDEV(depression_word_count)::numeric, 2) as stddev_word_count,
    COUNT(*) as sample_size
FROM daily_merged_data
WHERE depression_index IS NOT NULL 
  AND depression_word_count IS NOT NULL;

-- Days with high word counts (> 50)
SELECT 
    date,
    depression_word_count,
    depression_index,
    ROUND(sp500_return::numeric, 4) as sp500_return,
    ROUND(sp500_volatility::numeric, 4) as sp500_volatility
FROM daily_merged_data
WHERE depression_word_count > 50
ORDER BY depression_word_count DESC;

-- ============================================================================
-- 6. INDUSTRY ANALYSIS (if industry_analysis table is populated)
-- ============================================================================

-- Top 10 most volatile industries
SELECT 
    industry,
    num_stocks_in_industry,
    ROUND(avg_volatility::numeric, 4) as avg_volatility,
    ROUND(avg_correlation_with_depression::numeric, 4) as corr_with_depression,
    volatility_rank
FROM industry_analysis
ORDER BY avg_volatility DESC NULLS LAST
LIMIT 10;

-- Industries with strongest depression correlation
SELECT 
    industry,
    ROUND(avg_correlation_with_depression::numeric, 4) as correlation,
    num_stocks_in_industry,
    ROUND(avg_volatility::numeric, 4) as volatility
FROM industry_analysis
WHERE avg_correlation_with_depression IS NOT NULL
ORDER BY ABS(avg_correlation_with_depression) DESC
LIMIT 10;

-- ============================================================================
-- 7. STOCK DATA QUERIES
-- ============================================================================

-- Most traded stocks (by volume)
SELECT 
    ticker,
    industry,
    ROUND(AVG(volume)::numeric, 0) as avg_daily_volume,
    ROUND(AVG(close_price)::numeric, 2) as avg_price,
    COUNT(DISTINCT date) as trading_days
FROM stock_data
WHERE volume IS NOT NULL
GROUP BY ticker, industry
ORDER BY avg_daily_volume DESC
LIMIT 20;

-- Stock price ranges (volatility proxy)
SELECT 
    ticker,
    industry,
    ROUND(AVG(high_price - low_price)::numeric, 2) as avg_daily_range,
    ROUND(AVG(close_price)::numeric, 2) as avg_price,
    ROUND((AVG(high_price - low_price) / AVG(close_price) * 100)::numeric, 2) 
        as pct_range
FROM stock_data
WHERE high_price IS NOT NULL 
  AND low_price IS NOT NULL 
  AND close_price > 0
GROUP BY ticker, industry
HAVING COUNT(*) > 100  -- At least 100 trading days
ORDER BY pct_range DESC
LIMIT 20;

-- ============================================================================
-- 8. RAINFALL IMPACT ANALYSIS
-- ============================================================================

-- Rainfall vs stock metrics
SELECT 
    ROUND(CORR(rainfall_mm, avg_close)::numeric, 6) as rainfall_stock_corr,
    ROUND(CORR(rainfall_mm, sp500_close)::numeric, 6) as rainfall_sp500_corr,
    ROUND(CORR(rainfall_mm, depression_index)::numeric, 6) as rainfall_depression_corr,
    ROUND(AVG(rainfall_mm)::numeric, 2) as avg_rainfall,
    COUNT(*) as days
FROM daily_merged_data
WHERE rainfall_mm IS NOT NULL;

-- Days with significant rainfall (> 10mm)
SELECT 
    date,
    ROUND(rainfall_mm::numeric, 2) as rainfall,
    ROUND(sp500_return::numeric, 4) as sp500_return,
    ROUND(sp500_volatility::numeric, 4) as volatility,
    depression_index
FROM daily_merged_data
WHERE rainfall_mm > 10
ORDER BY rainfall_mm DESC;

-- ============================================================================
-- 9. STATISTICAL SUMMARIES
-- ============================================================================

-- Overall descriptive statistics
SELECT 
    'Stock Close' as metric,
    ROUND(AVG(avg_close)::numeric, 2) as mean,
    ROUND(STDDEV(avg_close)::numeric, 2) as std_dev,
    ROUND(MIN(avg_close)::numeric, 2) as min,
    ROUND(MAX(avg_close)::numeric, 2) as max,
    COUNT(*) as n
FROM daily_merged_data WHERE avg_close IS NOT NULL
UNION ALL
SELECT 'S&P 500', 
    ROUND(AVG(sp500_close)::numeric, 2),
    ROUND(STDDEV(sp500_close)::numeric, 2),
    ROUND(MIN(sp500_close)::numeric, 2),
    ROUND(MAX(sp500_close)::numeric, 2),
    COUNT(*)
FROM daily_merged_data WHERE sp500_close IS NOT NULL
UNION ALL
SELECT 'Depression Index',
    ROUND(AVG(depression_index)::numeric, 2),
    ROUND(STDDEV(depression_index)::numeric, 2),
    MIN(depression_index),
    MAX(depression_index),
    COUNT(*)
FROM daily_merged_data WHERE depression_index IS NOT NULL
UNION ALL
SELECT 'Price Range (Volatility)',
    ROUND(AVG(price_range)::numeric, 2),
    ROUND(STDDEV(price_range)::numeric, 2),
    ROUND(MIN(price_range)::numeric, 2),
    ROUND(MAX(price_range)::numeric, 2),
    COUNT(*)
FROM daily_merged_data WHERE price_range IS NOT NULL;

-- ============================================================================
-- 10. ADVANCED QUERIES
-- ============================================================================

-- Rolling 30-day correlation (requires window functions)
WITH rolling_data AS (
    SELECT 
        date,
        price_range,
        depression_index,
        COUNT(*) OVER w as window_size
    FROM daily_merged_data
    WHERE price_range IS NOT NULL AND depression_index IS NOT NULL
    WINDOW w AS (ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW)
)
SELECT 
    date,
    window_size,
    ROUND(price_range::numeric, 2) as volatility,
    depression_index
FROM rolling_data
WHERE window_size = 30
ORDER BY date;

-- Volatility regime analysis
WITH regimes AS (
    SELECT 
        date,
        price_range,
        depression_index,
        CASE 
            WHEN price_range >= PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY price_range) 
                OVER () THEN 'High Volatility'
            WHEN price_range >= PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY price_range) 
                OVER () THEN 'Medium Volatility'
            ELSE 'Low Volatility'
        END as regime
    FROM daily_merged_data
    WHERE price_range IS NOT NULL AND depression_index IS NOT NULL
)
SELECT 
    regime,
    ROUND(AVG(depression_index)::numeric, 2) as avg_depression,
    ROUND(AVG(price_range)::numeric, 2) as avg_volatility,
    COUNT(*) as days
FROM regimes
GROUP BY regime
ORDER BY avg_volatility DESC;

-- ============================================================================
-- 11. USING VIEWS
-- ============================================================================

-- Important correlations (pre-filtered view)
SELECT * FROM v_important_correlations;

-- Significant correlations (pre-filtered view)
SELECT * FROM v_significant_correlations
ORDER BY p_value;

-- Daily analysis view
SELECT * FROM v_daily_analysis
WHERE date BETWEEN '2017-01-01' AND '2017-12-31'
ORDER BY date;

-- Depression metrics view
SELECT * FROM v_depression_metrics
WHERE depression_index > 80
ORDER BY depression_index DESC;

-- ============================================================================
-- EXPORT QUERIES
-- ============================================================================

-- Export to CSV (use \copy in psql)
-- \copy (SELECT * FROM v_important_correlations) TO 'important_correlations.csv' CSV HEADER;
-- \copy (SELECT * FROM daily_merged_data) TO 'daily_data_export.csv' CSV HEADER;

-- ============================================================================
-- END OF QUERY EXAMPLES
-- ============================================================================
