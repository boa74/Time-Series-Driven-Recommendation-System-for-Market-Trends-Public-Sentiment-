-- ============================================================================
-- NORMALIZED DATABASE SCHEMA FOR TIME-SERIES ANALYSIS
-- ============================================================================
-- This SQL script creates a normalized database structure with:
-- 1. Dimension tables (company_info, date_dimension)
-- 2. Fact tables (stock_prices, sp500_index, depression_news, depression_index, rainfall)
-- 3. Proper relationships and foreign keys
-- 4. Indexes for performance
-- ============================================================================

-- Drop existing tables if they exist (be careful in production!)
DROP TABLE IF EXISTS stock_prices CASCADE;
DROP TABLE IF EXISTS sp500_index CASCADE;
DROP TABLE IF EXISTS depression_news CASCADE;
DROP TABLE IF EXISTS depression_index CASCADE;
DROP TABLE IF EXISTS rainfall CASCADE;
DROP TABLE IF EXISTS company_info CASCADE;
DROP TABLE IF EXISTS date_dimension CASCADE;

-- ============================================================================
-- DIMENSION TABLE: date_dimension
-- ============================================================================
-- Central date table for all time-series data
CREATE TABLE date_dimension (
    date DATE PRIMARY KEY,
    year INTEGER NOT NULL,
    month INTEGER NOT NULL,
    day INTEGER NOT NULL,
    quarter INTEGER NOT NULL,
    day_of_week INTEGER NOT NULL,  -- 0=Monday, 6=Sunday
    day_name VARCHAR(10),
    month_name VARCHAR(10),
    is_weekend BOOLEAN,
    CONSTRAINT check_month CHECK (month BETWEEN 1 AND 12),
    CONSTRAINT check_day CHECK (day BETWEEN 1 AND 31),
    CONSTRAINT check_quarter CHECK (quarter BETWEEN 1 AND 4),
    CONSTRAINT check_day_of_week CHECK (day_of_week BETWEEN 0 AND 6)
);

CREATE INDEX idx_date_year ON date_dimension(year);
CREATE INDEX idx_date_year_month ON date_dimension(year, month);
CREATE INDEX idx_date_quarter ON date_dimension(quarter);

COMMENT ON TABLE date_dimension IS 'Central date dimension for all time-series data';

-- ============================================================================
-- DIMENSION TABLE: company_info
-- ============================================================================
-- S&P 500 company information from Wikipedia
CREATE TABLE company_info (
    ticker VARCHAR(10) PRIMARY KEY,
    company_name VARCHAR(255) NOT NULL,
    sector VARCHAR(100),
    industry VARCHAR(100),
    CONSTRAINT ticker_format CHECK (ticker ~ '^[A-Z]{1,5}$')
);

CREATE INDEX idx_company_sector ON company_info(sector);
CREATE INDEX idx_company_industry ON company_info(industry);

COMMENT ON TABLE company_info IS 'S&P 500 company information including ticker, name, sector, and industry';

-- ============================================================================
-- FACT TABLE: stock_prices
-- ============================================================================
-- Daily stock prices for each ticker
CREATE TABLE stock_prices (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL REFERENCES date_dimension(date),
    ticker VARCHAR(10) NOT NULL REFERENCES company_info(ticker),
    open DECIMAL(12, 2),
    high DECIMAL(12, 2),
    low DECIMAL(12, 2),
    close DECIMAL(12, 2),
    volume BIGINT,
    CONSTRAINT stock_prices_unique UNIQUE (date, ticker),
    CONSTRAINT positive_prices CHECK (open > 0 AND high > 0 AND low > 0 AND close > 0),
    CONSTRAINT logical_prices CHECK (high >= low AND high >= open AND high >= close AND low <= open AND low <= close)
);

CREATE INDEX idx_stock_date ON stock_prices(date);
CREATE INDEX idx_stock_ticker ON stock_prices(ticker);
CREATE INDEX idx_stock_date_ticker ON stock_prices(date, ticker);

COMMENT ON TABLE stock_prices IS 'Daily stock prices for individual tickers with company information';

-- ============================================================================
-- FACT TABLE: sp500_index
-- ============================================================================
-- Daily S&P 500 index values and metrics
CREATE TABLE sp500_index (
    date DATE PRIMARY KEY REFERENCES date_dimension(date),
    close DECIMAL(12, 2) NOT NULL,
    open DECIMAL(12, 2),
    high DECIMAL(12, 2),
    low DECIMAL(12, 2),
    volume BIGINT,
    return DECIMAL(10, 8),
    volatility_7d DECIMAL(10, 8),
    CONSTRAINT positive_sp500_prices CHECK (close > 0)
);

CREATE INDEX idx_sp500_date ON sp500_index(date);

COMMENT ON TABLE sp500_index IS 'Daily S&P 500 index values including price, return, and volatility metrics';

-- ============================================================================
-- FACT TABLE: depression_news
-- ============================================================================
-- Daily count of depression mentions in news articles
CREATE TABLE depression_news (
    date DATE PRIMARY KEY REFERENCES date_dimension(date),
    depression_word_count INTEGER NOT NULL DEFAULT 0,
    total_articles INTEGER NOT NULL DEFAULT 0,
    avg_depression_per_article DECIMAL(10, 2),
    CONSTRAINT non_negative_counts CHECK (depression_word_count >= 0 AND total_articles >= 0)
);

CREATE INDEX idx_depression_news_date ON depression_news(date);

COMMENT ON TABLE depression_news IS 'Daily aggregated count of depression mentions in news articles';

-- ============================================================================
-- FACT TABLE: depression_index
-- ============================================================================
-- Google Trends depression index (weekly data)
CREATE TABLE depression_index (
    date DATE PRIMARY KEY REFERENCES date_dimension(date),
    depression_index INTEGER NOT NULL,
    CONSTRAINT index_range CHECK (depression_index BETWEEN 0 AND 100)
);

CREATE INDEX idx_depression_index_date ON depression_index(date);

COMMENT ON TABLE depression_index IS 'Google Trends depression index (0-100 scale)';

-- ============================================================================
-- FACT TABLE: rainfall
-- ============================================================================
-- Daily average rainfall across all US states
CREATE TABLE rainfall (
    date DATE PRIMARY KEY REFERENCES date_dimension(date),
    avg_rainfall DECIMAL(10, 2) NOT NULL DEFAULT 0,
    CONSTRAINT non_negative_rainfall CHECK (avg_rainfall >= 0)
);

CREATE INDEX idx_rainfall_date ON rainfall(date);

COMMENT ON TABLE rainfall IS 'Daily average rainfall across all US states';

-- ============================================================================
-- POPULATE DATE DIMENSION
-- ============================================================================
-- Populate date_dimension with dates from 2014-2024
INSERT INTO date_dimension (date, year, month, day, quarter, day_of_week, day_name, month_name, is_weekend)
SELECT 
    d::date,
    EXTRACT(YEAR FROM d)::INTEGER,
    EXTRACT(MONTH FROM d)::INTEGER,
    EXTRACT(DAY FROM d)::INTEGER,
    EXTRACT(QUARTER FROM d)::INTEGER,
    EXTRACT(DOW FROM d)::INTEGER,
    TO_CHAR(d, 'Day'),
    TO_CHAR(d, 'Month'),
    CASE WHEN EXTRACT(DOW FROM d) IN (0, 6) THEN TRUE ELSE FALSE END
FROM generate_series('2014-01-01'::date, '2024-12-31'::date, '1 day'::interval) d;

-- ============================================================================
-- VIEWS FOR EASY QUERYING
-- ============================================================================

-- View: Complete daily data (all metrics joined)
CREATE OR REPLACE VIEW daily_analysis_view AS
SELECT 
    dd.date,
    dd.year,
    dd.month,
    dd.quarter,
    dd.day_of_week,
    sp.close AS sp500_close,
    sp.return AS sp500_return,
    sp.volatility_7d AS sp500_volatility,
    dn.depression_word_count,
    dn.total_articles AS depression_articles,
    dn.avg_depression_per_article,
    di.depression_index,
    r.avg_rainfall,
    AVG(st.close) AS avg_stock_close,
    AVG(st.volume) AS avg_stock_volume
FROM date_dimension dd
LEFT JOIN sp500_index sp ON dd.date = sp.date
LEFT JOIN depression_news dn ON dd.date = dn.date
LEFT JOIN depression_index di ON dd.date = di.date
LEFT JOIN rainfall r ON dd.date = r.date
LEFT JOIN stock_prices st ON dd.date = st.date
GROUP BY dd.date, dd.year, dd.month, dd.quarter, dd.day_of_week,
         sp.close, sp.return, sp.volatility_7d,
         dn.depression_word_count, dn.total_articles, dn.avg_depression_per_article,
         di.depression_index, r.avg_rainfall
ORDER BY dd.date;

COMMENT ON VIEW daily_analysis_view IS 'Complete daily view with all metrics for time-series analysis';

-- View: Stock prices with company information
CREATE OR REPLACE VIEW stock_with_company_view AS
SELECT 
    st.date,
    st.ticker,
    c.company_name,
    c.sector,
    c.industry,
    st.open,
    st.high,
    st.low,
    st.close,
    st.volume
FROM stock_prices st
JOIN company_info c ON st.ticker = c.ticker
ORDER BY st.date, st.ticker;

COMMENT ON VIEW stock_with_company_view IS 'Stock prices enriched with company information';

-- ============================================================================
-- SUMMARY
-- ============================================================================

SELECT 'Schema created successfully!' AS status;

-- Show table counts
SELECT 
    'Tables Created' AS info,
    COUNT(*) AS count
FROM information_schema.tables 
WHERE table_schema = 'public' 
  AND table_type = 'BASE TABLE';

-- Show views created
SELECT 
    'Views Created' AS info,
    COUNT(*) AS count
FROM information_schema.views 
WHERE table_schema = 'public';

COMMENT ON SCHEMA public IS 'Normalized schema for time-series analysis of stock prices, depression metrics, and weather data';



