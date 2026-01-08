-- ============================================================================
-- PostgreSQL Schema for Time Series Depression-Stock Analysis
-- ============================================================================
-- This schema stores all data and results for the depression-stock analysis
-- Including: raw data, merged data, correlation results, and statistical tests
-- ============================================================================

-- Drop existing tables if they exist (use with caution in production)
DROP TABLE IF EXISTS correlation_statistics CASCADE;
DROP TABLE IF EXISTS daily_merged_data CASCADE;
DROP TABLE IF EXISTS depression_word_count CASCADE;
DROP TABLE IF EXISTS depression_index CASCADE;
DROP TABLE IF EXISTS rainfall_data CASCADE;
DROP TABLE IF EXISTS sp500_data CASCADE;
DROP TABLE IF EXISTS stock_data CASCADE;

-- ============================================================================
-- 1. RAW DATA TABLES
-- ============================================================================

-- Stock data from Wikipedia (individual stocks)
CREATE TABLE stock_data (
    stock_id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    industry VARCHAR(100),
    open_price NUMERIC(12, 4),
    high_price NUMERIC(12, 4),
    low_price NUMERIC(12, 4),
    close_price NUMERIC(12, 4),
    volume BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT stock_date_ticker_unique UNIQUE (date, ticker)
);

-- Create index for faster queries
CREATE INDEX idx_stock_date ON stock_data(date);
CREATE INDEX idx_stock_ticker ON stock_data(ticker);
CREATE INDEX idx_stock_industry ON stock_data(industry);
CREATE INDEX idx_stock_date_ticker ON stock_data(date, ticker);

-- S&P 500 data
CREATE TABLE sp500_data (
    sp500_id SERIAL PRIMARY KEY,
    date DATE NOT NULL UNIQUE,
    sp500_open NUMERIC(12, 4),
    sp500_high NUMERIC(12, 4),
    sp500_low NUMERIC(12, 4),
    sp500_close NUMERIC(12, 4),
    sp500_volume BIGINT,
    sp500_return NUMERIC(10, 6),      -- Daily return
    sp500_volatility NUMERIC(10, 6),  -- Price range / close
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_sp500_date ON sp500_data(date);

-- Depression Index from Google Trends (weekly data)
CREATE TABLE depression_index (
    index_id SERIAL PRIMARY KEY,
    week_start_date DATE NOT NULL UNIQUE,
    depression_index INTEGER NOT NULL CHECK (depression_index >= 0 AND depression_index <= 100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_depression_index_date ON depression_index(week_start_date);

-- Depression Word Count from news (daily data)
CREATE TABLE depression_word_count (
    count_id SERIAL PRIMARY KEY,
    date DATE NOT NULL UNIQUE,
    word_count INTEGER NOT NULL CHECK (word_count >= 0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_depression_count_date ON depression_word_count(date);

-- Rainfall data
CREATE TABLE rainfall_data (
    rainfall_id SERIAL PRIMARY KEY,
    date DATE NOT NULL UNIQUE,
    rainfall_mm NUMERIC(8, 2) CHECK (rainfall_mm >= 0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_rainfall_date ON rainfall_data(date);

-- ============================================================================
-- 2. AGGREGATED/MERGED DATA TABLE
-- ============================================================================

-- Daily merged dataset (this is your main analysis table)
CREATE TABLE daily_merged_data (
    merged_id SERIAL PRIMARY KEY,
    date DATE NOT NULL UNIQUE,
    
    -- Foreign key references (explicit columns for relationships)
    sp500_id INTEGER,
    rainfall_id INTEGER,
    depression_count_id INTEGER,
    
    -- Stock aggregates (daily averages across all stocks)
    avg_open NUMERIC(12, 4),
    avg_high NUMERIC(12, 4),
    avg_low NUMERIC(12, 4),
    avg_close NUMERIC(12, 4),
    avg_volume BIGINT,
    price_range NUMERIC(12, 4),        -- avg_high - avg_low
    
    -- S&P 500 metrics
    sp500_close NUMERIC(12, 4),
    sp500_return NUMERIC(10, 6),
    sp500_volatility NUMERIC(10, 6),
    
    -- Depression metrics (index is forward-filled from weekly)
    depression_index INTEGER,
    depression_word_count INTEGER,
    
    -- Rainfall
    rainfall_mm NUMERIC(8, 2),
    
    -- Metadata
    num_stocks_traded INTEGER,         -- Number of stocks with data on this date
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign key constraints for proper relationships
    CONSTRAINT fk_daily_sp500 FOREIGN KEY (sp500_id) REFERENCES sp500_data(sp500_id),
    CONSTRAINT fk_daily_rainfall FOREIGN KEY (rainfall_id) REFERENCES rainfall_data(rainfall_id),
    CONSTRAINT fk_daily_depression_count FOREIGN KEY (depression_count_id) REFERENCES depression_word_count(count_id)
);

CREATE INDEX idx_merged_date ON daily_merged_data(date);

-- ============================================================================
-- 3. ANALYSIS RESULTS TABLES
-- ============================================================================

-- Correlation statistics (stores all correlation analysis results)
CREATE TABLE correlation_statistics (
    stat_id SERIAL PRIMARY KEY,
    variable_x VARCHAR(100) NOT NULL,
    variable_y VARCHAR(100) NOT NULL,
    
    -- Correlation metrics
    correlation NUMERIC(10, 6) NOT NULL,
    p_value NUMERIC(15, 10) NOT NULL,
    r_squared NUMERIC(10, 6),           -- Variance explained
    
    -- Confidence intervals
    ci_lower NUMERIC(10, 6),
    ci_upper NUMERIC(10, 6),
    
    -- Sample characteristics
    sample_size INTEGER NOT NULL,
    date_start DATE,
    date_end DATE,
    
    -- Statistical interpretation
    significance_level VARCHAR(20),     -- '***', '**', '*', 'ns'
    effect_size VARCHAR(20),            -- 'Large', 'Medium', 'Small', 'Negligible'
    bonferroni_significant BOOLEAN,     -- Passes multiple comparison correction
    
    -- Importance scoring
    importance_score INTEGER,           -- 0-10 scale
    
    -- Metadata
    analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT,
    
    CONSTRAINT correlation_unique UNIQUE (variable_x, variable_y),
    
    -- Foreign key reference to daily data for date range validation
    CONSTRAINT fk_corr_date_start FOREIGN KEY (date_start) REFERENCES daily_merged_data(date),
    CONSTRAINT fk_corr_date_end FOREIGN KEY (date_end) REFERENCES daily_merged_data(date)
);

CREATE INDEX idx_corr_variables ON correlation_statistics(variable_x, variable_y);
CREATE INDEX idx_corr_significance ON correlation_statistics(bonferroni_significant);
CREATE INDEX idx_corr_importance ON correlation_statistics(importance_score DESC);

-- ============================================================================
-- 4. INDUSTRY-SPECIFIC ANALYSIS TABLE
-- ============================================================================

CREATE TABLE industry_analysis (
    analysis_id SERIAL PRIMARY KEY,
    industry VARCHAR(100) NOT NULL,
    
    -- Industry statistics
    avg_correlation_with_depression NUMERIC(10, 6),
    num_stocks_in_industry INTEGER,
    avg_volatility NUMERIC(10, 6),
    avg_return NUMERIC(10, 6),
    
    -- Time period
    date_start DATE,
    date_end DATE,
    
    -- Ranking
    volatility_rank INTEGER,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT industry_period_unique UNIQUE (industry, date_start, date_end),
    
    -- Foreign key references for date validation
    CONSTRAINT fk_industry_date_start FOREIGN KEY (date_start) REFERENCES daily_merged_data(date),
    CONSTRAINT fk_industry_date_end FOREIGN KEY (date_end) REFERENCES daily_merged_data(date)
);

CREATE INDEX idx_industry_name ON industry_analysis(industry);
CREATE INDEX idx_industry_volatility ON industry_analysis(avg_volatility DESC);

-- ============================================================================
-- 4. JUNCTION TABLES FOR MANY-TO-MANY RELATIONSHIPS
-- ============================================================================

-- Stock-Industry relationship (Many-to-Many)
-- A stock can belong to multiple industries, an industry can have multiple stocks
CREATE TABLE stock_industry (
    stock_industry_id SERIAL PRIMARY KEY,
    stock_id INTEGER NOT NULL,
    industry_id INTEGER NOT NULL,
    is_primary_industry BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT fk_stock_industry_stock FOREIGN KEY (stock_id) REFERENCES stock_data(stock_id),
    CONSTRAINT fk_stock_industry_analysis FOREIGN KEY (industry_id) REFERENCES industry_analysis(analysis_id),
    CONSTRAINT unique_stock_industry UNIQUE (stock_id, industry_id)
);

-- Daily Stock Performance (One-to-Many)
-- Each stock can have multiple daily records, each daily record belongs to one stock
CREATE TABLE daily_stock_performance (
    performance_id SERIAL PRIMARY KEY,
    merged_data_id INTEGER NOT NULL,
    stock_id INTEGER NOT NULL,
    daily_return NUMERIC(10, 6),
    daily_volatility NUMERIC(10, 6),
    relative_to_sp500 NUMERIC(10, 6),
    
    CONSTRAINT fk_performance_merged FOREIGN KEY (merged_data_id) REFERENCES daily_merged_data(merged_id),
    CONSTRAINT fk_performance_stock FOREIGN KEY (stock_id) REFERENCES stock_data(stock_id),
    CONSTRAINT unique_daily_stock UNIQUE (merged_data_id, stock_id)
);

CREATE INDEX idx_stock_industry_stock ON stock_industry(stock_id);
CREATE INDEX idx_stock_industry_industry ON stock_industry(industry_id);
CREATE INDEX idx_performance_date_stock ON daily_stock_performance(merged_data_id, stock_id);

-- ============================================================================
-- 5. VIEWS FOR COMMON QUERIES
-- ============================================================================

-- View: Most important correlations (score >= 8)
CREATE OR REPLACE VIEW v_important_correlations AS
SELECT 
    variable_x,
    variable_y,
    correlation,
    p_value,
    r_squared,
    significance_level,
    effect_size,
    bonferroni_significant,
    importance_score,
    sample_size
FROM correlation_statistics
WHERE importance_score >= 8
ORDER BY importance_score DESC, ABS(correlation) DESC;

-- View: Significant correlations only
CREATE OR REPLACE VIEW v_significant_correlations AS
SELECT 
    variable_x,
    variable_y,
    correlation,
    p_value,
    r_squared,
    ci_lower,
    ci_upper,
    significance_level,
    bonferroni_significant
FROM correlation_statistics
WHERE p_value < 0.05
ORDER BY p_value ASC;

-- View: Daily data with all metrics
CREATE OR REPLACE VIEW v_daily_analysis AS
SELECT 
    date,
    avg_close,
    price_range,
    sp500_close,
    sp500_return,
    sp500_volatility,
    depression_index,
    depression_word_count,
    rainfall_mm,
    num_stocks_traded
FROM daily_merged_data
ORDER BY date;

-- View: Depression metrics comparison
CREATE OR REPLACE VIEW v_depression_metrics AS
SELECT 
    date,
    depression_index,
    depression_word_count,
    sp500_close,
    sp500_volatility,
    price_range
FROM daily_merged_data
WHERE depression_index IS NOT NULL
ORDER BY date;

-- ============================================================================
-- 6. HELPER FUNCTIONS
-- ============================================================================

-- Function to calculate correlation between two columns
CREATE OR REPLACE FUNCTION calculate_correlation(
    table_name TEXT,
    col_x TEXT,
    col_y TEXT
) RETURNS NUMERIC AS $$
DECLARE
    correlation_value NUMERIC;
BEGIN
    EXECUTE format('
        SELECT CORR(%I, %I)::NUMERIC(10,6)
        FROM %I
        WHERE %I IS NOT NULL AND %I IS NOT NULL
    ', col_x, col_y, table_name, col_x, col_y)
    INTO correlation_value;
    
    RETURN correlation_value;
END;
$$ LANGUAGE plpgsql;

-- Function to get date range for a table
CREATE OR REPLACE FUNCTION get_date_range(
    table_name TEXT,
    date_column TEXT DEFAULT 'date'
) RETURNS TABLE(min_date DATE, max_date DATE, num_days INTEGER) AS $$
BEGIN
    RETURN QUERY EXECUTE format('
        SELECT 
            MIN(%I)::DATE,
            MAX(%I)::DATE,
            COUNT(DISTINCT %I)::INTEGER
        FROM %I
    ', date_column, date_column, date_column, table_name);
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- 7. SAMPLE QUERIES (COMMENTED OUT - FOR REFERENCE)
-- ============================================================================

/*
-- Get all highly significant correlations
SELECT * FROM v_important_correlations;

-- Get correlations that survive Bonferroni correction
SELECT * FROM correlation_statistics 
WHERE bonferroni_significant = TRUE
ORDER BY ABS(correlation) DESC;

-- Get depression metrics over time
SELECT * FROM v_depression_metrics
WHERE date BETWEEN '2017-01-01' AND '2018-07-05';

-- Calculate correlation between price_range and depression_index
SELECT calculate_correlation('daily_merged_data', 'price_range', 'depression_index');

-- Get date range for merged data
SELECT * FROM get_date_range('daily_merged_data');

-- Get top 10 most volatile industries
SELECT * FROM industry_analysis
ORDER BY avg_volatility DESC
LIMIT 10;

-- Daily stats with depression context
SELECT 
    date,
    avg_close,
    sp500_close,
    depression_index,
    CASE 
        WHEN depression_index >= 85 THEN 'High'
        WHEN depression_index >= 75 THEN 'Medium'
        ELSE 'Low'
    END as depression_level
FROM daily_merged_data
WHERE depression_index IS NOT NULL
ORDER BY date;

-- Correlation between stock volatility and depression by month
SELECT 
    DATE_TRUNC('month', date) as month,
    CORR(price_range, depression_index) as monthly_correlation,
    AVG(price_range) as avg_volatility,
    AVG(depression_index) as avg_depression,
    COUNT(*) as num_days
FROM daily_merged_data
WHERE price_range IS NOT NULL AND depression_index IS NOT NULL
GROUP BY DATE_TRUNC('month', date)
ORDER BY month;
*/

-- ============================================================================
-- 8. TABLE COMMENTS (DOCUMENTATION)
-- ============================================================================

COMMENT ON TABLE stock_data IS 'Individual stock daily OHLCV data from Wikipedia';
COMMENT ON TABLE sp500_data IS 'S&P 500 index daily data with calculated returns and volatility';
COMMENT ON TABLE depression_index IS 'Google Trends weekly depression search interest index (0-100)';
COMMENT ON TABLE depression_word_count IS 'Daily count of depression-related words in news articles';
COMMENT ON TABLE rainfall_data IS 'Daily rainfall measurements in millimeters';
COMMENT ON TABLE daily_merged_data IS 'Main analysis table with all metrics merged by date';
COMMENT ON TABLE correlation_statistics IS 'Statistical correlation analysis results with significance testing';
COMMENT ON TABLE industry_analysis IS 'Industry-level aggregated statistics and rankings';

COMMENT ON COLUMN daily_merged_data.price_range IS 'Average (high - low) across all stocks, proxy for market volatility';
COMMENT ON COLUMN daily_merged_data.depression_index IS 'Forward-filled from weekly Google Trends data';
COMMENT ON COLUMN correlation_statistics.bonferroni_significant IS 'TRUE if p-value < 0.05/17 (survives multiple comparison correction)';
COMMENT ON COLUMN correlation_statistics.importance_score IS '0-10 scale based on p-value, effect size, and Bonferroni correction';

-- ============================================================================
-- 9. FOREIGN KEY RELATIONSHIPS SUMMARY
-- ============================================================================

/*
RELATIONSHIP DIAGRAM:

sp500_data (date) ←──── daily_merged_data (date)
rainfall_data (date) ←──┘
depression_word_count (date) ←──┘

daily_merged_data (date) ←──── correlation_statistics (date_start, date_end)
                              ↑
daily_merged_data (date) ←──── industry_analysis (date_start, date_end)

stock_data: Independent table (connected via date joins in queries)
depression_index: Independent table (weekly data, connected via date joins)

KEY RELATIONSHIPS:
1. daily_merged_data is the central fact table
2. sp500_data, rainfall_data, depression_word_count provide daily metrics
3. correlation_statistics and industry_analysis reference date ranges from daily_merged_data
4. stock_data and depression_index are joined by date in analytical queries
*/

-- ============================================================================
-- SCHEMA CREATION COMPLETE
-- ============================================================================

-- Display success message
DO $$
BEGIN
    RAISE NOTICE '✓ Schema created successfully!';
    RAISE NOTICE '✓ Tables: 7 (5 raw data + 1 merged + 1 results + 1 industry)';
    RAISE NOTICE '✓ Views: 4 (important correlations, significant, daily, depression)';
    RAISE NOTICE '✓ Functions: 2 (calculate_correlation, get_date_range)';
    RAISE NOTICE '✓ Ready to load data from CSV files';
    RAISE NOTICE '';
    RAISE NOTICE 'Next steps:';
    RAISE NOTICE '1. Load data using: psql -d your_database -f load_data.sql';
    RAISE NOTICE '2. Run analysis queries from the comments above';
    RAISE NOTICE '3. Use views for quick access to important results';
END $$;
