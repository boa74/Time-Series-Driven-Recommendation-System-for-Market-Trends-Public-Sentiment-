======================================================================
POSTGRESQL DATABASE EXPORTS - TUTORIAL_DB
======================================================================

This folder contains CSV exports from the tutorial_db PostgreSQL database.

FILES INCLUDED:
----------------------------------------------------------------------
• depression_index.csv
  Description: Depression index data by date

• ccnews_depression.csv
  Description: News articles related to depression

• stock_data.csv
  Description: Stock market data (normalized format)

• sp500.csv
  Description: S&P 500 index data

• rainfall.csv
  Description: Rainfall data by US state


======================================================================
HOW TO IMPORT INTO PGADMIN:
======================================================================
1. Open pgAdmin and connect to your PostgreSQL server
2. Create a new database called 'tutorial_db' (if it doesn't exist)
3. Right-click on the database → Query Tool
4. Create tables using the schema below
5. Right-click on each table → Import/Export Data
6. Select the corresponding CSV file
7. Set format to CSV with header row enabled

======================================================================
TABLE SCHEMAS (CREATE THESE FIRST):
======================================================================

-- 1. Depression Index Table
CREATE TABLE depression_index (
    date DATE PRIMARY KEY,
    depression_index FLOAT
);

-- 2. News Articles Table
CREATE TABLE ccnews_depression (
    date VARCHAR(20),
    title TEXT,
    text TEXT
);

-- 3. Stock Data Table (Normalized)
CREATE TABLE stock_data (
    date VARCHAR(20),
    ticker VARCHAR(10),
    open FLOAT,
    high FLOAT,
    low FLOAT,
    close FLOAT,
    volume FLOAT
);

-- 4. S&P 500 Table
CREATE TABLE sp500 (
    date VARCHAR(20),
    close_gspc FLOAT,
    high_gspc FLOAT,
    low_gspc FLOAT,
    open_gspc FLOAT,
    volume_gspc FLOAT,
    return FLOAT,
    volatility_7 FLOAT
);

-- 5. Rainfall Table
CREATE TABLE rainfall (
    date VARCHAR(20),
    -- Add columns for each state (51 columns total)
    -- See the CSV header for complete column list
);

======================================================================
ALTERNATIVE: QUICK IMPORT WITH PSQL COMMAND
======================================================================
If you have psql command-line tool:

psql -U postgres -d tutorial_db -c "\COPY depression_index FROM 'depression_index.csv' CSV HEADER"
psql -U postgres -d tutorial_db -c "\COPY ccnews_depression FROM 'ccnews_depression.csv' CSV HEADER"
psql -U postgres -d tutorial_db -c "\COPY stock_data FROM 'stock_data.csv' CSV HEADER"
psql -U postgres -d tutorial_db -c "\COPY sp500 FROM 'sp500.csv' CSV HEADER"
psql -U postgres -d tutorial_db -c "\COPY rainfall FROM 'rainfall.csv' CSV HEADER"

