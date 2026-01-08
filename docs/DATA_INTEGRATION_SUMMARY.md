# Data Integration Summary

## Overview
This document describes the two-pronged approach for data integration: a **merged CSV for analysis** and a **normalized SQL schema for database management**.

---

## 1. Python Merge Script ✓ COMPLETE

**File:** `merge_all_datasets.py`

### What it does:
Merges 5 datasets into a single denormalized CSV file for time-series analysis.

### Input Files:
1. **stock_data_wiki.csv** - Stock prices with company info (2,017,538 rows)
2. **sp500.csv** - S&P 500 index data (4,019 rows)  
3. **ccnews_depression_daily_count_final.csv** - Depression news mentions (206 rows)
4. **depression_index.csv** - Google Trends depression index (332 rows)
5. **rainfall.csv** - Rainfall data averaged across 50 US states (4,019 rows)

### Output File:
**`csv_exports/merged_analysis_data.csv`**
- **4,019 rows** (one row per date from 2014-01-01 to 2025-01-01)
- **18 columns** including all metrics + date features

### Columns in Merged Data:
1. `date` - Date
2. `avg_stock_open` - Average opening price across all stocks
3. `avg_stock_high` - Average high price
4. `avg_stock_low` - Average low price
5. `avg_stock_close` - Average closing price
6. `total_stock_volume` - Total trading volume
7. `sp500_close` - S&P 500 closing price
8. `sp500_return` - S&P 500 daily return
9. `sp500_volatility_7d` - 7-day volatility
10. `avg_rainfall` - Average rainfall across all states
11. `depression_index` - Google Trends index (0-100)
12. `depression_word_count` - Total depression mentions in news
13. `total_articles` - Number of articles analyzed
14. `avg_depression_per_article` - Average mentions per article
15. `year` - Year
16. `month` - Month (1-12)
17. `day_of_week` - Day of week (0=Monday, 6=Sunday)
18. `quarter` - Quarter (1-4)

### Missing Data:
- **Depression Index:** 92% missing (only weekly data from Google Trends)
- **Depression News:** 94.9% missing (only covers 2017-01-01 to 2018-07-05)

### Usage:
```bash
python merge_all_datasets.py
```

---

## 2. SQL Normalized Schema

**File:** `create_normalized_schema.sql`

### What it does:
Creates a normalized database structure with proper relationships, foreign keys, and indexes.

### Database Structure:

#### Dimension Tables:
1. **`date_dimension`** - Central date table (2014-2024)
   - Primary Key: `date`
   - Includes: year, month, quarter, day_of_week, is_weekend

2. **`company_info`** - S&P 500 company information  
   - Primary Key: `ticker`
   - Columns: company_name, sector, industry

#### Fact Tables:
1. **`stock_prices`** - Daily stock prices per ticker
   - Foreign Keys: date → date_dimension, ticker → company_info
   - Columns: open, high, low, close, volume

2. **`sp500_index`** - S&P 500 index metrics
   - Foreign Key: date → date_dimension
   - Columns: close, open, high, low, volume, return, volatility_7d

3. **`depression_news`** - Daily depression mentions in news
   - Foreign Key: date → date_dimension
   - Columns: depression_word_count, total_articles, avg_depression_per_article

4. **`depression_index`** - Google Trends depression index
   - Foreign Key: date → date_dimension
   - Columns: depression_index

5. **`rainfall`** - Daily average rainfall
   - Foreign Key: date → date_dimension
   - Columns: avg_rainfall

#### Views:
1. **`daily_analysis_view`** - All metrics joined by date
2. **`stock_with_company_view`** - Stock prices with company info

### Usage:
```bash
psql -U postgres -d tutorial_db -f create_normalized_schema.sql
```

---

## 3. ERD Diagram Generation

To generate an ERD diagram from your PostgreSQL schema, use one of these tools:

### Option A: pgAdmin (GUI)
1. Open pgAdmin
2. Connect to `tutorial_db`
3. Right-click on database → "Generate ERD"

### Option B: SchemaSpy (Command Line)
```bash
java -jar schemaspy.jar -t pgsql -db tutorial_db -u postgres -p 123 \
     -host localhost -o ./erd_output
```

### Option C: DBeaver (Free GUI Tool)
1. Install DBeaver
2. Connect to PostgreSQL
3. Right-click database → "View Diagram"

---

## 4. When to Use Each Approach

### Use Merged CSV (`merged_analysis_data.csv`) when:
- ✓ Doing time-series analysis
- ✓ Building ML models
- ✓ Creating visualizations
- ✓ Exploratory data analysis
- ✓ Quick pandas operations

### Use SQL Database when:
- ✓ Need to query specific tickers/companies
- ✓ Want to see data relationships (ERD)
- ✓ Need to update individual tables
- ✓ Maintain data integrity
- ✓ Complex joins and aggregations
- ✓ Multi-user access

---

## 5. Next Steps

### For Analysis:
1. Load `merged_analysis_data.csv` into pandas/R
2. Handle missing values (imputation or filtering to 2017-2018)
3. Perform correlation analysis
4. Build time-series models

### For Database:
1. Run `create_normalized_schema.sql` to create tables
2. Load CSV data into respective tables using Python scripts
3. Generate ERD diagram
4. Create additional views/queries as needed

---

## Files Created:
- ✓ `merge_all_datasets.py` - Python merge script
- ✓ `create_normalized_schema.sql` - SQL schema definition
- ✓ `csv_exports/merged_analysis_data.csv` - Merged data output
- ✓ `DATA_INTEGRATION_SUMMARY.md` - This documentation

---

**Status:** Both approaches ready to use! 🎉
