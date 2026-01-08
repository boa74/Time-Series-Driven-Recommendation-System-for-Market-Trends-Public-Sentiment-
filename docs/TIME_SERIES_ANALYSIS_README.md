# Time Series Analysis: Depression Data and Stock Market Performance

## Overview
This comprehensive analysis examines the relationships between depression indicators (word count and index), stock market performance, rainfall data, and industry-specific patterns from 2017-2018.

## Analysis Components

### 1. Stock Market Metrics vs Depression Indicators
**Objective**: Analyze how stock market metrics (open, high, low, close, volume) relate to depression index and word counts.

**Key Findings**:
- **Price Range shows strongest correlation** (0.2745) with depression index, suggesting higher market volatility during periods of elevated depression sentiment
- Stock prices show weak positive correlation with depression index (~0.11)
- Volume has minimal correlation with depression word count (0.0191)

**Visualizations**:
- `analysis_1_stock_vs_depression.png`: Scatter plots showing relationships between individual metrics
- `analysis_1_time_series.png`: Time series overlay of stock prices, volume, and depression indicators

### 2. S&P 500 Performance vs Depression Indicators
**Objective**: Examine relationship between S&P 500 index performance and depression metrics.

**Key Findings**:
- S&P 500 close price has **weak positive correlation** (0.1436) with depression index
- **Market volatility (7-day) shows stronger correlation** (0.2668) with depression index
- Daily returns show slight negative correlation (-0.0721) with depression word count
- Stock performance by depression level:
  - **Low depression**: -0.09% average return, 0.44% volatility
  - **Medium depression**: +0.10% average return, 0.43% volatility
  - **High depression**: +0.08% average return, 0.65% volatility
  - **Very high depression**: +0.04% average return, 0.40% volatility

**Visualizations**:
- `analysis_2_sp500_depression.png`: Multiple views of S&P 500 vs depression metrics

### 3. Rainfall Impact Analysis
**Objective**: Determine if rainfall affects stock prices and depression indicators.

**Key Findings**:
- **Minimal correlation** between rainfall and all variables (all < |0.04|)
- Rainfall vs close price: -0.0289
- Rainfall vs depression index: 0.0054
- **Conclusion**: Rainfall does not appear to be a significant factor in either stock performance or depression indicators

**Visualizations**:
- `analysis_3_rainfall_impact.png`: Rainfall relationships with stocks and depression

### 4. Industry-Specific Analysis
**Objective**: Identify industry patterns and relationships with depression indicators using stock_data_wiki.

**Top Performing Industries** (by average price change):
1. **Heavy Electrical Equipment**: +14.13% (but only 1 stock tracked)
2. **Specialized Consumer Services**: +4.13%
3. **Personal Care Products**: +1.81%
4. **Human Resource & Employment Services**: +1.77%
5. **Industrial Machinery & Components**: +0.89%

**Industries Most Negatively Impacted by Depression** (negative correlation):
- Copper: -0.142
- Cargo Ground Transportation: -0.102
- Other Specialized REITs: -0.101
- Gold: -0.087
- Homebuilding: -0.078

**Industries Positively Correlated with Depression** (counter-intuitive):
- Leisure Products: +0.085
- Publishing: +0.080
- Restaurants: +0.073
- Reinsurance: +0.068
- Regional Banks: +0.042

**Sector Performance**:
- Most sectors showed minimal average price changes
- **Financials** and **Information Technology** sectors had varied performance across industries
- **Utilities** sector showed consistent positive returns (~0.70% for Electric Utilities)

**Visualizations**:
- `analysis_4_industry_patterns.png`: Overall industry performance and correlations
- `analysis_4_industry_detail.png`: Detailed time series for key industries (Semiconductors, Pharmaceuticals, Regional Banks, Oil & Gas)

## Data Sources

### Input Datasets
1. **stock_data_wiki_clean.csv**: Daily stock data with company, sector, and industry information
   - 274,398 records covering 2017-2018
   - Fields: date, ticker, open, high, low, close, volume, company_name, sector, industry

2. **Depression_index_data.csv**: Weekly depression index from Google Trends
   - 332 records
   - Scale: 0-100

3. **CCnews_Depression_data.csv**: News articles mentioning depression
   - 98,135 articles analyzed
   - Daily aggregation in `ccnews_depression_daily_count_final.csv`

4. **SP500_data.csv**: S&P 500 index data
   - 4,019 daily records
   - Includes returns and 7-day volatility

5. **Rainfall_data.csv**: Daily rainfall by US state
   - 4,019 records
   - Aggregated to national average

### Output Files
1. **merged_time_series_data.csv**: Complete merged dataset with all variables
2. **time_series_analysis_report.txt**: Text summary of key findings
3. **6 PNG visualization files**: Detailed charts and graphs

## Key Insights

### Overall Conclusions

1. **Weak but Consistent Relationships**: Depression indicators show weak positive correlations with stock prices and stronger correlations with market volatility

2. **Volatility as Key Indicator**: Market volatility (price range, 7-day volatility) shows stronger relationships with depression than absolute price levels

3. **Industry Variability**: Different industries respond differently to depression sentiment:
   - **Defensive sectors** (utilities, consumer staples) show resilience
   - **Cyclical sectors** (copper, transportation, homebuilding) show negative correlation
   - **Counter-cyclical patterns** in some consumer-facing industries (restaurants, publishing)

4. **Environmental Factors**: Rainfall shows negligible impact on both stock performance and depression indicators

5. **Data Period**: Analysis covers 2017-2018, a period of:
   - General economic growth
   - Low unemployment
   - Relatively stable markets
   - Results may differ in recession or crisis periods

## Methodology

### Data Processing
- **Time Alignment**: Weekly depression index forward-filled to match daily stock data
- **Aggregation**: Individual stock data aggregated to daily market averages
- **Missing Data**: Depression word counts filled with 0 for days without articles
- **Derived Metrics**: Price range, percentage changes, category classifications

### Statistical Analysis
- Pearson correlation coefficients
- Time series overlays
- Categorical aggregations
- Industry-level grouping and analysis

### Visualization Strategy
- Dual-axis time series for comparing different scales
- Scatter plots with trend lines for relationships
- Bar charts for categorical comparisons
- Industry-specific detailed analysis

## Usage

### Running the Analysis

```bash
# Install required packages
pip install -r requirements_analysis.txt

# Run the complete analysis
python time_series_depression_stock_analysis.py
```

### Script Structure

```python
class DepressionStockAnalyzer:
    - load_data()                           # Load all datasets
    - prepare_merged_dataset()              # Merge and align data
    - analyze_stock_depression_relationship() # Analysis 1
    - analyze_sp500_depression()            # Analysis 2
    - analyze_rainfall_impact()             # Analysis 3
    - analyze_industry_patterns()           # Analysis 4
    - generate_summary_report()             # Create summary
    - run_complete_analysis()               # Execute all
```

## Limitations and Considerations

1. **Correlation vs Causation**: Correlations identified do not imply causation
2. **Time Period**: Limited to 2017-2018; results may not generalize to other periods
3. **Depression Metrics**: Based on Google Trends and news mentions, not clinical data
4. **Geographic Scope**: US-focused analysis
5. **Sample Size**: Only 551 complete daily records with all variables
6. **Economic Context**: Conducted during economic expansion, not recession

## Future Research Directions

1. **Extended Time Period**: Include 2020-2023 to capture pandemic effects
2. **Geographic Analysis**: State-level or regional patterns
3. **Sentiment Analysis**: More sophisticated NLP on news content
4. **Lagged Effects**: Examine time-delayed relationships
5. **Macro Variables**: Include unemployment, GDP, consumer confidence
6. **Company-Level**: Individual stock analysis rather than aggregates

## Technical Requirements

- Python 3.8+
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- scipy >= 1.10.0

## Author
Data Engineering Analysis Project
November 2025

## License
For educational and research purposes
