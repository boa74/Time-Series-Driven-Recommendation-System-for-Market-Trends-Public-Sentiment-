# 📊 Time Series Analysis: Depression & Stock Market
## Complete Analysis Package - November 2025

---

## 🎯 Start Here

### New to this analysis? 
→ **Open `EXECUTIVE_SUMMARY_DASHBOARD.png`** for visual overview
→ **Read `QUICK_REFERENCE.md`** for key findings

### Want detailed insights?
→ **Read `TIME_SERIES_ANALYSIS_README.md`** for comprehensive documentation

---

## 📁 File Directory

### 🖼️ Visualizations (7 files)
```
EXECUTIVE_SUMMARY_DASHBOARD.png  ⭐ START HERE - One-page overview
analysis_1_stock_vs_depression.png    Stock metrics vs depression
analysis_1_time_series.png            Time series overlays  
analysis_2_sp500_depression.png       S&P 500 analysis
analysis_3_rainfall_impact.png        Environmental factors
analysis_4_industry_patterns.png      Industry performance
analysis_4_industry_detail.png        Key industries deep dive
```

### 📄 Documentation (3 files)
```
QUICK_REFERENCE.md               ⭐ Quick findings & how-to
TIME_SERIES_ANALYSIS_README.md      Complete documentation
time_series_analysis_report.txt     Text summary of results
```

### 💾 Data & Code (4 files)
```
merged_time_series_data.csv              Final merged dataset (551 records)
time_series_depression_stock_analysis.py Main analysis script (800+ lines)
create_executive_summary.py              Dashboard generator
requirements_analysis.txt                Python dependencies
```

---

## 🔑 Key Questions Answered

| Question | Answer | Details |
|----------|--------|---------|
| **1. Do stock prices (open/high/low/close/volume) change with depression?** | **Yes, but weakly** | Price range correlation: 0.27; prices ~0.11 |
| **2. How does S&P 500 relate to depression index/words?** | **Volatility correlates, not price** | Volatility correlation: 0.27; Price: 0.14 |
| **3. Does rainfall affect stocks/depression?** | **No significant impact** | All correlations < 0.04 |
| **4. Industry-specific patterns?** | **Yes, major variation** | Top: +14%, Bottom: -15% correlation |

---

## 📊 Analysis Breakdown

### Analysis 1: Stock Metrics vs Depression
- **Files**: `analysis_1_stock_vs_depression.png`, `analysis_1_time_series.png`
- **Finding**: Price volatility (range) shows strongest correlation (0.274)
- **Insight**: Market uncertainty ↑ when depression sentiment ↑

### Analysis 2: S&P 500 vs Depression  
- **File**: `analysis_2_sp500_depression.png`
- **Finding**: 7-day volatility correlates 0.267 with depression index
- **Insight**: Depression affects market stability more than price level

### Analysis 3: Rainfall Impact
- **File**: `analysis_3_rainfall_impact.png`
- **Finding**: Negligible correlations (all < |0.04|)
- **Insight**: Weather doesn't meaningfully impact stocks or depression in this data

### Analysis 4: Industry Patterns
- **Files**: `analysis_4_industry_patterns.png`, `analysis_4_industry_detail.png`
- **Finding**: Industries show -15% to +14% correlation range
- **Key Winners**: Heavy Electrical Equipment (+14%), Personal Care (+1.8%)
- **Most Impacted**: Copper (-0.14), Transportation (-0.10), Homebuilding (-0.08)

---

## 🚀 Quick Start

### Option 1: View Results Only
```bash
# Open the executive dashboard image
open EXECUTIVE_SUMMARY_DASHBOARD.png

# Read the quick reference
open QUICK_REFERENCE.md
```

### Option 2: Re-run Analysis
```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scipy

# Run complete analysis (takes ~30 seconds)
python time_series_depression_stock_analysis.py
```

### Option 3: Explore Data
```python
import pandas as pd

# Load merged dataset
df = pd.read_csv('merged_time_series_data.csv')
df['date'] = pd.to_datetime(df['date'])

# Available columns:
# date, open, high, low, close, volume, num_stocks,
# Close_^GSPC, Return, Volatility_7, 
# depression_word_count, total_articles,
# depression_index, avg_national_rainfall,
# price_range, price_change_pct, depression_index_category
```

---

## 📈 Data Coverage

- **Time Period**: January 1, 2017 → July 5, 2018
- **Trading Days**: 551 complete records
- **Stock Records**: 274,398 individual stock-day observations
- **Industries**: 100+ unique industries analyzed
- **Companies**: Multiple companies across all major sectors
- **Depression Index**: Weekly Google Trends data (51-90 range)
- **News Articles**: 98,135 articles mentioning depression

---

## 🎓 Research Context

### Economic Period
2017-2018 was characterized by:
- ✓ Economic expansion
- ✓ Low unemployment  
- ✓ Rising stock markets
- ✓ Pre-pandemic, pre-2020 volatility

### Implications
Results reflect **growth period dynamics**. Relationships may differ during:
- Economic recessions
- Market crashes  
- Crisis periods (COVID-19, 2008, etc.)

---

## 💡 Key Insights

### 1️⃣ Volatility > Price Level
Market uncertainty correlates with depression more than absolute prices

### 2️⃣ Industry Specificity
Different sectors respond differently:
- **Cyclical** (transport, homebuilding): Negative correlation
- **Defensive** (utilities, consumer staples): Resilient
- **Counter-cyclical** (restaurants, publishing): Positive correlation

### 3️⃣ Weak Predictive Power
Depression indicators don't strongly predict returns
- Useful for understanding context
- Not reliable for trading signals

### 4️⃣ Environmental Factors
Rainfall shows no significant relationship

---

## 🔬 Methodology

### Statistical Approaches
- Pearson correlation analysis
- Time series alignment and merging
- Categorical aggregation by industry/sector
- Volatility calculations (7-day rolling)

### Data Processing
- Weekly → Daily interpolation for depression index
- Multi-source data merging (5 datasets)
- Missing value handling (forward fill, zeros)
- Derived metrics (price range, % change, categories)

---

## 🛠️ Technical Stack

**Languages**: Python 3.8+

**Libraries**:
- pandas (data manipulation)
- numpy (numerical operations)
- matplotlib (visualization)
- seaborn (statistical graphics)
- scipy (statistical functions)

**Input Formats**: CSV
**Output Formats**: PNG (visualizations), CSV (merged data), TXT/MD (reports)

---

## 📚 Further Reading

### Related Topics
- Behavioral finance and market sentiment
- Google Trends as economic indicator
- Industry sector rotation strategies
- Market volatility forecasting

### Potential Extensions
- Include 2020-2024 data (pandemic effects)
- Add more sentiment sources (Twitter, Reddit)
- Incorporate macroeconomic variables
- Machine learning prediction models
- Geographic/regional analysis

---

## ⚠️ Important Notes

### Limitations
1. **Correlation ≠ Causation**: Relationships shown don't imply cause
2. **Limited Time Period**: 18 months of data
3. **Economic Context**: Growth period only
4. **Depression Metrics**: Based on searches/news, not clinical data

### Recommendations
- Use for exploratory analysis
- Combine with other indicators for decisions
- Consider economic context when interpreting
- Validate findings with additional time periods

---

## 📞 File Navigation Map

```
START
  ↓
EXECUTIVE_SUMMARY_DASHBOARD.png (Visual Overview)
  ↓
QUICK_REFERENCE.md (Quick Findings)
  ↓
TIME_SERIES_ANALYSIS_README.md (Full Documentation)
  ↓
Individual Analysis PNGs (Detailed Charts)
  ↓
merged_time_series_data.csv (Raw Data)
```

---

## ✅ Deliverables Checklist

✓ Main analysis script
✓ 7 visualization files  
✓ Merged dataset (CSV)
✓ Text summary report
✓ Comprehensive README
✓ Quick reference guide
✓ This index file
✓ Requirements file

**Total: 13 files delivered**

---

## 🎯 Analysis Objectives: COMPLETED

✅ **Objective 1**: Stock metrics (OHLCV) vs depression → **Analyzed**
✅ **Objective 2**: S&P 500 vs depression indicators → **Analyzed**  
✅ **Objective 3**: Rainfall impact assessment → **Analyzed**
✅ **Objective 4**: Industry-specific patterns → **Analyzed**

---

**Analysis completed**: November 22, 2025
**Project**: Fundamentals of Data Engineering
**Dataset period**: 2017-2018

*For questions or additional analysis, modify the Python scripts and re-run.*
