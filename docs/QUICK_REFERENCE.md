# Time Series Analysis - Quick Reference Guide

## 📊 Generated Files Overview

### Main Analysis Script
- **`time_series_depression_stock_analysis.py`** - Complete analysis pipeline (800+ lines)

### Visualizations (PNG files)
1. **`EXECUTIVE_SUMMARY_DASHBOARD.png`** - Comprehensive one-page overview 🎯
2. **`analysis_1_stock_vs_depression.png`** - Stock metrics scatter plots
3. **`analysis_1_time_series.png`** - Time series overlays
4. **`analysis_2_sp500_depression.png`** - S&P 500 relationships
5. **`analysis_3_rainfall_impact.png`** - Environmental factor analysis
6. **`analysis_4_industry_patterns.png`** - Industry performance overview
7. **`analysis_4_industry_detail.png`** - Key industry deep dive

### Data & Reports
- **`merged_time_series_data.csv`** - Complete merged dataset (551 records)
- **`time_series_analysis_report.txt`** - Text summary of findings
- **`TIME_SERIES_ANALYSIS_README.md`** - Detailed documentation

---

## 🔍 Quick Findings Summary

### Question 1: Do stock metrics change with depression indicators?
**Answer**: YES, but weakly
- Price range (volatility) shows strongest correlation: **0.274**
- Stock prices have weak positive correlation: **~0.11**
- Volume shows minimal relationship with depression

### Question 2: How does S&P 500 relate to depression?
**Answer**: Volatility matters more than price
- S&P 500 volatility ↔ Depression index: **0.267** ✓
- S&P 500 close ↔ Depression index: **0.144**
- Higher depression → Higher market volatility

### Question 3: Does rainfall affect these relationships?
**Answer**: NO, negligible impact
- All rainfall correlations < **|0.04|**
- Rainfall is not a meaningful factor in this analysis

### Question 4: Industry-specific patterns?
**Answer**: YES, significant variation

**Winners** (positive returns):
- Heavy Electrical Equipment: +14.13%
- Personal Care Products: +1.81%
- Electric Utilities: +0.70%

**Most affected by depression** (negative correlation):
- Copper mining: -0.142
- Ground Transportation: -0.102
- Homebuilding: -0.078

**Counter-intuitive gainers during high depression**:
- Leisure Products: +0.085
- Publishing: +0.080
- Restaurants: +0.073

---

## 📈 Key Metrics at a Glance

| Metric | Value |
|--------|-------|
| Analysis Period | 2017-01-01 to 2018-07-05 |
| Trading Days | 551 |
| Companies Tracked | Multiple across all sectors |
| Industries Analyzed | 100+ unique industries |
| Depression Index Range | 51 - 90 |
| Average S&P 500 Return | ~0.05% daily |

---

## 🚀 How to Use These Files

### For Presentations
→ Use **`EXECUTIVE_SUMMARY_DASHBOARD.png`**

### For Detailed Analysis
1. Start with **`TIME_SERIES_ANALYSIS_README.md`**
2. Review **`time_series_analysis_report.txt`**
3. Examine individual analysis PNG files

### For Further Research
- Use **`merged_time_series_data.csv`** for custom analysis
- Modify **`time_series_depression_stock_analysis.py`** for new questions

---

## 💡 Main Takeaways

1. **Volatility is the Story**: Market uncertainty (not price level) correlates with depression
2. **Industry Matters**: Different sectors respond very differently to sentiment
3. **Weak Overall Effect**: Depression indicators don't strongly predict stock returns
4. **Economic Context**: 2017-2018 was a growth period; results may differ in recessions

---

## 🔧 Running the Analysis

```bash
# One-time setup
pip install pandas numpy matplotlib seaborn scipy

# Run complete analysis (generates all files)
python time_series_depression_stock_analysis.py

# Generate executive summary only
python create_executive_summary.py
```

---

## 📚 Data Sources Used

✓ `stock_data_wiki_clean.csv` - 274K stock records with industry info
✓ `SP500_data.csv` - S&P 500 index daily data
✓ `Depression_index_data.csv` - Weekly Google Trends depression index
✓ `ccnews_depression_daily_count_final.csv` - Daily news depression mentions
✓ `Rainfall_data.csv` - Daily rainfall by state

---

Generated: November 22, 2025
Project: Fundamentals of Data Engineering
