# 📊 Statistical Significance Analysis Results
## Explaining Why Some Correlations Matter More Than Others

---

## 🎯 Quick Answer

**Only 3 correlations are statistically robust and trustworthy:**

| Rank | Relationship | r | p-value | R² | Score |
|------|-------------|---|---------|-----|-------|
| 1 | **Price Range (Volatility) ↔ Depression Index** | 0.274 | <0.001 | 7.5% | ⭐⭐⭐⭐⭐ 10/10 |
| 2 | **S&P 500 Volatility ↔ Depression Index** | 0.267 | <0.001 | 7.1% | ⭐⭐⭐⭐⭐ 10/10 |
| 3 | **S&P 500 Close ↔ Depression Index** | 0.144 | <0.001 | 2.1% | ⭐⭐⭐⭐⭐ 10/10 |

These survived all statistical tests including:
- ✅ P-value < 0.001 (highly significant)
- ✅ Bonferroni correction for 17 multiple comparisons
- ✅ Large sample size (N = 551)
- ✅ Confidence intervals don't include zero

---

## 📁 Generated Files

### 📊 Visualizations (4 files)
1. **`STATISTICAL_SIGNIFICANCE_ANALYSIS.png`** ⭐ Main dashboard with 7 plots
2. **`CORRELATION_IMPORTANCE_RANKING.png`** - Ranked by importance score
3. **`CORRELATION_COMPARISON_TABLE.png`** - Side-by-side comparison table
4. *(Previous)* `EXECUTIVE_SUMMARY_DASHBOARD.png` - Original analysis overview

### 📄 Documentation (3 files)
1. **`WHY_CORRELATIONS_MATTER.md`** ⭐ START HERE - Clear explanation
2. **`STATISTICAL_INTERPRETATION_GUIDE.txt`** - Detailed technical guide
3. **`STATISTICAL_ANALYSIS_README.md`** - This file

### 💾 Data (1 file)
1. **`correlation_statistics_full.csv`** - All 17 correlations with full statistics

### 🔧 Code (1 file)
1. **`statistical_significance_analysis.py`** - Analysis script

---

## 🔬 The 5 Statistical Tests Explained

### 1. P-Value (Statistical Significance)
**Question:** Is this correlation real or just random chance?

| P-Value | Symbol | Interpretation | Our Results |
|---------|--------|----------------|-------------|
| < 0.001 | *** | Highly significant - trust it | 3 correlations ✓ |
| 0.001-0.01 | ** | Very significant - probably real | 4 correlations |
| 0.01-0.05 | * | Significant - possibly real | 1 correlation |
| ≥ 0.05 | ns | Not significant - don't trust | 9 correlations ❌ |

**Winner:** Price Range ↔ Depression (p = 5.5×10⁻¹¹)
- Probability it's random: 0.000000005%
- Virtually impossible this is chance

### 2. Effect Size (Correlation Strength)
**Question:** Even if real, is it strong enough to matter?

| |r| Range | Label | Our Results |
|-----------|-------|-------------|
| ≥ 0.50 | Large | 0 correlations |
| 0.30-0.49 | Medium | 0 correlations |
| 0.10-0.29 | Small | 8 correlations ✓ |
| < 0.10 | Negligible | 9 correlations |

**Best:** Price Range ↔ Depression (r = 0.274)
- Small but meaningful
- Explains 7.5% of variance
- Other 92.5% from other factors

**Reality Check:** Even strongest correlations explain < 8% of variance
- Depression is ONE of MANY factors
- Economics, news, policy explain most variance

### 3. Confidence Interval (Precision)
**Question:** How precisely do we know the correlation?

**Example - Precise:**
- Price Range ↔ Depression: 95% CI [0.196, 0.350]
- Narrow range, doesn't include zero
- True correlation almost certainly positive

**Example - Imprecise:**
- Rainfall ↔ S&P Return: 95% CI [-0.078, 0.111]
- Wide range, includes zero
- Could be positive, negative, or none at all

### 4. Sample Size (Reliability)
**Our Data:** N = 551 observations

✅ **Excellent sample size!**
- Rule of thumb: Need N > 30 for reliable results
- We have 551 → Very reliable
- More data = More confident in findings

### 5. Bonferroni Correction (Multiple Comparisons)
**The Problem:**
- We tested 17 correlations
- With α = 0.05, expect ~1 false positive by chance
- Can't trust everything showing p < 0.05

**The Solution:**
- Bonferroni adjusted α = 0.05/17 = 0.00294
- Only accept p < 0.00294 as truly significant
- Protects against false discoveries

**Results:**
- Started with 8 "significant" correlations (p < 0.05)
- Only 3 survive Bonferroni correction
- These 3 are truly robust

---

## 📈 Results Summary

### ✅ Trust Completely (Score 10/10)
**3 correlations passed ALL tests:**

#### 1. Price Range (Volatility) ↔ Depression Index
- **r = 0.2745, p < 0.001, R² = 7.5%**
- ✓ Highly significant (p = 5.5×10⁻¹¹)
- ✓ Survives Bonferroni correction
- ✓ Precise CI [0.196, 0.350]
- **Meaning:** Market volatility ↑ when depression sentiment ↑

#### 2. S&P 500 Volatility ↔ Depression Index
- **r = 0.2668, p < 0.001, R² = 7.1%**
- ✓ Highly significant (p = 2.0×10⁻¹⁰)
- ✓ Survives Bonferroni correction
- ✓ Precise CI [0.188, 0.343]
- **Meaning:** Market-level volatility linked to depression

#### 3. S&P 500 Close ↔ Depression Index
- **r = 0.1436, p < 0.001, R² = 2.1%**
- ✓ Highly significant (p = 7.3×10⁻⁴)
- ✓ Survives Bonferroni correction
- ✓ CI [0.061, 0.224]
- **Meaning:** Weak positive relationship (markets rose during study)

### ⚠️ Use With Caution (Score 3-7/10)
**5 correlations significant but don't survive Bonferroni:**

- Stock High/Open/Close/Low ↔ Depression Index (all r ~ 0.11, p < 0.01)
- Trading Volume ↔ Depression Index (r = 0.110, p = 0.010)

**Why caution?**
- Statistically significant individually
- Fail strict multiple comparison test
- Could be spurious from testing many variables
- Small effect sizes (explain only 1-1.5% variance)

### ❌ Don't Trust (Score 0-2/10)
**9 correlations not statistically significant:**

**Depression Word Count:** All p > 0.05
- vs Price Range: p = 0.071
- vs S&P Return: p = 0.091
- vs Stock Close: p = 0.542
- vs Volume: p = 0.655

**Rainfall:** All p > 0.05
- vs Stock Close: p = 0.498
- vs Depression Index: p = 0.894
- vs S&P Return: p = 0.695

**S&P 500 Returns:**
- vs Depression Index: p = 0.621 (no relationship!)
- vs Depression Word Count: p = 0.091

---

## 💡 Key Insights

### What This Tells Us:

1. **Volatility is the Real Story**
   - Market uncertainty correlates with depression
   - Price levels show weak/unreliable relationships
   - Investors should watch volatility, not prices

2. **Depression Index > Word Counts**
   - Google Trends index (weekly) is reliable
   - Daily word counts are too noisy
   - Use index for serious analysis

3. **Rainfall Doesn't Matter**
   - No relationship with stocks or depression
   - All rainfall correlations p > 0.05
   - Can safely ignore weather factors

4. **Small Effect Sizes**
   - Even best correlations explain < 8% variance
   - Depression is one of MANY factors
   - Don't expect strong predictive power

### What We Can Confidently Say:

✅ **TRUE:** Market volatility increases with depression sentiment (strong evidence)
✅ **TRUE:** This relationship is real, not random (p < 0.001)
✅ **TRUE:** Effect is small but meaningful (explains ~7% variance)

### What We CANNOT Say:

❌ **FALSE:** Depression strongly predicts stock returns (r = -0.02, p = 0.62)
❌ **FALSE:** News mentions of depression correlate with markets (all p > 0.05)
❌ **FALSE:** Rainfall affects stocks or depression (all p > 0.05)
❌ **FALSE:** Individual stock prices reliably track depression (fail Bonferroni)

---

## 🎓 Interpretation Guidelines

### For Academic/Research Use:

**Report these 3 findings:**
1. Volatility-depression correlation (r = 0.27, p < 0.001)
2. S&P 500 volatility-depression link (r = 0.27, p < 0.001)
3. Weak S&P close-depression relationship (r = 0.14, p < 0.001)

**Include these caveats:**
- Small effect sizes (< 8% variance)
- Limited time period (2017-2018, growth era)
- Correlation ≠ causation
- Need replication in other periods

### For Investment/Trading:

**Can use:**
- Volatility as depression sentiment indicator
- Monitor VIX-like metrics alongside depression trends

**Cannot use:**
- Depression for price prediction (too weak)
- News word counts (not significant)
- Individual stock predictions (unreliable)

**Recommendation:**
- Depression is context/background information
- Not a trading signal
- Combine with other economic indicators

### For General Understanding:

**Simple takeaway:**
> When depression sentiment is high, markets tend to be more volatile (choppy, uncertain) but not necessarily going down. The relationship is real but explains only ~7% of market movements. Most price action comes from economic data, earnings, policy, etc.

---

## 📊 Visual Guide

### Start with these files:

1. **First-time viewer?**
   - Open: `CORRELATION_IMPORTANCE_RANKING.png`
   - Shows: Color-coded importance scores + decision tree

2. **Want detailed statistics?**
   - Open: `STATISTICAL_SIGNIFICANCE_ANALYSIS.png`
   - Shows: 7-plot dashboard with all tests

3. **Need comparison table?**
   - Open: `CORRELATION_COMPARISON_TABLE.png`
   - Shows: Side-by-side comparison of all 17 correlations

4. **Want to read explanation?**
   - Read: `WHY_CORRELATIONS_MATTER.md`
   - Format: Markdown with clear sections

5. **Need full technical details?**
   - Read: `STATISTICAL_INTERPRETATION_GUIDE.txt`
   - Format: Text with examples

---

## 🔧 Technical Details

### Statistical Methods:
- **Correlation:** Pearson's r (parametric)
- **Significance:** Two-tailed t-test
- **Confidence Intervals:** Fisher's z-transformation
- **Multiple Comparisons:** Bonferroni correction
- **Effect Size:** Cohen's guidelines

### Sample Characteristics:
- **N:** 551 complete observations
- **Time Period:** 2017-01-01 to 2018-07-05 (18 months)
- **Variables Tested:** 17 correlation pairs
- **Economic Context:** Growth period, not recession

### Significance Levels:
- **Standard α:** 0.05
- **Bonferroni-adjusted α:** 0.00294 (0.05/17)
- **Used symbols:**
  - `***` = p < 0.001
  - `**` = p < 0.01
  - `*` = p < 0.05
  - `ns` = p ≥ 0.05 (not significant)

---

## 📚 Files Overview

```
Statistical Analysis Package
├── Visualizations/
│   ├── CORRELATION_IMPORTANCE_RANKING.png      ⭐ Best overview
│   ├── STATISTICAL_SIGNIFICANCE_ANALYSIS.png   📊 7-plot dashboard
│   └── CORRELATION_COMPARISON_TABLE.png        📋 Comparison table
├── Documentation/
│   ├── WHY_CORRELATIONS_MATTER.md              📖 Main explanation
│   ├── STATISTICAL_INTERPRETATION_GUIDE.txt    📝 Technical guide
│   └── STATISTICAL_ANALYSIS_README.md          📄 This file
├── Data/
│   └── correlation_statistics_full.csv          💾 All results
└── Code/
    └── statistical_significance_analysis.py     💻 Analysis script
```

---

## 🚀 How to Use

### Option 1: Quick Understanding
```
1. Open: CORRELATION_IMPORTANCE_RANKING.png
2. Read: Top 3 green bars (score 10/10)
3. Note: These are the only reliable findings
```

### Option 2: Deep Dive
```
1. Read: WHY_CORRELATIONS_MATTER.md (10 min)
2. View: STATISTICAL_SIGNIFICANCE_ANALYSIS.png
3. Review: correlation_statistics_full.csv
4. Read: STATISTICAL_INTERPRETATION_GUIDE.txt
```

### Option 3: For Your Research
```python
import pandas as pd

# Load results
df = pd.read_csv('correlation_statistics_full.csv')

# Get only robust findings (Bonferroni survivors)
robust = df[df['Bonferroni_Significant'] == True]
print(robust[['Variable_X', 'Variable_Y', 'Correlation', 'P_Value']])

# Output:
# Variable_X                   Variable_Y          Correlation  P_Value
# Price Range (Volatility)     Depression Index    0.2745      5.54e-11
# S&P 500 Volatility          Depression Index    0.2668      1.96e-10
# S&P 500 Close               Depression Index    0.1436      7.25e-04
```

---

## ⚠️ Important Disclaimers

### Limitations:
1. **Time Period:** Only 18 months (2017-2018)
2. **Economic Context:** Growth period, not tested in recession
3. **Causation:** Correlations don't prove cause and effect
4. **Effect Size:** Small effects (< 8% variance explained)
5. **Generalization:** Results may not apply to other time periods

### Recommendations:
- ✓ Use for understanding context
- ✓ Combine with other indicators
- ✓ Validate in multiple time periods
- ✗ Don't use alone for trading decisions
- ✗ Don't assume causation
- ✗ Don't extrapolate beyond growth periods

---

## 📞 Quick Reference

**Bottom Line:**
> Only market VOLATILITY reliably correlates with depression (r ≈ 0.27, p < 0.001). Price levels, returns, word counts, and rainfall show no reliable relationships.

**Best Evidence:**
- Price Range ↔ Depression: r = 0.27 ✓✓✓
- S&P Volatility ↔ Depression: r = 0.27 ✓✓✓
- S&P Close ↔ Depression: r = 0.14 ✓✓✓

**No Evidence:**
- Returns ↔ Depression: r = -0.02, p = 0.62 ❌
- Word Count correlations: All p > 0.05 ❌
- Rainfall correlations: All p > 0.05 ❌

---

**Analysis Date:** November 22, 2025  
**Project:** Fundamentals of Data Engineering  
**Method:** Comprehensive statistical significance testing with multiple comparison correction

For questions or additional analysis, see the Python scripts in this directory.
