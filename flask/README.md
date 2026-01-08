# Depression Index & Stock Market Analysis Dashboard

🎓 **Columbia University Style** - Professional Flask Web Application

## 📊 Overview

This Flask application provides comprehensive visualization and analysis of the relationship between depression index (derived from news articles) and stock market performance from 2017-2024.

## 🚀 Quick Start

### Start the Server
```bash
cd flask
./start_server.sh
```

### Access the Application
Open your browser and navigate to: **http://127.0.0.1:18502**

## 📁 Project Structure

```
flask/
├── app.py                      # Main Flask application
├── start_server.sh            # Server startup script
├── templates/                 # HTML templates
│   ├── base.html             # Base template with Columbia colors
│   ├── index.html            # Homepage with today's indicators
│   ├── executive_summary.html # Complete analysis dashboard
│   ├── timeseries_analysis.html
│   ├── industry_impact.html
│   └── lag_volatility_analysis.html
└── static/
    └── images/               # Analysis visualizations (11 PNG files)
```

## 🎨 Design Features

### Color Scheme (Columbia University)
- **Primary Blue**: #75A8C8, #B9D9EB
- **Dark Blue**: #1E3A5F
- **Accent**: #E8F4F8
- Modern gradient backgrounds
- Smooth hover effects and animations

### Typography
- **Font**: Inter (Google Fonts)
- Clean, professional appearance
- Excellent readability

## 📱 Pages & Features

### 1. **Homepage** (`/`)
- **Today's Indicators** (3 cards):
  - 🌧️ Rainfall (with progress bar)
  - 🌡️ Temperature (with progress bar)
  - 😊 Depression Index (Plotly circular gauge)
- **Stock Volatility Charts** (2 interactive charts):
  - S&P 500 7-day rolling volatility
  - Stock portfolio volatility
- **Analysis Overview Cards**: Total observations, date range, variables, industries
- **Quick Navigation**: Links to all analysis pages

### 2. **Executive Summary** (`/executive-summary`)
Shows all 11 analysis images from the research:

#### Main Dashboards:
- **EXECUTIVE_SUMMARY_DASHBOARD.png**: Complete overview
- **STATISTICAL_SIGNIFICANCE_ANALYSIS.png**: P-values and significance levels
- **CORRELATION_COMPARISON_TABLE.png**: Side-by-side correlation metrics
- **CORRELATION_IMPORTANCE_RANKING.png**: Ranked importance chart
- **DEPRESSION_INDEX_VS_WORDCOUNT.png**: Index validation

#### Key Research Findings:
1. **Price Range (Volatility)**: r=0.275, p<0.001 ⭐ Highest correlation
2. **S&P 500 Volatility**: r=0.267, p<0.001 ⭐ Strong market correlation
3. **S&P 500 Close**: r=0.144, p<0.001 ⭐ Market level indicator

### 3. **Time Series Analysis** (`/timeseries-analysis`)

#### Static Analysis Images:
- **analysis_1_time_series.png**: Complete time series overview
- **analysis_1_stock_vs_depression.png**: Stock correlation analysis
- **analysis_2_sp500_depression.png**: S&P 500 relationship

#### Interactive Features:
- **Correlation Statistics Table**: Top significant correlations with p-values
- **Summary Cards**: Total correlations, significant count, success rate
- **4 Interactive Charts** (Plotly.js):
  - Depression Index over time
  - S&P 500 Close Price over time
  - Rainfall patterns
  - Stock Close Price trends

### 4. **Industry Impact Analysis** (`/industry-impact`)

#### Static Analysis Images:
- **analysis_4_industry_patterns.png**: Industry-specific patterns
- **analysis_4_industry_detail.png**: Detailed industry breakdown
- **analysis_3_rainfall_impact.png**: Weather correlation analysis

#### Interactive Features:
- **Portfolio Statistics**: 498 stocks, 12 industries, 551 observations
- **Performance by Depression Category**: Bar chart (Low/Medium/High)
- **Scatter Plots** (2 charts):
  - Volatility vs Depression Index
  - Trading Volume vs Depression Index
- **Key Insights**: Industry patterns and correlations

### 5. **Lag Volatility Analysis** (`/lag-volatility-analysis`)

#### Lag Effect Analysis:
- **Lag 1 Day**: Immediate next-day correlation
- **Lag 2 Days**: Two-day delayed impact
- **Lag 3 Days**: Extended correlation pattern

#### Interactive Features:
- **Summary Cards**: Correlation coefficients for each lag
- **Lag Comparison Chart**: Bar chart comparing all 3 lags
- **Dual-Axis Time Series**: Depression Index + Volatility overlay
- **3 Scatter Plots**: One for each lag period
- **Interpretation Guide**: Research implications

## 🔧 API Endpoints

```python
GET /                           # Homepage
GET /executive-summary          # Complete analysis dashboard
GET /timeseries-analysis       # Time series page
GET /industry-impact           # Industry analysis page
GET /lag-volatility-analysis   # Lag analysis page

# Data APIs
GET /api/today-data            # Latest rainfall, temp, depression index
GET /api/volatility-data       # Historical volatility time series
GET /api/correlation-data      # Correlation statistics
GET /api/timeseries-data       # Complete time series dataset
```

## 📊 Data Sources

### Input Files (Parent Directory):
- `merged_time_series_data.csv` (551 rows, 2017-2024)
  - Depression index, S&P 500, stock prices, rainfall, volatility
- `correlation_statistics_full.csv`
  - Complete correlation analysis with p-values

### Analysis Images:
All 11 PNG files from `analysis/` folder are served via Flask static files at `/static/images/`

## 🎯 Key Features

### 1. **Real-Time Data Display**
- Today's metrics fetched from latest CSV row
- Circular gauge for depression index (0-100 scale)
- Color-coded categories: Green (<30), Yellow (30-60), Red (>60)

### 2. **Interactive Visualizations**
- **Plotly.js** for dynamic charts
- Hover effects show detailed values
- Responsive design for all screen sizes
- Columbia blue color scheme throughout

### 3. **Statistical Rigor**
- P-values and confidence intervals displayed
- Bonferroni correction for multiple comparisons
- Effect sizes (Small/Medium/Large) indicated
- R² values showing explained variance

### 4. **Professional Design**
- Bootstrap 5 framework
- Font Awesome 6 icons
- Smooth CSS animations
- Card-based layout
- Breadcrumb navigation

## 📈 Research Insights

### Significant Findings:
1. **Market Volatility Correlation**: Depression index explains 7.54% of price volatility variance
2. **Statistical Significance**: All top correlations pass Bonferroni correction (p<0.05)
3. **Temporal Patterns**: Consistent effects across 7+ years of daily data
4. **Multi-Industry Coverage**: 498 stocks across 12 different sectors

### Practical Applications:
- 📊 Short-term volatility forecasting
- 🎯 Market sentiment analysis
- 📉 Risk management insights
- 🔮 Behavioral finance research

## 🛠️ Technology Stack

- **Backend**: Flask (Python 3.13)
- **Frontend**: Bootstrap 5.1.3, Font Awesome 6.0
- **Visualization**: Plotly.js (interactive), Static PNG (research figures)
- **Data**: Pandas for CSV processing
- **Styling**: Custom CSS with Columbia University colors

## 📝 Development

### Server Management:
```bash
# Start server
./start_server.sh

# Check if running
ps aux | grep "python3 app.py"

# View logs
tail -f flask.log

# Stop server
lsof -ti:18502 | xargs kill -9
```

### Adding New Analysis:
1. Generate PNG in `analysis/` folder
2. Copy to `flask/static/images/`
3. Add `<img>` tag in relevant template
4. Restart server

## 🎓 Academic Context

**Project**: Time Series Analysis of Depression Index and Stock Market Performance  
**Period**: 2017-2024 (551 daily observations)  
**Data Sources**: 
- News articles (depression word count)
- S&P 500 market data
- Stock portfolio (498 stocks)
- National rainfall data

**Key Metrics**:
- 17 analyzed variables
- 12 industry sectors
- 551 time points
- Multiple correlation methods (Pearson)

## 📄 License

Academic research project for educational purposes.

---

**Last Updated**: November 22, 2025  
**Server**: http://127.0.0.1:18502  
**Status**: ✅ Production Ready
