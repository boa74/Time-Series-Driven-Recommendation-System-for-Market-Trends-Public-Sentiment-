# 🚀 Quick Start Guide

## Project Structure
```
Fundamentals-of-Data-Engineering_bk/
├── 📁 src/                    # Source code (organized)
│   ├── etl/                   # ETL pipeline scripts
│   ├── analysis/              # Statistical analysis
│   ├── visualization/         # Charts and reports
│   └── utils/                 # Helper utilities
├── 📁 data/                   # Data files (organized)
│   ├── raw/                   # Original source data
│   ├── processed/             # Cleaned data
│   ├── final/                 # Analysis-ready data
│   └── exports/               # CSV exports
├── 📁 flask/                  # Web dashboard
├── 📁 reports/               # Analysis outputs
├── 📁 docs/                  # Documentation
├── 📁 config/                # Configuration files
└── 📁 archive/               # Old/deprecated files
```

## Quick Commands

### 🔄 Run ETL Pipeline
```bash
cd src/etl
python Extract_MongoDB.py      # Extract data
python clean_raw_data.py       # Clean data
python merge_all_datasets.py   # Integrate data
python export_to_postgres.py   # Load to database
```

### 📊 Run Analysis
```bash
cd src/analysis
python time_series_depression_stock_analysis.py  # Main analysis
python statistical_significance_analysis.py      # Statistical validation
```

### 🌐 Launch Dashboard
```bash
cd flask
./start_server.sh
# Visit: http://127.0.0.1:18502
```

### 📈 Generate Reports
```bash
cd src/visualization
python create_executive_summary.py    # Create dashboard
python create_final_integrated_dataset.py  # Prepare final data
```

## File Locations
- **Main data**: `data/final/merged_time_series_data.csv`
- **Analysis results**: `reports/analysis/`
- **Documentation**: `docs/`
- **Web app**: `flask/`
- **Configuration**: `config/`