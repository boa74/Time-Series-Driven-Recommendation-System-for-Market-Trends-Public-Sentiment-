#!/usr/bin/env python3
"""
Complete Time Series Analysis - Depression & Stock Market
==========================================================
Integrated script that performs all analysis steps:
1. Load raw data from CSV files
2. Merge data by date
3. Perform 4 main analyses
4. Calculate statistical significance
5. Export all results to CSV files
6. Generate visualizations

No external function files needed - everything is self-contained.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for data files and output"""
    
    # Input data files
    STOCK_DATA = 'stock_data_wiki_clean.csv'
    SP500_DATA = 'SP500_data.csv'
    DEPRESSION_INDEX = 'Depression_index_data.csv'
    DEPRESSION_WORDS = 'CCnews_Depression_data.csv'
    RAINFALL_DATA = 'Rainfall_data.csv'
    
    # Output directories
    OUTPUT_DIR = 'analysis_results'
    FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
    DATA_DIR = os.path.join(OUTPUT_DIR, 'data')
    
    # Create directories if they don't exist
    @classmethod
    def setup_directories(cls):
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.FIGURES_DIR, exist_ok=True)
        os.makedirs(cls.DATA_DIR, exist_ok=True)


# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

class DataLoader:
    """Load and prepare all data sources"""
    
    @staticmethod
    def load_stock_data(filepath):
        """Load stock data from Wikipedia"""
        print(f"Loading stock data from {filepath}...")
        df = pd.read_csv(filepath)
        
        # Standardize column names
        column_mapping = {
            'Date': 'date',
            'Ticker': 'ticker',
            'Industry': 'industry',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        
        # Rename columns if they exist
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        df['date'] = pd.to_datetime(df['date'])
        print(f"  ✓ Loaded {len(df):,} stock records")
        print(f"  ✓ Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  ✓ Unique tickers: {df['ticker'].nunique()}")
        print(f"  ✓ Industries: {df['industry'].nunique()}")
        
        return df
    
    @staticmethod
    def load_sp500_data(filepath):
        """Load S&P 500 index data"""
        print(f"\nLoading S&P 500 data from {filepath}...")
        df = pd.read_csv(filepath)
        
        # Standardize column names
        column_mapping = {
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate returns and volatility
        df['return'] = df['close'].pct_change()
        df['volatility'] = (df['high'] - df['low']) / df['close']
        
        print(f"  ✓ Loaded {len(df):,} S&P 500 records")
        print(f"  ✓ Date range: {df['date'].min()} to {df['date'].max()}")
        
        return df
    
    @staticmethod
    def load_depression_index(filepath):
        """Load Google Trends depression index (weekly data)"""
        print(f"\nLoading depression index from {filepath}...")
        df = pd.read_csv(filepath)
        
        # Standardize column names
        if 'Week' in df.columns:
            df = df.rename(columns={'Week': 'date'})
        if 'depression' in df.columns:
            df = df.rename(columns={'depression': 'depression_index'})
        
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"  ✓ Loaded {len(df):,} weekly depression index records")
        print(f"  ✓ Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  ✓ Index range: {df['depression_index'].min()} to {df['depression_index'].max()}")
        
        return df
    
    @staticmethod
    def load_depression_words(filepath):
        """Load depression word count from news (daily data)"""
        print(f"\nLoading depression word count from {filepath}...")
        df = pd.read_csv(filepath)
        
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"  ✓ Loaded {len(df):,} daily word count records")
        print(f"  ✓ Date range: {df['date'].min()} to {df['date'].max()}")
        
        return df
    
    @staticmethod
    def load_rainfall_data(filepath):
        """Load rainfall data"""
        print(f"\nLoading rainfall data from {filepath}...")
        df = pd.read_csv(filepath)
        
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"  ✓ Loaded {len(df):,} rainfall records")
        print(f"  ✓ Date range: {df['date'].min()} to {df['date'].max()}")
        
        return df


# ============================================================================
# DATA MERGING
# ============================================================================

class DataMerger:
    """Merge all data sources into a single daily dataset"""
    
    @staticmethod
    def aggregate_stock_data(stock_df):
        """Aggregate stock data to daily averages"""
        print("\nAggregating stock data to daily averages...")
        
        daily_stock = stock_df.groupby('date').agg({
            'open': 'mean',
            'high': 'mean',
            'low': 'mean',
            'close': 'mean',
            'volume': 'mean',
            'ticker': 'count'  # Number of stocks traded
        }).reset_index()
        
        daily_stock = daily_stock.rename(columns={
            'open': 'avg_stock_open',
            'high': 'avg_stock_high',
            'low': 'avg_stock_low',
            'close': 'avg_stock_close',
            'volume': 'avg_stock_volume',
            'ticker': 'num_stocks'
        })
        
        # Calculate price range (volatility proxy)
        daily_stock['price_range'] = daily_stock['avg_stock_high'] - daily_stock['avg_stock_low']
        
        print(f"  ✓ Aggregated to {len(daily_stock):,} daily records")
        
        return daily_stock
    
    @staticmethod
    def prepare_sp500_data(sp500_df):
        """Prepare S&P 500 data for merging"""
        sp500_prepared = sp500_df[['date', 'close', 'return', 'volatility']].copy()
        sp500_prepared = sp500_prepared.rename(columns={
            'close': 'sp500_close',
            'return': 'sp500_return',
            'volatility': 'sp500_volatility'
        })
        return sp500_prepared
    
    @staticmethod
    def forward_fill_depression_index(depression_df, date_range):
        """Forward fill weekly depression index to daily"""
        print("\nForward-filling weekly depression index to daily...")
        
        # Create daily date range
        daily_dates = pd.DataFrame({'date': pd.date_range(
            start=date_range[0],
            end=date_range[1],
            freq='D'
        )})
        
        # Merge and forward fill
        merged = daily_dates.merge(depression_df, on='date', how='left')
        merged['depression_index'] = merged['depression_index'].fillna(method='ffill')
        
        print(f"  ✓ Created {len(merged):,} daily depression index records")
        
        return merged
    
    @staticmethod
    def merge_all_data(stock_daily, sp500_prepared, depression_daily, 
                      depression_words, rainfall):
        """Merge all datasets into single daily dataset"""
        print("\nMerging all data sources...")
        
        # Start with stock data
        merged = stock_daily.copy()
        
        # Merge S&P 500
        merged = merged.merge(sp500_prepared, on='date', how='left')
        print(f"  ✓ Merged S&P 500 data")
        
        # Merge depression index
        merged = merged.merge(depression_daily, on='date', how='left')
        print(f"  ✓ Merged depression index")
        
        # Merge depression word count
        merged = merged.merge(depression_words, on='date', how='left')
        print(f"  ✓ Merged depression word count")
        
        # Merge rainfall
        merged = merged.merge(rainfall, on='date', how='left')
        print(f"  ✓ Merged rainfall data")
        
        # Sort by date
        merged = merged.sort_values('date').reset_index(drop=True)
        
        print(f"\n✓ Final merged dataset: {len(merged):,} records")
        print(f"  Date range: {merged['date'].min()} to {merged['date'].max()}")
        print(f"  Complete records: {merged.dropna().shape[0]:,}")
        
        return merged


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

class StatisticalAnalyzer:
    """Perform statistical correlation analysis"""
    
    @staticmethod
    def calculate_correlation_stats(x, y, var_x_name, var_y_name):
        """Calculate comprehensive correlation statistics"""
        
        # Remove NaN values
        mask = ~(pd.isna(x) | pd.isna(y))
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) < 3:
            return None
        
        # Calculate correlation and p-value
        corr, p_value = stats.pearsonr(x_clean, y_clean)
        
        # Calculate R²
        r_squared = corr ** 2
        
        # Calculate confidence interval using Fisher's z-transformation
        n = len(x_clean)
        fisher_z = np.arctanh(corr)
        se = 1 / np.sqrt(n - 3)
        z_critical = 1.96
        ci_lower = np.tanh(fisher_z - z_critical * se)
        ci_upper = np.tanh(fisher_z + z_critical * se)
        
        # Determine significance level
        if p_value < 0.001:
            significance = "***"
        elif p_value < 0.01:
            significance = "**"
        elif p_value < 0.05:
            significance = "*"
        else:
            significance = "ns"
        
        # Determine effect size
        abs_corr = abs(corr)
        if abs_corr >= 0.50:
            effect_size = "Large"
        elif abs_corr >= 0.30:
            effect_size = "Medium"
        elif abs_corr >= 0.10:
            effect_size = "Small"
        else:
            effect_size = "Negligible"
        
        # Bonferroni correction (α = 0.05/17 for 17 tests)
        bonferroni_alpha = 0.05 / 17
        bonferroni_significant = p_value < bonferroni_alpha
        
        # Calculate importance score (0-10)
        importance_score = 0
        if p_value < 0.001:
            importance_score += 3
        elif p_value < 0.01:
            importance_score += 2
        elif p_value < 0.05:
            importance_score += 1
        
        if abs_corr >= 0.30:
            importance_score += 3
        elif abs_corr >= 0.10:
            importance_score += 1
        
        if bonferroni_significant:
            importance_score += 2
        
        if n > 100:
            importance_score += 1
        
        return {
            'Variable_X': var_x_name,
            'Variable_Y': var_y_name,
            'Correlation': corr,
            'P_Value': p_value,
            'R_Squared': r_squared,
            'CI_Lower': ci_lower,
            'CI_Upper': ci_upper,
            'N': n,
            'Significance': significance,
            'Effect_Size': effect_size,
            'Bonferroni_Significant': bonferroni_significant,
            'Importance_Score': importance_score
        }
    
    @staticmethod
    def analyze_all_correlations(merged_df):
        """Analyze all relevant correlations"""
        print("\nCalculating correlation statistics...")
        
        correlations = []
        
        # Define correlation pairs to test
        test_pairs = [
            ('avg_stock_close', 'depression_index', 'Stock Close', 'Depression Index'),
            ('avg_stock_open', 'depression_index', 'Stock Open', 'Depression Index'),
            ('avg_stock_high', 'depression_index', 'Stock High', 'Depression Index'),
            ('avg_stock_low', 'depression_index', 'Stock Low', 'Depression Index'),
            ('avg_stock_volume', 'depression_index', 'Stock Volume', 'Depression Index'),
            ('price_range', 'depression_index', 'Price Range (Volatility)', 'Depression Index'),
            
            ('avg_stock_close', 'depression_word_count', 'Stock Close', 'Depression Word Count'),
            ('price_range', 'depression_word_count', 'Price Range (Volatility)', 'Depression Word Count'),
            
            ('sp500_close', 'depression_index', 'S&P 500 Close', 'Depression Index'),
            ('sp500_return', 'depression_index', 'S&P 500 Return', 'Depression Index'),
            ('sp500_volatility', 'depression_index', 'S&P 500 Volatility', 'Depression Index'),
            
            ('sp500_close', 'depression_word_count', 'S&P 500 Close', 'Depression Word Count'),
            ('sp500_return', 'depression_word_count', 'S&P 500 Return', 'Depression Word Count'),
            
            ('rainfall', 'avg_stock_close', 'Rainfall', 'Stock Close'),
            ('rainfall', 'depression_index', 'Rainfall', 'Depression Index'),
            ('rainfall', 'sp500_return', 'Rainfall', 'S&P 500 Return'),
            
            ('depression_index', 'depression_word_count', 'Depression Index', 'Depression Word Count'),
        ]
        
        for col_x, col_y, name_x, name_y in test_pairs:
            if col_x in merged_df.columns and col_y in merged_df.columns:
                result = StatisticalAnalyzer.calculate_correlation_stats(
                    merged_df[col_x], 
                    merged_df[col_y],
                    name_x,
                    name_y
                )
                if result:
                    correlations.append(result)
                    print(f"  ✓ {name_x} ↔ {name_y}: r={result['Correlation']:.3f}, p={result['P_Value']:.2e}")
        
        return pd.DataFrame(correlations)


# ============================================================================
# INDUSTRY ANALYSIS
# ============================================================================

class IndustryAnalyzer:
    """Analyze industry-specific patterns"""
    
    @staticmethod
    def analyze_industries(stock_df, depression_index_df):
        """Analyze volatility and depression correlation by industry"""
        print("\nAnalyzing industry patterns...")
        
        # Merge stock data with depression index
        stock_with_depression = stock_df.merge(
            depression_index_df[['date', 'depression_index']], 
            on='date', 
            how='left'
        )
        
        # Calculate daily price range for each stock
        stock_with_depression['price_range'] = (
            stock_with_depression['high'] - stock_with_depression['low']
        )
        
        # Group by industry
        industry_stats = stock_with_depression.groupby('industry').agg({
            'price_range': 'mean',
            'ticker': 'nunique',
            'close': 'mean',
            'depression_index': lambda x: x.corr(stock_with_depression.loc[x.index, 'price_range'])
        }).reset_index()
        
        industry_stats.columns = ['Industry', 'Avg_Volatility', 'Num_Stocks', 
                                  'Avg_Price', 'Depression_Correlation']
        
        # Sort by volatility
        industry_stats = industry_stats.sort_values('Avg_Volatility', ascending=False)
        industry_stats['Volatility_Rank'] = range(1, len(industry_stats) + 1)
        
        print(f"  ✓ Analyzed {len(industry_stats)} industries")
        
        return industry_stats


# ============================================================================
# VISUALIZATION
# ============================================================================

class Visualizer:
    """Create all analysis visualizations"""
    
    @staticmethod
    def plot_correlation_heatmap(correlation_df, output_path):
        """Create correlation importance heatmap"""
        print("\nGenerating correlation heatmap...")
        
        # Prepare data for heatmap
        top_corr = correlation_df.nlargest(10, 'Importance_Score')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create horizontal bar chart
        colors = ['green' if x >= 8 else 'orange' if x >= 5 else 'red' 
                 for x in top_corr['Importance_Score']]
        
        y_labels = [f"{row['Variable_X']} ↔ {row['Variable_Y']}" 
                   for _, row in top_corr.iterrows()]
        
        bars = ax.barh(range(len(top_corr)), top_corr['Importance_Score'], color=colors)
        ax.set_yticks(range(len(top_corr)))
        ax.set_yticklabels(y_labels, fontsize=10)
        ax.set_xlabel('Importance Score (0-10)', fontsize=12, fontweight='bold')
        ax.set_title('Top 10 Correlation Importance Rankings', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim(0, 10)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, top_corr['Importance_Score'])):
            ax.text(score + 0.2, bar.get_y() + bar.get_height()/2, 
                   f'{score:.0f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved to {output_path}")
    
    @staticmethod
    def plot_time_series_overview(merged_df, output_path):
        """Create time series overview"""
        print("\nGenerating time series overview...")
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        fig.suptitle('Time Series Overview: Depression & Market Metrics', 
                    fontsize=16, fontweight='bold')
        
        # 1. Stock prices and S&P 500
        ax1 = axes[0]
        ax1_twin = ax1.twinx()
        ax1.plot(merged_df['date'], merged_df['avg_stock_close'], 
                label='Avg Stock Close', color='blue', linewidth=1.5)
        ax1_twin.plot(merged_df['date'], merged_df['sp500_close'], 
                     label='S&P 500', color='orange', linewidth=1.5)
        ax1.set_ylabel('Avg Stock Close', color='blue', fontweight='bold')
        ax1_twin.set_ylabel('S&P 500', color='orange', fontweight='bold')
        ax1.set_title('Stock Prices & S&P 500', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        
        # 2. Volatility
        ax2 = axes[1]
        ax2.plot(merged_df['date'], merged_df['price_range'], 
                label='Price Range', color='red', linewidth=1.5)
        ax2.set_ylabel('Price Range (Volatility)', fontweight='bold')
        ax2.set_title('Market Volatility', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. Depression metrics
        ax3 = axes[2]
        ax3_twin = ax3.twinx()
        ax3.plot(merged_df['date'], merged_df['depression_index'], 
                label='Depression Index', color='purple', linewidth=1.5)
        ax3_twin.plot(merged_df['date'], merged_df['depression_word_count'], 
                     label='Word Count', color='green', linewidth=1, alpha=0.7)
        ax3.set_ylabel('Depression Index', color='purple', fontweight='bold')
        ax3_twin.set_ylabel('Word Count', color='green', fontweight='bold')
        ax3.set_title('Depression Metrics', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        
        # 4. Rainfall
        ax4 = axes[3]
        ax4.bar(merged_df['date'], merged_df['rainfall'], 
               color='skyblue', alpha=0.7, width=1)
        ax4.set_ylabel('Rainfall (mm)', fontweight='bold')
        ax4.set_xlabel('Date', fontweight='bold')
        ax4.set_title('Rainfall', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved to {output_path}")
    
    @staticmethod
    def plot_industry_analysis(industry_df, output_path):
        """Create industry analysis visualization"""
        print("\nGenerating industry analysis...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Industry Analysis: Volatility & Depression Correlation', 
                    fontsize=14, fontweight='bold')
        
        # Top 20 most volatile industries
        top_20 = industry_df.head(20)
        
        ax1 = axes[0]
        bars = ax1.barh(range(len(top_20)), top_20['Avg_Volatility'], 
                       color='steelblue')
        ax1.set_yticks(range(len(top_20)))
        ax1.set_yticklabels(top_20['Industry'], fontsize=9)
        ax1.set_xlabel('Average Volatility (Price Range)', fontweight='bold')
        ax1.set_title('Top 20 Most Volatile Industries', fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        ax1.invert_yaxis()
        
        # Industry correlation with depression
        ax2 = axes[1]
        colors = ['red' if x < 0 else 'green' 
                 for x in top_20['Depression_Correlation']]
        ax2.barh(range(len(top_20)), top_20['Depression_Correlation'], 
                color=colors, alpha=0.7)
        ax2.set_yticks(range(len(top_20)))
        ax2.set_yticklabels(top_20['Industry'], fontsize=9)
        ax2.set_xlabel('Correlation with Depression Index', fontweight='bold')
        ax2.set_title('Depression Correlation by Industry', fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax2.grid(axis='x', alpha=0.3)
        ax2.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved to {output_path}")


# ============================================================================
# CSV EXPORT
# ============================================================================

class CSVExporter:
    """Export all results to CSV files"""
    
    @staticmethod
    def export_merged_data(merged_df, output_dir):
        """Export merged daily dataset"""
        filepath = os.path.join(output_dir, 'merged_time_series_data.csv')
        merged_df.to_csv(filepath, index=False)
        print(f"  ✓ Exported merged data: {filepath}")
        return filepath
    
    @staticmethod
    def export_correlation_stats(correlation_df, output_dir):
        """Export correlation statistics"""
        filepath = os.path.join(output_dir, 'correlation_statistics_full.csv')
        correlation_df.to_csv(filepath, index=False)
        print(f"  ✓ Exported correlation stats: {filepath}")
        return filepath
    
    @staticmethod
    def export_industry_analysis(industry_df, output_dir):
        """Export industry analysis"""
        filepath = os.path.join(output_dir, 'industry_analysis.csv')
        industry_df.to_csv(filepath, index=False)
        print(f"  ✓ Exported industry analysis: {filepath}")
        return filepath
    
    @staticmethod
    def export_summary_stats(merged_df, correlation_df, output_dir):
        """Export summary statistics"""
        filepath = os.path.join(output_dir, 'summary_statistics.csv')
        
        summary_data = []
        
        # Overall statistics
        for col in ['avg_stock_close', 'sp500_close', 'depression_index', 
                   'depression_word_count', 'price_range', 'rainfall']:
            if col in merged_df.columns:
                summary_data.append({
                    'Metric': col.replace('_', ' ').title(),
                    'Mean': merged_df[col].mean(),
                    'Std': merged_df[col].std(),
                    'Min': merged_df[col].min(),
                    'Max': merged_df[col].max(),
                    'Count': merged_df[col].notna().sum()
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(filepath, index=False)
        print(f"  ✓ Exported summary stats: {filepath}")
        return filepath
    
    @staticmethod
    def export_top_correlations(correlation_df, output_dir):
        """Export top correlations only"""
        filepath = os.path.join(output_dir, 'top_correlations.csv')
        
        # Get significant correlations
        top_corr = correlation_df[
            correlation_df['Bonferroni_Significant'] == True
        ].sort_values('Importance_Score', ascending=False)
        
        top_corr.to_csv(filepath, index=False)
        print(f"  ✓ Exported top correlations: {filepath}")
        return filepath


# ============================================================================
# MAIN EXECUTION
# ============================================================================

class TimeSeriesAnalysis:
    """Main analysis orchestrator"""
    
    def __init__(self):
        self.config = Config()
        self.config.setup_directories()
        
        self.stock_df = None
        self.sp500_df = None
        self.depression_index_df = None
        self.depression_words_df = None
        self.rainfall_df = None
        self.merged_df = None
        self.correlation_df = None
        self.industry_df = None
    
    def run_full_analysis(self):
        """Execute complete analysis pipeline"""
        
        print("="*70)
        print("TIME SERIES ANALYSIS: DEPRESSION & STOCK MARKET")
        print("="*70)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        # Step 1: Load data
        print("\n" + "="*70)
        print("STEP 1: LOADING DATA")
        print("="*70)
        
        self.stock_df = DataLoader.load_stock_data(Config.STOCK_DATA)
        self.sp500_df = DataLoader.load_sp500_data(Config.SP500_DATA)
        self.depression_index_df = DataLoader.load_depression_index(Config.DEPRESSION_INDEX)
        self.depression_words_df = DataLoader.load_depression_words(Config.DEPRESSION_WORDS)
        self.rainfall_df = DataLoader.load_rainfall_data(Config.RAINFALL_DATA)
        
        # Step 2: Merge data
        print("\n" + "="*70)
        print("STEP 2: MERGING DATA")
        print("="*70)
        
        stock_daily = DataMerger.aggregate_stock_data(self.stock_df)
        sp500_prepared = DataMerger.prepare_sp500_data(self.sp500_df)
        
        # Determine date range
        date_range = (
            min(stock_daily['date'].min(), sp500_prepared['date'].min()),
            max(stock_daily['date'].max(), sp500_prepared['date'].max())
        )
        
        depression_daily = DataMerger.forward_fill_depression_index(
            self.depression_index_df, date_range
        )
        
        self.merged_df = DataMerger.merge_all_data(
            stock_daily, sp500_prepared, depression_daily,
            self.depression_words_df, self.rainfall_df
        )
        
        # Step 3: Statistical analysis
        print("\n" + "="*70)
        print("STEP 3: STATISTICAL ANALYSIS")
        print("="*70)
        
        self.correlation_df = StatisticalAnalyzer.analyze_all_correlations(self.merged_df)
        
        print(f"\n✓ Calculated {len(self.correlation_df)} correlations")
        print(f"  Significant (p<0.05): {(self.correlation_df['P_Value'] < 0.05).sum()}")
        print(f"  Highly significant (p<0.001): {(self.correlation_df['P_Value'] < 0.001).sum()}")
        print(f"  Bonferroni survivors: {self.correlation_df['Bonferroni_Significant'].sum()}")
        
        # Step 4: Industry analysis
        print("\n" + "="*70)
        print("STEP 4: INDUSTRY ANALYSIS")
        print("="*70)
        
        self.industry_df = IndustryAnalyzer.analyze_industries(
            self.stock_df, self.depression_index_df
        )
        
        # Step 5: Export to CSV
        print("\n" + "="*70)
        print("STEP 5: EXPORTING TO CSV")
        print("="*70)
        
        CSVExporter.export_merged_data(self.merged_df, Config.DATA_DIR)
        CSVExporter.export_correlation_stats(self.correlation_df, Config.DATA_DIR)
        CSVExporter.export_industry_analysis(self.industry_df, Config.DATA_DIR)
        CSVExporter.export_summary_stats(self.merged_df, self.correlation_df, Config.DATA_DIR)
        CSVExporter.export_top_correlations(self.correlation_df, Config.DATA_DIR)
        
        # Step 6: Create visualizations
        print("\n" + "="*70)
        print("STEP 6: CREATING VISUALIZATIONS")
        print("="*70)
        
        Visualizer.plot_correlation_heatmap(
            self.correlation_df,
            os.path.join(Config.FIGURES_DIR, 'correlation_importance_ranking.png')
        )
        
        Visualizer.plot_time_series_overview(
            self.merged_df,
            os.path.join(Config.FIGURES_DIR, 'time_series_overview.png')
        )
        
        Visualizer.plot_industry_analysis(
            self.industry_df,
            os.path.join(Config.FIGURES_DIR, 'industry_analysis.png')
        )
        
        # Step 7: Print summary
        self.print_summary()
        
        print("\n" + "="*70)
        print("✓ ANALYSIS COMPLETE!")
        print("="*70)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nResults saved to: {Config.OUTPUT_DIR}/")
        print(f"  - CSV files: {Config.DATA_DIR}/")
        print(f"  - Figures: {Config.FIGURES_DIR}/")
        print("="*70)
    
    def print_summary(self):
        """Print analysis summary"""
        print("\n" + "="*70)
        print("ANALYSIS SUMMARY")
        print("="*70)
        
        print("\nDATA COVERAGE:")
        print(f"  Total daily records: {len(self.merged_df):,}")
        print(f"  Date range: {self.merged_df['date'].min()} to {self.merged_df['date'].max()}")
        print(f"  Complete records: {self.merged_df.dropna().shape[0]:,}")
        
        print("\nTOP 3 CORRELATIONS:")
        top_3 = self.correlation_df.nlargest(3, 'Importance_Score')
        for i, row in enumerate(top_3.itertuples(), 1):
            print(f"\n  {i}. {row.Variable_X} ↔ {row.Variable_Y}")
            print(f"     r = {row.Correlation:.4f}, p = {row.P_Value:.2e}")
            print(f"     R² = {row.R_Squared:.1%}, Score = {row.Importance_Score}/10")
            print(f"     Bonferroni: {'✓ PASS' if row.Bonferroni_Significant else '✗ FAIL'}")
        
        print("\nINDUSTRY INSIGHTS:")
        print(f"  Total industries analyzed: {len(self.industry_df)}")
        top_volatile = self.industry_df.head(3)
        print(f"  Top 3 most volatile:")
        for i, row in enumerate(top_volatile.itertuples(), 1):
            print(f"    {i}. {row.Industry} (volatility: {row.Avg_Volatility:.2f})")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Create and run analysis
    analysis = TimeSeriesAnalysis()
    analysis.run_full_analysis()
