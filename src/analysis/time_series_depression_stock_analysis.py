"""
Time Series Analysis: Depression Data, Stock Market, and Environmental Factors
===============================================================================
This script performs comprehensive time series analysis examining relationships between:
1. Stock market metrics (Open, High, Low, Close, Volume) and depression indicators
2. S&P 500 performance and depression word counts/index
3. Rainfall impact on stock prices and depression
4. Industry-specific patterns using stock_data_wiki

Author: Data Engineering Analysis
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class DepressionStockAnalyzer:
    """
    Main class for analyzing relationships between depression indicators,
    stock market data, and environmental factors
    """
    
    def __init__(self, data_dir='csv_exports', reports_dir='../data_quality_reports'):
        """Initialize the analyzer with data directories"""
        self.data_dir = data_dir
        self.reports_dir = reports_dir
        self.stock_data = None
        self.sp500_data = None
        self.depression_index = None
        self.depression_word_count = None
        self.rainfall_data = None
        self.merged_data = None
        
    def load_data(self):
        """Load all required datasets"""
        print("Loading datasets...")
        
        # Load stock data with wiki information
        self.stock_data = pd.read_csv(f'{self.data_dir}/stock_data_wiki_clean.csv')
        self.stock_data['date'] = pd.to_datetime(self.stock_data['date'])
        
        # Load S&P 500 data
        self.sp500_data = pd.read_csv(f'{self.reports_dir}/SP500_data.csv')
        self.sp500_data['Date'] = pd.to_datetime(self.sp500_data['Date'])
        self.sp500_data.rename(columns={'Date': 'date'}, inplace=True)
        
        # Load depression index (weekly data)
        self.depression_index = pd.read_csv(f'{self.reports_dir}/Depression_index_data.csv')
        self.depression_index['date'] = pd.to_datetime(self.depression_index['date'])
        
        # Load depression word count (daily)
        self.depression_word_count = pd.read_csv(f'{self.data_dir}/ccnews_depression_daily_count_final.csv')
        self.depression_word_count['date'] = pd.to_datetime(self.depression_word_count['date'])
        
        # Load rainfall data
        self.rainfall_data = pd.read_csv(f'{self.reports_dir}/Rainfall_data.csv')
        self.rainfall_data['Date'] = pd.to_datetime(self.rainfall_data['Date'])
        self.rainfall_data.rename(columns={'Date': 'date'}, inplace=True)
        
        # Calculate national average rainfall
        rainfall_cols = [col for col in self.rainfall_data.columns if col != 'date']
        self.rainfall_data['avg_national_rainfall'] = self.rainfall_data[rainfall_cols].mean(axis=1)
        
        print(f"✓ Stock data loaded: {len(self.stock_data)} records")
        print(f"✓ S&P 500 data loaded: {len(self.sp500_data)} records")
        print(f"✓ Depression index loaded: {len(self.depression_index)} records")
        print(f"✓ Depression word count loaded: {len(self.depression_word_count)} records")
        print(f"✓ Rainfall data loaded: {len(self.rainfall_data)} records")
        
    def prepare_merged_dataset(self):
        """Merge all datasets for comprehensive analysis"""
        print("\nPreparing merged dataset...")
        
        # Aggregate stock data by date
        daily_stock = self.stock_data.groupby('date').agg({
            'open': 'mean',
            'high': 'mean',
            'low': 'mean',
            'close': 'mean',
            'volume': 'sum',
            'ticker': 'count'  # Number of stocks
        }).reset_index()
        daily_stock.rename(columns={'ticker': 'num_stocks'}, inplace=True)
        
        # Start with daily stock data
        merged = daily_stock.copy()
        
        # Merge S&P 500 data
        merged = merged.merge(
            self.sp500_data[['date', 'Close_^GSPC', 'Return', 'Volatility_7']],
            on='date',
            how='left'
        )
        
        # Merge depression word count
        merged = merged.merge(
            self.depression_word_count[['date', 'depression_word_count', 'total_articles']],
            on='date',
            how='left'
        )
        
        # For depression index (weekly), forward fill to match daily data
        depression_daily = self.depression_index.set_index('date').reindex(
            pd.date_range(start=self.depression_index['date'].min(),
                         end=self.depression_index['date'].max(),
                         freq='D')
        ).fillna(method='ffill').reset_index()
        depression_daily.columns = ['date', 'depression_index']
        
        merged = merged.merge(depression_daily, on='date', how='left')
        
        # Merge rainfall data
        merged = merged.merge(
            self.rainfall_data[['date', 'avg_national_rainfall']],
            on='date',
            how='left'
        )
        
        # Fill missing depression word counts with 0
        merged['depression_word_count'] = merged['depression_word_count'].fillna(0)
        merged['total_articles'] = merged['total_articles'].fillna(0)
        
        # Calculate additional metrics
        merged['price_range'] = merged['high'] - merged['low']
        merged['price_change_pct'] = ((merged['close'] - merged['open']) / merged['open']) * 100
        
        # Create depression intensity categories
        merged['depression_index_category'] = pd.cut(
            merged['depression_index'],
            bins=[0, 60, 75, 90, 100],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        self.merged_data = merged
        print(f"✓ Merged dataset created: {len(self.merged_data)} records")
        print(f"Date range: {self.merged_data['date'].min()} to {self.merged_data['date'].max()}")
        
    def analyze_stock_depression_relationship(self):
        """
        Analysis 1: Stock metrics (open, high, low, close, volume) 
        vs Depression index/word count
        """
        print("\n" + "="*80)
        print("ANALYSIS 1: Stock Market Metrics vs Depression Indicators")
        print("="*80)
        
        # Filter data with valid depression measurements
        df = self.merged_data.dropna(subset=['depression_index', 'depression_word_count'])
        
        # Calculate correlations
        correlation_vars = ['open', 'high', 'low', 'close', 'volume', 'price_range']
        depression_vars = ['depression_index', 'depression_word_count']
        
        print("\nCorrelation Analysis:")
        print("-" * 80)
        
        correlations = []
        for stock_var in correlation_vars:
            for dep_var in depression_vars:
                corr = df[[stock_var, dep_var]].corr().iloc[0, 1]
                correlations.append({
                    'Stock Metric': stock_var,
                    'Depression Metric': dep_var,
                    'Correlation': corr
                })
                print(f"{stock_var:15} vs {dep_var:25} : {corr:>8.4f}")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Stock Market Metrics vs Depression Index', fontsize=16, fontweight='bold')
        
        metrics = ['open', 'high', 'low', 'close', 'volume', 'price_range']
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            
            # Scatter plot with trend line
            valid_data = df[[metric, 'depression_index']].dropna()
            ax.scatter(valid_data['depression_index'], valid_data[metric], 
                      alpha=0.3, s=10)
            
            # Add trend line
            z = np.polyfit(valid_data['depression_index'], valid_data[metric], 1)
            p = np.poly1d(z)
            ax.plot(valid_data['depression_index'].sort_values(), 
                   p(valid_data['depression_index'].sort_values()), 
                   "r--", alpha=0.8, linewidth=2)
            
            ax.set_xlabel('Depression Index', fontsize=10)
            ax.set_ylabel(metric.upper(), fontsize=10)
            ax.set_title(f'{metric.upper()} vs Depression Index', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('analysis_1_stock_vs_depression.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved: analysis_1_stock_vs_depression.png")
        
        # Time series comparison
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        # Plot 1: Close price and Depression Index
        ax1 = axes[0]
        ax1_twin = ax1.twinx()
        ax1.plot(df['date'], df['close'], color='blue', label='Avg Close Price', linewidth=1.5)
        ax1_twin.plot(df['date'], df['depression_index'], color='red', 
                     label='Depression Index', linewidth=1.5, alpha=0.7)
        ax1.set_ylabel('Average Close Price ($)', color='blue', fontsize=11)
        ax1_twin.set_ylabel('Depression Index', color='red', fontsize=11)
        ax1.set_title('Stock Close Price vs Depression Index Over Time', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        
        # Plot 2: Volume and Depression Word Count
        ax2 = axes[1]
        ax2_twin = ax2.twinx()
        ax2.plot(df['date'], df['volume'], color='green', label='Total Volume', linewidth=1.5)
        ax2_twin.plot(df['date'], df['depression_word_count'], color='orange', 
                     label='Depression Word Count', linewidth=1.5, alpha=0.7)
        ax2.set_ylabel('Total Trading Volume', color='green', fontsize=11)
        ax2_twin.set_ylabel('Depression Word Count', color='orange', fontsize=11)
        ax2.set_title('Trading Volume vs Depression Word Count Over Time', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        
        # Plot 3: Price Range and Depression Index
        ax3 = axes[2]
        ax3_twin = ax3.twinx()
        ax3.plot(df['date'], df['price_range'], color='purple', label='Price Range (High-Low)', linewidth=1.5)
        ax3_twin.plot(df['date'], df['depression_index'], color='red', 
                     label='Depression Index', linewidth=1.5, alpha=0.7)
        ax3.set_ylabel('Price Range ($)', color='purple', fontsize=11)
        ax3_twin.set_ylabel('Depression Index', color='red', fontsize=11)
        ax3.set_xlabel('Date', fontsize=11)
        ax3.set_title('Price Volatility (Range) vs Depression Index Over Time', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig('analysis_1_time_series.png', dpi=300, bbox_inches='tight')
        print("✓ Visualization saved: analysis_1_time_series.png")
        
        return pd.DataFrame(correlations)
    
    def analyze_sp500_depression(self):
        """
        Analysis 2: S&P 500 performance vs Depression indicators
        """
        print("\n" + "="*80)
        print("ANALYSIS 2: S&P 500 vs Depression Indicators")
        print("="*80)
        
        df = self.merged_data.dropna(subset=['Close_^GSPC', 'depression_index'])
        
        # Correlation analysis
        print("\nCorrelation Analysis:")
        print("-" * 80)
        
        sp500_metrics = ['Close_^GSPC', 'Return', 'Volatility_7']
        depression_metrics = ['depression_index', 'depression_word_count']
        
        correlations = []
        for sp_metric in sp500_metrics:
            for dep_metric in depression_metrics:
                valid = df[[sp_metric, dep_metric]].dropna()
                if len(valid) > 0:
                    corr = valid.corr().iloc[0, 1]
                    correlations.append({
                        'S&P 500 Metric': sp_metric,
                        'Depression Metric': dep_metric,
                        'Correlation': corr
                    })
                    print(f"{sp_metric:20} vs {dep_metric:25} : {corr:>8.4f}")
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('S&P 500 Performance vs Depression Indicators', fontsize=16, fontweight='bold')
        
        # Plot 1: S&P 500 Close vs Depression Index
        ax = axes[0, 0]
        ax_twin = ax.twinx()
        valid = df.dropna(subset=['Close_^GSPC', 'depression_index'])
        ax.plot(valid['date'], valid['Close_^GSPC'], color='blue', label='S&P 500 Close', linewidth=2)
        ax_twin.plot(valid['date'], valid['depression_index'], color='red', 
                    label='Depression Index', linewidth=2, alpha=0.7)
        ax.set_ylabel('S&P 500 Close Price', color='blue', fontsize=10)
        ax_twin.set_ylabel('Depression Index', color='red', fontsize=10)
        ax.set_title('S&P 500 Close vs Depression Index', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        ax_twin.legend(loc='upper right')
        
        # Plot 2: S&P 500 Returns vs Depression Word Count
        ax = axes[0, 1]
        valid = df.dropna(subset=['Return', 'depression_word_count'])
        scatter = ax.scatter(valid['depression_word_count'], valid['Return'], 
                           c=valid['Volatility_7'], cmap='viridis', alpha=0.5, s=30)
        ax.set_xlabel('Depression Word Count', fontsize=10)
        ax.set_ylabel('S&P 500 Daily Return', fontsize=10)
        ax.set_title('S&P 500 Returns vs Depression Word Count', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='7-Day Volatility')
        
        # Add trend line
        if len(valid) > 0:
            z = np.polyfit(valid['depression_word_count'], valid['Return'], 1)
            p = np.poly1d(z)
            ax.plot(valid['depression_word_count'].sort_values(), 
                   p(valid['depression_word_count'].sort_values()), 
                   "r--", alpha=0.8, linewidth=2)
        
        # Plot 3: Volatility vs Depression Index
        ax = axes[1, 0]
        valid = df.dropna(subset=['Volatility_7', 'depression_index'])
        ax.scatter(valid['depression_index'], valid['Volatility_7'], alpha=0.3, s=20)
        ax.set_xlabel('Depression Index', fontsize=10)
        ax.set_ylabel('S&P 500 7-Day Volatility', fontsize=10)
        ax.set_title('Market Volatility vs Depression Index', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        if len(valid) > 0:
            z = np.polyfit(valid['depression_index'], valid['Volatility_7'], 1)
            p = np.poly1d(z)
            ax.plot(valid['depression_index'].sort_values(), 
                   p(valid['depression_index'].sort_values()), 
                   "r--", alpha=0.8, linewidth=2)
        
        # Plot 4: Depression categories vs S&P 500 performance
        ax = axes[1, 1]
        valid = df.dropna(subset=['depression_index_category', 'Return'])
        category_returns = valid.groupby('depression_index_category')['Return'].mean()
        colors = ['green', 'yellow', 'orange', 'red']
        ax.bar(range(len(category_returns)), category_returns.values, color=colors, alpha=0.7)
        ax.set_xticks(range(len(category_returns)))
        ax.set_xticklabels(category_returns.index, fontsize=10)
        ax.set_ylabel('Average S&P 500 Return', fontsize=10)
        ax.set_xlabel('Depression Index Category', fontsize=10)
        ax.set_title('Average S&P 500 Returns by Depression Level', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        
        plt.tight_layout()
        plt.savefig('analysis_2_sp500_depression.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved: analysis_2_sp500_depression.png")
        
        return pd.DataFrame(correlations)
    
    def analyze_rainfall_impact(self):
        """
        Analysis 3: Impact of rainfall on stock prices and depression
        """
        print("\n" + "="*80)
        print("ANALYSIS 3: Rainfall Impact on Stocks and Depression")
        print("="*80)
        
        df = self.merged_data.dropna(subset=['avg_national_rainfall'])
        
        # Correlation analysis
        print("\nCorrelation Analysis:")
        print("-" * 80)
        
        variables = ['close', 'volume', 'Return', 'depression_index', 'depression_word_count']
        
        correlations = []
        for var in variables:
            valid = df[['avg_national_rainfall', var]].dropna()
            if len(valid) > 0:
                corr = valid.corr().iloc[0, 1]
                correlations.append({
                    'Variable': var,
                    'Correlation with Rainfall': corr
                })
                print(f"Rainfall vs {var:25} : {corr:>8.4f}")
        
        # Create rainfall categories
        df['rainfall_category'] = pd.cut(
            df['avg_national_rainfall'],
            bins=[0, 1, 5, 10, 100],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Rainfall Impact on Stock Market and Depression', fontsize=16, fontweight='bold')
        
        # Plot 1: Rainfall over time with stock close price
        ax = axes[0, 0]
        ax_twin = ax.twinx()
        valid = df.dropna(subset=['avg_national_rainfall', 'close'])
        ax.plot(valid['date'], valid['avg_national_rainfall'], color='blue', 
               label='Avg National Rainfall', linewidth=1.5, alpha=0.7)
        ax_twin.plot(valid['date'], valid['close'], color='green', 
                    label='Avg Close Price', linewidth=1.5)
        ax.set_ylabel('Average Rainfall (mm)', color='blue', fontsize=10)
        ax_twin.set_ylabel('Average Close Price ($)', color='green', fontsize=10)
        ax.set_title('Rainfall and Stock Prices Over Time', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        ax_twin.legend(loc='upper right')
        
        # Plot 2: Stock performance by rainfall category
        ax = axes[0, 1]
        valid = df.dropna(subset=['rainfall_category', 'price_change_pct'])
        rainfall_performance = valid.groupby('rainfall_category')['price_change_pct'].mean()
        colors = ['lightblue', 'blue', 'darkblue', 'navy']
        ax.bar(range(len(rainfall_performance)), rainfall_performance.values, 
              color=colors, alpha=0.7)
        ax.set_xticks(range(len(rainfall_performance)))
        ax.set_xticklabels(rainfall_performance.index, fontsize=10)
        ax.set_ylabel('Average Price Change (%)', fontsize=10)
        ax.set_xlabel('Rainfall Category', fontsize=10)
        ax.set_title('Stock Price Change by Rainfall Level', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        
        # Plot 3: Rainfall vs Depression Index
        ax = axes[1, 0]
        valid = df.dropna(subset=['avg_national_rainfall', 'depression_index'])
        ax.scatter(valid['avg_national_rainfall'], valid['depression_index'], 
                  alpha=0.3, s=20, color='purple')
        ax.set_xlabel('Average National Rainfall (mm)', fontsize=10)
        ax.set_ylabel('Depression Index', fontsize=10)
        ax.set_title('Rainfall vs Depression Index', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        if len(valid) > 0:
            z = np.polyfit(valid['avg_national_rainfall'], valid['depression_index'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(valid['avg_national_rainfall'].min(), 
                               valid['avg_national_rainfall'].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
        
        # Plot 4: Depression by rainfall category
        ax = axes[1, 1]
        valid = df.dropna(subset=['rainfall_category', 'depression_index'])
        rainfall_depression = valid.groupby('rainfall_category')['depression_index'].mean()
        ax.bar(range(len(rainfall_depression)), rainfall_depression.values, 
              color=colors, alpha=0.7)
        ax.set_xticks(range(len(rainfall_depression)))
        ax.set_xticklabels(rainfall_depression.index, fontsize=10)
        ax.set_ylabel('Average Depression Index', fontsize=10)
        ax.set_xlabel('Rainfall Category', fontsize=10)
        ax.set_title('Depression Index by Rainfall Level', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('analysis_3_rainfall_impact.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved: analysis_3_rainfall_impact.png")
        
        return pd.DataFrame(correlations)
    
    def analyze_industry_patterns(self):
        """
        Analysis 4: Industry-specific analysis using stock_data_wiki
        """
        print("\n" + "="*80)
        print("ANALYSIS 4: Industry-Specific Stock Performance vs Depression")
        print("="*80)
        
        # Merge stock data with depression data
        stock_with_depression = self.stock_data.merge(
            self.depression_word_count[['date', 'depression_word_count']],
            on='date',
            how='left'
        )
        
        # Merge with depression index (need to expand weekly to daily)
        depression_daily = self.depression_index.set_index('date').reindex(
            pd.date_range(start=self.depression_index['date'].min(),
                         end=self.depression_index['date'].max(),
                         freq='D')
        ).fillna(method='ffill').reset_index()
        depression_daily.columns = ['date', 'depression_index']
        
        stock_with_depression = stock_with_depression.merge(
            depression_daily,
            on='date',
            how='left'
        )
        
        # Calculate price change percentage
        stock_with_depression['price_change_pct'] = (
            (stock_with_depression['close'] - stock_with_depression['open']) / 
            stock_with_depression['open']
        ) * 100
        
        # Industry analysis
        print("\nIndustry Performance Summary:")
        print("-" * 80)
        
        industry_stats = stock_with_depression.groupby('industry').agg({
            'price_change_pct': ['mean', 'std', 'count'],
            'volume': 'sum',
            'close': 'mean'
        }).round(4)
        
        industry_stats.columns = ['Avg_Price_Change_%', 'Std_Price_Change', 
                                 'Trading_Days', 'Total_Volume', 'Avg_Close_Price']
        industry_stats = industry_stats.sort_values('Avg_Price_Change_%', ascending=False)
        
        print(industry_stats.head(20).to_string())
        
        # Top industries by sector
        print("\n\nTop Industries by Sector:")
        print("-" * 80)
        
        sector_industry = stock_with_depression.groupby(['sector', 'industry']).agg({
            'price_change_pct': 'mean',
            'volume': 'sum',
            'ticker': 'nunique'
        }).round(4)
        sector_industry.columns = ['Avg_Price_Change_%', 'Total_Volume', 'Num_Companies']
        
        for sector in stock_with_depression['sector'].unique()[:5]:  # Top 5 sectors
            print(f"\n{sector}:")
            sector_data = sector_industry.loc[sector].sort_values('Avg_Price_Change_%', ascending=False).head(3)
            print(sector_data.to_string())
        
        # Correlation between industry performance and depression
        industry_depression_corr = stock_with_depression.groupby('industry').apply(
            lambda x: x[['price_change_pct', 'depression_index']].dropna().corr().iloc[0, 1]
            if len(x.dropna(subset=['price_change_pct', 'depression_index'])) > 10 else np.nan
        ).dropna().sort_values()
        
        print("\n\nIndustries Most Correlated with Depression Index:")
        print("-" * 80)
        print("Most Negatively Correlated (Depression ↑, Performance ↓):")
        print(industry_depression_corr.head(10).to_string())
        print("\nMost Positively Correlated (Depression ↑, Performance ↑):")
        print(industry_depression_corr.tail(10).to_string())
        
        # Create visualizations
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Plot 1: Top 15 industries by average price change
        ax1 = fig.add_subplot(gs[0, :])
        top_15_industries = industry_stats.head(15)
        colors_pos_neg = ['green' if x > 0 else 'red' for x in top_15_industries['Avg_Price_Change_%']]
        ax1.barh(range(len(top_15_industries)), top_15_industries['Avg_Price_Change_%'].values,
                color=colors_pos_neg, alpha=0.7)
        ax1.set_yticks(range(len(top_15_industries)))
        ax1.set_yticklabels(top_15_industries.index, fontsize=9)
        ax1.set_xlabel('Average Price Change (%)', fontsize=11)
        ax1.set_title('Top 15 Industries by Average Price Performance', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
        
        # Plot 2: Sector performance
        ax2 = fig.add_subplot(gs[1, 0])
        sector_performance = stock_with_depression.groupby('sector')['price_change_pct'].mean().sort_values()
        colors_sectors = ['green' if x > 0 else 'red' for x in sector_performance.values]
        ax2.barh(range(len(sector_performance)), sector_performance.values, 
                color=colors_sectors, alpha=0.7)
        ax2.set_yticks(range(len(sector_performance)))
        ax2.set_yticklabels(sector_performance.index, fontsize=9)
        ax2.set_xlabel('Average Price Change (%)', fontsize=10)
        ax2.set_title('Sector Performance', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
        
        # Plot 3: Industries vs Depression correlation
        ax3 = fig.add_subplot(gs[1, 1])
        top_corr = industry_depression_corr.tail(10)
        bottom_corr = industry_depression_corr.head(10)
        all_corr = pd.concat([bottom_corr, top_corr])
        colors_corr = ['red' if x < 0 else 'blue' for x in all_corr.values]
        ax3.barh(range(len(all_corr)), all_corr.values, color=colors_corr, alpha=0.7)
        ax3.set_yticks(range(len(all_corr)))
        ax3.set_yticklabels(all_corr.index, fontsize=8)
        ax3.set_xlabel('Correlation with Depression Index', fontsize=10)
        ax3.set_title('Industries: Correlation with Depression Index', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        ax3.axvline(x=0, color='black', linestyle='--', linewidth=1)
        
        # Plot 4: Time series of top 5 sectors
        ax4 = fig.add_subplot(gs[2, :])
        top_sectors = stock_with_depression.groupby('sector')['volume'].sum().nlargest(5).index
        
        for sector in top_sectors:
            sector_data = stock_with_depression[stock_with_depression['sector'] == sector]
            daily_sector = sector_data.groupby('date')['close'].mean()
            ax4.plot(daily_sector.index, daily_sector.values, label=sector, linewidth=2, alpha=0.7)
        
        ax4.set_xlabel('Date', fontsize=11)
        ax4.set_ylabel('Average Close Price ($)', fontsize=11)
        ax4.set_title('Top 5 Sectors: Average Close Price Over Time', fontsize=12, fontweight='bold')
        ax4.legend(loc='best', fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        plt.savefig('analysis_4_industry_patterns.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved: analysis_4_industry_patterns.png")
        
        # Create detailed industry-depression analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Industry-Specific Depression Impact Analysis', fontsize=16, fontweight='bold')
        
        # Select a few key industries for detailed analysis
        key_industries = ['Semiconductors', 'Pharmaceuticals', 'Regional Banks', 
                         'Oil & Gas Exploration & Production']
        
        for idx, industry in enumerate(key_industries[:4]):
            ax = axes[idx // 2, idx % 2]
            
            industry_data = stock_with_depression[
                stock_with_depression['industry'] == industry
            ].dropna(subset=['depression_index', 'close'])
            
            if len(industry_data) > 0:
                # Group by date to get daily average
                daily_industry = industry_data.groupby('date').agg({
                    'close': 'mean',
                    'depression_index': 'first'
                }).reset_index()
                
                ax_twin = ax.twinx()
                ax.plot(daily_industry['date'], daily_industry['close'], 
                       color='blue', label='Avg Close Price', linewidth=2)
                ax_twin.plot(daily_industry['date'], daily_industry['depression_index'], 
                           color='red', label='Depression Index', linewidth=2, alpha=0.7)
                
                ax.set_ylabel('Average Close Price ($)', color='blue', fontsize=10)
                ax_twin.set_ylabel('Depression Index', color='red', fontsize=10)
                ax.set_title(f'{industry}', fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper left', fontsize=8)
                ax_twin.legend(loc='upper right', fontsize=8)
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('analysis_4_industry_detail.png', dpi=300, bbox_inches='tight')
        print("✓ Visualization saved: analysis_4_industry_detail.png")
        
        return industry_stats, industry_depression_corr
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "="*80)
        print("GENERATING SUMMARY REPORT")
        print("="*80)
        
        report = []
        report.append("="*80)
        report.append("TIME SERIES ANALYSIS SUMMARY REPORT")
        report.append("Depression Data, Stock Market, and Environmental Factors")
        report.append("="*80)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\nData Range: {self.merged_data['date'].min()} to {self.merged_data['date'].max()}")
        report.append(f"Total Records: {len(self.merged_data)}")
        
        report.append("\n" + "-"*80)
        report.append("KEY FINDINGS:")
        report.append("-"*80)
        
        # Calculate key statistics
        df = self.merged_data.dropna(subset=['depression_index', 'Close_^GSPC'])
        
        corr_sp500_depression = df[['Close_^GSPC', 'depression_index']].corr().iloc[0, 1]
        corr_volume_depression = df[['volume', 'depression_word_count']].dropna().corr().iloc[0, 1]
        
        report.append(f"\n1. S&P 500 vs Depression Index Correlation: {corr_sp500_depression:.4f}")
        report.append(f"2. Trading Volume vs Depression Word Count Correlation: {corr_volume_depression:.4f}")
        
        # Average metrics by depression level
        report.append("\n" + "-"*80)
        report.append("STOCK PERFORMANCE BY DEPRESSION LEVEL:")
        report.append("-"*80)
        
        perf_by_depression = df.groupby('depression_index_category').agg({
            'Return': 'mean',
            'Volatility_7': 'mean',
            'close': 'mean'
        }).round(6)
        
        report.append(perf_by_depression.to_string())
        
        # Industry insights
        report.append("\n" + "-"*80)
        report.append("TOP PERFORMING INDUSTRIES:")
        report.append("-"*80)
        
        industry_perf = self.stock_data.copy()
        industry_perf['price_change_pct'] = (
            (industry_perf['close'] - industry_perf['open']) / industry_perf['open']
        ) * 100
        
        top_industries = industry_perf.groupby('industry')['price_change_pct'].mean().nlargest(10)
        report.append(top_industries.to_string())
        
        # Save report
        report_text = '\n'.join(report)
        with open('time_series_analysis_report.txt', 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print("\n✓ Report saved: time_series_analysis_report.txt")
        
        # Save merged dataset
        self.merged_data.to_csv('merged_time_series_data.csv', index=False)
        print("✓ Merged dataset saved: merged_time_series_data.csv")
    
    def run_complete_analysis(self):
        """Run all analyses"""
        print("\n" + "="*80)
        print("STARTING COMPREHENSIVE TIME SERIES ANALYSIS")
        print("="*80)
        
        # Load and prepare data
        self.load_data()
        self.prepare_merged_dataset()
        
        # Run all analyses
        corr1 = self.analyze_stock_depression_relationship()
        corr2 = self.analyze_sp500_depression()
        corr3 = self.analyze_rainfall_impact()
        industry_stats, industry_corr = self.analyze_industry_patterns()
        
        # Generate summary
        self.generate_summary_report()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print("\nGenerated Files:")
        print("  - analysis_1_stock_vs_depression.png")
        print("  - analysis_1_time_series.png")
        print("  - analysis_2_sp500_depression.png")
        print("  - analysis_3_rainfall_impact.png")
        print("  - analysis_4_industry_patterns.png")
        print("  - analysis_4_industry_detail.png")
        print("  - time_series_analysis_report.txt")
        print("  - merged_time_series_data.csv")
        print("="*80)


if __name__ == "__main__":
    # Initialize and run analyzer
    analyzer = DepressionStockAnalyzer()
    analyzer.run_complete_analysis()
