"""
Executive Summary Visualization
Creates a single comprehensive dashboard showing key findings
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load merged data
df = pd.read_csv('merged_time_series_data.csv')
df['date'] = pd.to_datetime(df['date'])

# Create executive summary dashboard
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

# Main title
fig.suptitle('EXECUTIVE SUMMARY: Depression Indicators & Stock Market Analysis (2017-2018)', 
             fontsize=18, fontweight='bold', y=0.98)

# 1. Main time series (large plot)
ax1 = fig.add_subplot(gs[0, :])
ax1_twin = ax1.twinx()

valid = df.dropna(subset=['Close_^GSPC', 'depression_index'])
ax1.plot(valid['date'], valid['Close_^GSPC'], color='#2E86AB', label='S&P 500', linewidth=2.5)
ax1_twin.plot(valid['date'], valid['depression_index'], color='#A23B72', 
             label='Depression Index', linewidth=2.5, alpha=0.8)

ax1.set_ylabel('S&P 500 Close Price ($)', color='#2E86AB', fontsize=12, fontweight='bold')
ax1_twin.set_ylabel('Depression Index (0-100)', color='#A23B72', fontsize=12, fontweight='bold')
ax1.set_title('Market Performance vs Depression Index Over Time', fontsize=14, fontweight='bold', pad=15)
ax1.grid(True, alpha=0.2)
ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)
ax1_twin.legend(loc='upper right', fontsize=11, framealpha=0.9)
ax1.tick_params(axis='x', rotation=30)

# 2. Correlation matrix
ax2 = fig.add_subplot(gs[1, 0])
corr_vars = ['close', 'volume', 'Return', 'Volatility_7', 'depression_index', 'depression_word_count']
corr_data = df[corr_vars].dropna()
corr_matrix = corr_data.corr()

sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlGn', center=0, 
           square=True, ax=ax2, cbar_kws={'shrink': 0.8})
ax2.set_title('Correlation Matrix', fontsize=12, fontweight='bold', pad=10)
ax2.set_xticklabels(['Close', 'Volume', 'Return', 'Volatility', 'Dep.Index', 'Dep.Words'], 
                    rotation=45, ha='right', fontsize=9)
ax2.set_yticklabels(['Close', 'Volume', 'Return', 'Volatility', 'Dep.Index', 'Dep.Words'], 
                    rotation=0, fontsize=9)

# 3. Depression categories vs returns
ax3 = fig.add_subplot(gs[1, 1])
valid = df.dropna(subset=['depression_index_category', 'Return'])
cat_stats = valid.groupby('depression_index_category').agg({
    'Return': ['mean', 'std']
}).reset_index()
cat_stats.columns = ['Category', 'Mean', 'Std']

colors = ['#06A77D', '#F5B700', '#D4691C', '#C73E1D']
bars = ax3.bar(range(len(cat_stats)), cat_stats['Mean']*100, 
              yerr=cat_stats['Std']*100, color=colors, alpha=0.8, capsize=5)
ax3.set_xticks(range(len(cat_stats)))
ax3.set_xticklabels(cat_stats['Category'], fontsize=10)
ax3.set_ylabel('Average Daily Return (%)', fontsize=11, fontweight='bold')
ax3.set_title('Returns by Depression Level', fontsize=12, fontweight='bold', pad=10)
ax3.grid(True, alpha=0.2, axis='y')
ax3.axhline(y=0, color='black', linestyle='--', linewidth=1.5)

# 4. Volatility relationship
ax4 = fig.add_subplot(gs[1, 2])
valid = df.dropna(subset=['depression_index', 'Volatility_7'])
scatter = ax4.scatter(valid['depression_index'], valid['Volatility_7']*100, 
                     c=valid['Return']*100, cmap='RdYlGn', alpha=0.5, s=40)
ax4.set_xlabel('Depression Index', fontsize=11, fontweight='bold')
ax4.set_ylabel('7-Day Volatility (%)', fontsize=11, fontweight='bold')
ax4.set_title('Market Volatility vs Depression', fontsize=12, fontweight='bold', pad=10)
ax4.grid(True, alpha=0.2)

# Add trend line
z = np.polyfit(valid['depression_index'], valid['Volatility_7']*100, 1)
p = np.poly1d(z)
ax4.plot(valid['depression_index'].sort_values(), 
        p(valid['depression_index'].sort_values()), 
        "r--", alpha=0.8, linewidth=2.5, label=f'Trend (r={valid[["depression_index", "Volatility_7"]].corr().iloc[0,1]:.3f})')
ax4.legend(loc='best', fontsize=9)

cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_label('Daily Return (%)', fontsize=9)

# 5. Top/Bottom industries
ax5 = fig.add_subplot(gs[2, :2])
stock_data = pd.read_csv('csv_exports/stock_data_wiki_clean.csv')
stock_data['price_change_pct'] = ((stock_data['close'] - stock_data['open']) / stock_data['open']) * 100
industry_perf = stock_data.groupby('industry')['price_change_pct'].mean().sort_values()

top_5 = industry_perf.tail(5)
bottom_5 = industry_perf.head(5)
combined = pd.concat([bottom_5, top_5])

colors_ind = ['#C73E1D' if x < 0 else '#06A77D' for x in combined.values]
ax5.barh(range(len(combined)), combined.values, color=colors_ind, alpha=0.8)
ax5.set_yticks(range(len(combined)))
ax5.set_yticklabels(combined.index, fontsize=9)
ax5.set_xlabel('Average Price Change (%)', fontsize=11, fontweight='bold')
ax5.set_title('Best and Worst Performing Industries', fontsize=12, fontweight='bold', pad=10)
ax5.grid(True, alpha=0.2, axis='x')
ax5.axvline(x=0, color='black', linestyle='-', linewidth=1.5)

# 6. Key statistics box
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')

stats_text = f"""
KEY STATISTICS
{'='*35}

Data Period: {df['date'].min().strftime('%Y-%m-%d')} 
             to {df['date'].max().strftime('%Y-%m-%d')}

Total Trading Days: {len(df)}

S&P 500:
  • Avg Return: {df['Return'].mean()*100:.3f}%
  • Avg Volatility: {df['Volatility_7'].mean()*100:.3f}%

Depression Index:
  • Range: {df['depression_index'].min():.0f} - {df['depression_index'].max():.0f}
  • Average: {df['depression_index'].mean():.1f}

Correlations:
  • S&P vs Depression: {df[['Close_^GSPC', 'depression_index']].dropna().corr().iloc[0,1]:.4f}
  • Volatility vs Depression: {df[['Volatility_7', 'depression_index']].dropna().corr().iloc[0,1]:.4f}
  • Returns vs Dep.Words: {df[['Return', 'depression_word_count']].dropna().corr().iloc[0,1]:.4f}

Industries Analyzed: {stock_data['industry'].nunique()}
Companies Tracked: {stock_data['ticker'].nunique()}
"""

ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, 
        fontsize=10, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.savefig('EXECUTIVE_SUMMARY_DASHBOARD.png', dpi=300, bbox_inches='tight')
print("✓ Executive summary dashboard saved: EXECUTIVE_SUMMARY_DASHBOARD.png")
print("\nAnalysis complete! All visualizations and reports generated successfully.")
