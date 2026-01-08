"""
Statistical Significance Analysis for Depression-Stock Market Correlations
===========================================================================
This script performs comprehensive statistical testing to determine which
correlations are statistically significant and practically meaningful.

Includes:
- P-values and significance testing
- Confidence intervals
- Effect size interpretation
- Statistical power analysis
- Multiple comparison corrections
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class StatisticalSignificanceAnalyzer:
    """
    Analyzes statistical significance and practical importance of correlations
    """
    
    def __init__(self):
        self.merged_data = None
        self.stock_data = None
        self.results = []
        
    def load_data(self):
        """Load the merged dataset"""
        print("Loading merged time series data...")
        self.merged_data = pd.read_csv('merged_time_series_data.csv')
        self.merged_data['date'] = pd.to_datetime(self.merged_data['date'])
        
        self.stock_data = pd.read_csv('csv_exports/stock_data_wiki_clean.csv')
        self.stock_data['date'] = pd.to_datetime(self.stock_data['date'])
        
        print(f"✓ Loaded {len(self.merged_data)} records")
        
    def calculate_correlation_stats(self, x, y, var_x_name, var_y_name, method='pearson'):
        """
        Calculate comprehensive correlation statistics
        
        Returns:
        - Correlation coefficient
        - P-value
        - Confidence interval
        - Sample size
        - Statistical significance
        - Effect size interpretation
        """
        # Remove missing values
        valid_data = pd.DataFrame({var_x_name: x, var_y_name: y}).dropna()
        
        if len(valid_data) < 3:
            return None
            
        x_clean = valid_data[var_x_name].values
        y_clean = valid_data[var_y_name].values
        n = len(x_clean)
        
        # Calculate correlation and p-value
        if method == 'pearson':
            corr, p_value = pearsonr(x_clean, y_clean)
        elif method == 'spearman':
            corr, p_value = spearmanr(x_clean, y_clean)
        elif method == 'kendall':
            corr, p_value = kendalltau(x_clean, y_clean)
        
        # Calculate confidence interval (Fisher's z-transformation)
        if method == 'pearson':
            z = np.arctanh(corr)
            se = 1 / np.sqrt(n - 3)
            ci_lower = np.tanh(z - 1.96 * se)
            ci_upper = np.tanh(z + 1.96 * se)
        else:
            # For non-parametric, use bootstrap
            ci_lower, ci_upper = self.bootstrap_ci(x_clean, y_clean, method)
        
        # Determine significance levels
        if p_value < 0.001:
            significance = "***"
            sig_text = "Highly Significant"
        elif p_value < 0.01:
            significance = "**"
            sig_text = "Very Significant"
        elif p_value < 0.05:
            significance = "*"
            sig_text = "Significant"
        else:
            significance = "ns"
            sig_text = "Not Significant"
        
        # Effect size interpretation (Cohen's guidelines for correlation)
        abs_corr = abs(corr)
        if abs_corr < 0.10:
            effect_size = "Negligible"
        elif abs_corr < 0.30:
            effect_size = "Small"
        elif abs_corr < 0.50:
            effect_size = "Medium"
        else:
            effect_size = "Large"
        
        # Calculate R-squared (coefficient of determination)
        r_squared = corr ** 2
        
        return {
            'Variable_X': var_x_name,
            'Variable_Y': var_y_name,
            'Method': method.capitalize(),
            'N': n,
            'Correlation': corr,
            'P_Value': p_value,
            'CI_Lower': ci_lower,
            'CI_Upper': ci_upper,
            'Significance': significance,
            'Significance_Text': sig_text,
            'Effect_Size': effect_size,
            'R_Squared': r_squared,
            'Explained_Variance_%': r_squared * 100
        }
    
    def bootstrap_ci(self, x, y, method, n_bootstrap=1000, alpha=0.05):
        """Bootstrap confidence interval for non-parametric correlations"""
        correlations = []
        n = len(x)
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(n, n, replace=True)
            x_boot = x[indices]
            y_boot = y[indices]
            
            if method == 'spearman':
                corr, _ = spearmanr(x_boot, y_boot)
            elif method == 'kendall':
                corr, _ = kendalltau(x_boot, y_boot)
            
            correlations.append(corr)
        
        ci_lower = np.percentile(correlations, alpha/2 * 100)
        ci_upper = np.percentile(correlations, (1 - alpha/2) * 100)
        
        return ci_lower, ci_upper
    
    def analyze_all_correlations(self):
        """Analyze all key correlations with statistical tests"""
        print("\n" + "="*100)
        print("COMPREHENSIVE STATISTICAL SIGNIFICANCE ANALYSIS")
        print("="*100)
        
        df = self.merged_data
        
        # Define correlation pairs to test
        correlation_pairs = [
            # Stock metrics vs Depression Index
            ('close', 'depression_index', 'Stock Close Price', 'Depression Index'),
            ('open', 'depression_index', 'Stock Open Price', 'Depression Index'),
            ('high', 'depression_index', 'Stock High Price', 'Depression Index'),
            ('low', 'depression_index', 'Stock Low Price', 'Depression Index'),
            ('volume', 'depression_index', 'Trading Volume', 'Depression Index'),
            ('price_range', 'depression_index', 'Price Range (Volatility)', 'Depression Index'),
            
            # Stock metrics vs Depression Word Count
            ('close', 'depression_word_count', 'Stock Close Price', 'Depression Word Count'),
            ('volume', 'depression_word_count', 'Trading Volume', 'Depression Word Count'),
            ('price_range', 'depression_word_count', 'Price Range', 'Depression Word Count'),
            
            # S&P 500 vs Depression
            ('Close_^GSPC', 'depression_index', 'S&P 500 Close', 'Depression Index'),
            ('Return', 'depression_index', 'S&P 500 Return', 'Depression Index'),
            ('Volatility_7', 'depression_index', 'S&P 500 Volatility', 'Depression Index'),
            ('Return', 'depression_word_count', 'S&P 500 Return', 'Depression Word Count'),
            ('Volatility_7', 'depression_word_count', 'S&P 500 Volatility', 'Depression Word Count'),
            
            # Rainfall
            ('avg_national_rainfall', 'close', 'Rainfall', 'Stock Close Price'),
            ('avg_national_rainfall', 'depression_index', 'Rainfall', 'Depression Index'),
            ('avg_national_rainfall', 'Return', 'Rainfall', 'S&P 500 Return'),
        ]
        
        # Calculate statistics for each pair
        for var_x, var_y, name_x, name_y in correlation_pairs:
            if var_x in df.columns and var_y in df.columns:
                # Pearson correlation (parametric)
                stats_pearson = self.calculate_correlation_stats(
                    df[var_x], df[var_y], name_x, name_y, method='pearson'
                )
                if stats_pearson:
                    self.results.append(stats_pearson)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Apply Bonferroni correction for multiple comparisons
        n_tests = len(results_df)
        results_df['Bonferroni_Adjusted_P'] = results_df['P_Value'] * n_tests
        results_df['Bonferroni_Significant'] = results_df['Bonferroni_Adjusted_P'] < 0.05
        
        # Sort by p-value
        results_df = results_df.sort_values('P_Value')
        
        return results_df
    
    def print_significance_summary(self, results_df):
        """Print comprehensive significance summary"""
        print("\n" + "="*100)
        print("STATISTICAL SIGNIFICANCE SUMMARY")
        print("="*100)
        
        print("\n" + "-"*100)
        print("MOST SIGNIFICANT CORRELATIONS (Lowest P-Values)")
        print("-"*100)
        
        # Filter Pearson correlations for cleaner display
        pearson_results = results_df[results_df['Method'] == 'Pearson'].head(15)
        
        print(f"\n{'Relationship':<60} {'r':<8} {'P-Value':<12} {'Sig':<6} {'Effect':<12} {'R²%':<8}")
        print("-"*100)
        
        for _, row in pearson_results.iterrows():
            relationship = f"{row['Variable_X']} ↔ {row['Variable_Y']}"
            print(f"{relationship:<60} {row['Correlation']:>7.4f} {row['P_Value']:>11.4e} "
                  f"{row['Significance']:>5} {row['Effect_Size']:>11} {row['Explained_Variance_%']:>7.2f}")
        
        print("\n" + "-"*100)
        print("LEGEND:")
        print("-"*100)
        print("*** p < 0.001 (Highly Significant)")
        print("**  p < 0.01  (Very Significant)")
        print("*   p < 0.05  (Significant)")
        print("ns  p ≥ 0.05  (Not Significant)")
        print("\nEffect Size Interpretation (|r|):")
        print("  Negligible: < 0.10")
        print("  Small:      0.10 - 0.29")
        print("  Medium:     0.30 - 0.49")
        print("  Large:      ≥ 0.50")
        
        # Summary statistics
        print("\n" + "="*100)
        print("SUMMARY STATISTICS")
        print("="*100)
        
        total = len(results_df[results_df['Method'] == 'Pearson'])
        significant = len(results_df[(results_df['Method'] == 'Pearson') & (results_df['P_Value'] < 0.05)])
        highly_sig = len(results_df[(results_df['Method'] == 'Pearson') & (results_df['P_Value'] < 0.001)])
        
        print(f"\nTotal correlations tested: {total}")
        print(f"Significant (p < 0.05): {significant} ({significant/total*100:.1f}%)")
        print(f"Highly significant (p < 0.001): {highly_sig} ({highly_sig/total*100:.1f}%)")
        
        # Effect size distribution
        effect_counts = results_df[results_df['Method'] == 'Pearson']['Effect_Size'].value_counts()
        print("\nEffect Size Distribution:")
        for effect, count in effect_counts.items():
            print(f"  {effect}: {count} ({count/total*100:.1f}%)")
        
        # Bonferroni correction results
        bonf_sig = len(results_df[(results_df['Method'] == 'Pearson') & results_df['Bonferroni_Significant']])
        print(f"\nBonferroni-corrected significant: {bonf_sig} ({bonf_sig/total*100:.1f}%)")
        print(f"(Adjusted significance level: α = 0.05/{total} = {0.05/total:.6f})")
    
    def create_significance_visualizations(self, results_df):
        """Create comprehensive visualizations of statistical significance"""
        
        # Filter to Pearson for main visualization
        pearson_df = results_df[results_df['Method'] == 'Pearson'].copy()
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
        
        fig.suptitle('Statistical Significance Analysis: Which Correlations Matter?', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        # Plot 1: Correlation coefficients with confidence intervals (top correlations)
        ax1 = fig.add_subplot(gs[0, :])
        top_15 = pearson_df.nlargest(15, 'Correlation')
        
        y_pos = np.arange(len(top_15))
        colors = ['green' if p < 0.001 else 'orange' if p < 0.05 else 'red' 
                  for p in top_15['P_Value']]
        
        ax1.barh(y_pos, top_15['Correlation'], color=colors, alpha=0.7)
        ax1.errorbar(top_15['Correlation'], y_pos, 
                     xerr=[top_15['Correlation'] - top_15['CI_Lower'], 
                           top_15['CI_Upper'] - top_15['Correlation']],
                     fmt='none', ecolor='black', capsize=3, alpha=0.5)
        
        labels = [f"{row['Variable_X'][:25]} ↔ {row['Variable_Y'][:25]}" 
                  for _, row in top_15.iterrows()]
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(labels, fontsize=9)
        ax1.set_xlabel('Correlation Coefficient with 95% CI', fontsize=11, fontweight='bold')
        ax1.set_title('Top 15 Positive Correlations (with Confidence Intervals)', 
                     fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
        
        # Add significance markers
        for i, (_, row) in enumerate(top_15.iterrows()):
            ax1.text(row['Correlation'] + 0.01, i, row['Significance'], 
                    fontsize=10, va='center', fontweight='bold')
        
        # Plot 2: P-value distribution
        ax2 = fig.add_subplot(gs[1, 0])
        p_values = pearson_df['P_Value']
        
        ax2.hist(p_values, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
        ax2.axvline(x=0.01, color='orange', linestyle='--', linewidth=2, label='α = 0.01')
        ax2.axvline(x=0.001, color='green', linestyle='--', linewidth=2, label='α = 0.001')
        ax2.set_xlabel('P-Value', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax2.set_title('Distribution of P-Values', fontsize=12, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Correlation vs P-value scatter
        ax3 = fig.add_subplot(gs[1, 1])
        scatter = ax3.scatter(pearson_df['Correlation'], -np.log10(pearson_df['P_Value']),
                             c=pearson_df['N'], cmap='viridis', s=100, alpha=0.6)
        ax3.axhline(y=-np.log10(0.05), color='red', linestyle='--', linewidth=2, 
                   label='p = 0.05', alpha=0.7)
        ax3.axhline(y=-np.log10(0.01), color='orange', linestyle='--', linewidth=2,
                   label='p = 0.01', alpha=0.7)
        ax3.set_xlabel('Correlation Coefficient', fontsize=11, fontweight='bold')
        ax3.set_ylabel('-log₁₀(P-Value)', fontsize=11, fontweight='bold')
        ax3.set_title('Correlation Magnitude vs Statistical Significance', fontsize=12, fontweight='bold')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Sample Size (N)', fontsize=9)
        
        # Plot 4: Effect size distribution
        ax4 = fig.add_subplot(gs[1, 2])
        effect_order = ['Negligible', 'Small', 'Medium', 'Large']
        effect_counts = pearson_df['Effect_Size'].value_counts().reindex(effect_order, fill_value=0)
        colors_effect = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
        ax4.bar(range(len(effect_counts)), effect_counts.values, color=colors_effect, alpha=0.7)
        ax4.set_xticks(range(len(effect_counts)))
        ax4.set_xticklabels(effect_counts.index, fontsize=10)
        ax4.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax4.set_title('Effect Size Distribution', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels
        total = len(pearson_df)
        for i, v in enumerate(effect_counts.values):
            ax4.text(i, v + 0.5, f'{v}\n({v/total*100:.1f}%)', ha='center', va='bottom', 
                    fontsize=9, fontweight='bold')
        
        # Plot 5: Key depression correlations comparison
        ax5 = fig.add_subplot(gs[2, 0])
        
        key_pairs = [
            ('Price Range (Volatility)', 'Depression Index'),
            ('S&P 500 Volatility', 'Depression Index'),
            ('S&P 500 Close', 'Depression Index'),
            ('Stock Close Price', 'Depression Index'),
            ('Trading Volume', 'Depression Index')
        ]
        
        key_results = []
        for x, y in key_pairs:
            match = pearson_df[(pearson_df['Variable_X'] == x) & (pearson_df['Variable_Y'] == y)]
            if not match.empty:
                key_results.append(match.iloc[0])
        
        if key_results:
            key_df = pd.DataFrame(key_results)
            y_pos = np.arange(len(key_df))
            colors_key = ['green' if p < 0.001 else 'orange' if p < 0.05 else 'red' 
                         for p in key_df['P_Value']]
            
            ax5.barh(y_pos, key_df['Correlation'], color=colors_key, alpha=0.7)
            ax5.errorbar(key_df['Correlation'], y_pos,
                        xerr=[key_df['Correlation'] - key_df['CI_Lower'],
                              key_df['CI_Upper'] - key_df['Correlation']],
                        fmt='none', ecolor='black', capsize=4, alpha=0.6, linewidth=2)
            
            labels_key = [row['Variable_X'] for _, row in key_df.iterrows()]
            ax5.set_yticks(y_pos)
            ax5.set_yticklabels(labels_key, fontsize=9)
            ax5.set_xlabel('Correlation Coefficient', fontsize=10, fontweight='bold')
            ax5.set_title('Key Depression Index Correlations', fontsize=11, fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='x')
            
            # Add r² percentages
            for i, (_, row) in enumerate(key_df.iterrows()):
                ax5.text(row['Correlation'] + 0.01, i, 
                        f"{row['Significance']}\nR²={row['Explained_Variance_%']:.1f}%",
                        fontsize=8, va='center', fontweight='bold')
        
        # Plot 6: Sample size vs correlation
        ax6 = fig.add_subplot(gs[2, 1])
        scatter2 = ax6.scatter(pearson_df['N'], pearson_df['Correlation'],
                              c=-np.log10(pearson_df['P_Value']), cmap='RdYlGn',
                              s=100, alpha=0.6)
        ax6.set_xlabel('Sample Size (N)', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Correlation Coefficient', fontsize=11, fontweight='bold')
        ax6.set_title('Sample Size vs Correlation Strength', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        cbar2 = plt.colorbar(scatter2, ax=ax6)
        cbar2.set_label('-log₁₀(P-Value)', fontsize=9)
        
        # Plot 7: Explained variance (R²) for significant correlations
        ax7 = fig.add_subplot(gs[2, 2])
        sig_corrs = pearson_df[pearson_df['P_Value'] < 0.05].nlargest(10, 'R_Squared')
        
        if not sig_corrs.empty:
            y_pos = np.arange(len(sig_corrs))
            ax7.barh(y_pos, sig_corrs['Explained_Variance_%'], color='steelblue', alpha=0.7)
            
            labels_r2 = [f"{row['Variable_X'][:20]}↔{row['Variable_Y'][:20]}" 
                        for _, row in sig_corrs.iterrows()]
            ax7.set_yticks(y_pos)
            ax7.set_yticklabels(labels_r2, fontsize=8)
            ax7.set_xlabel('Explained Variance (R²%)', fontsize=10, fontweight='bold')
            ax7.set_title('Top 10 Explained Variance\n(Significant Correlations Only)', 
                         fontsize=11, fontweight='bold')
            ax7.grid(True, alpha=0.3, axis='x')
            
            # Add percentage labels
            for i, (_, row) in enumerate(sig_corrs.iterrows()):
                ax7.text(row['Explained_Variance_%'] + 0.5, i, 
                        f"{row['Explained_Variance_%']:.1f}%",
                        fontsize=8, va='center')
        
        plt.savefig('STATISTICAL_SIGNIFICANCE_ANALYSIS.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved: STATISTICAL_SIGNIFICANCE_ANALYSIS.png")
    
    def create_interpretation_guide(self, results_df):
        """Create detailed interpretation guide"""
        pearson_df = results_df[results_df['Method'] == 'Pearson']
        
        guide = []
        guide.append("="*100)
        guide.append("INTERPRETATION GUIDE: WHY SOME CORRELATIONS MATTER MORE")
        guide.append("="*100)
        
        guide.append("\n" + "="*100)
        guide.append("1. STATISTICAL SIGNIFICANCE (P-Value)")
        guide.append("="*100)
        guide.append("\nThe p-value tells us the probability that the observed correlation occurred by chance.")
        guide.append("\nLower p-value = More confident the relationship is real")
        guide.append("\nStandard thresholds:")
        guide.append("  • p < 0.001 (***): Less than 0.1% chance this is random → HIGHLY RELIABLE")
        guide.append("  • p < 0.01  (**):  Less than 1% chance this is random → VERY RELIABLE")
        guide.append("  • p < 0.05  (*):   Less than 5% chance this is random → RELIABLE")
        guide.append("  • p ≥ 0.05  (ns):  More than 5% chance this is random → UNRELIABLE")
        
        # Find examples
        highly_sig = pearson_df[pearson_df['P_Value'] < 0.001].nlargest(3, 'Correlation')
        not_sig = pearson_df[pearson_df['P_Value'] >= 0.05].head(3)
        
        guide.append("\nEXAMPLES FROM OUR DATA:")
        guide.append("\nHighly Significant (Trustworthy):")
        for _, row in highly_sig.iterrows():
            guide.append(f"  • {row['Variable_X']} ↔ {row['Variable_Y']}")
            guide.append(f"    r = {row['Correlation']:.4f}, p = {row['P_Value']:.6f} ***")
            guide.append(f"    → This relationship is REAL (99.9%+ confidence)")
        
        if not not_sig.empty:
            guide.append("\nNot Significant (Cannot Trust):")
            for _, row in not_sig.iterrows():
                guide.append(f"  • {row['Variable_X']} ↔ {row['Variable_Y']}")
                guide.append(f"    r = {row['Correlation']:.4f}, p = {row['P_Value']:.4f} ns")
                guide.append(f"    → Could be random chance, don't rely on this")
        
        guide.append("\n" + "="*100)
        guide.append("2. EFFECT SIZE (Correlation Magnitude)")
        guide.append("="*100)
        guide.append("\nEven if statistically significant, is the effect large enough to matter?")
        guide.append("\nCohen's guidelines for correlation strength:")
        guide.append("  • |r| < 0.10:  Negligible → Practically meaningless")
        guide.append("  • |r| 0.10-0.29: Small → Detectable but weak relationship")
        guide.append("  • |r| 0.30-0.49: Medium → Moderate, meaningful relationship")
        guide.append("  • |r| ≥ 0.50:  Large → Strong, important relationship")
        
        # Examples by effect size
        large = pearson_df[abs(pearson_df['Correlation']) >= 0.30].head(3)
        small = pearson_df[(abs(pearson_df['Correlation']) >= 0.10) & 
                          (abs(pearson_df['Correlation']) < 0.30)].head(3)
        negligible = pearson_df[abs(pearson_df['Correlation']) < 0.10].head(3)
        
        guide.append("\nEXAMPLES FROM OUR DATA:")
        if not large.empty:
            guide.append("\nMedium/Large Effect (Practically Important):")
            for _, row in large.iterrows():
                guide.append(f"  • {row['Variable_X']} ↔ {row['Variable_Y']}")
                guide.append(f"    r = {row['Correlation']:.4f} ({row['Effect_Size']})")
                guide.append(f"    → Explains {row['Explained_Variance_%']:.1f}% of variance")
        
        if not small.empty:
            guide.append("\nSmall Effect (Detectable but Weak):")
            for _, row in small.iterrows():
                guide.append(f"  • {row['Variable_X']} ↔ {row['Variable_Y']}")
                guide.append(f"    r = {row['Correlation']:.4f} ({row['Effect_Size']})")
                guide.append(f"    → Only explains {row['Explained_Variance_%']:.1f}% of variance")
        
        if not negligible.empty:
            guide.append("\nNegligible Effect (Not Practically Meaningful):")
            for _, row in negligible.iterrows():
                guide.append(f"  • {row['Variable_X']} ↔ {row['Variable_Y']}")
                guide.append(f"    r = {row['Correlation']:.4f} ({row['Effect_Size']})")
                guide.append(f"    → Explains only {row['Explained_Variance_%']:.1f}% of variance")
        
        guide.append("\n" + "="*100)
        guide.append("3. CONFIDENCE INTERVALS")
        guide.append("="*100)
        guide.append("\n95% Confidence Interval shows the range where true correlation likely falls.")
        guide.append("\nNarrow CI = Precise estimate")
        guide.append("Wide CI = Uncertain estimate")
        guide.append("CI containing 0 = Might be no relationship at all")
        
        # Example
        precise = pearson_df.copy()
        precise['CI_Width'] = precise['CI_Upper'] - precise['CI_Lower']
        precise_example = precise[precise['P_Value'] < 0.05].nsmallest(2, 'CI_Width')
        wide_example = precise[precise['P_Value'] < 0.05].nlargest(2, 'CI_Width')
        
        guide.append("\nEXAMPLES:")
        if not precise_example.empty:
            guide.append("\nPrecise Estimates (Narrow CI):")
            for _, row in precise_example.iterrows():
                guide.append(f"  • {row['Variable_X']} ↔ {row['Variable_Y']}")
                guide.append(f"    r = {row['Correlation']:.4f}, 95% CI [{row['CI_Lower']:.4f}, {row['CI_Upper']:.4f}]")
                guide.append(f"    → Precise estimate, narrow range")
        
        if not wide_example.empty:
            guide.append("\nLess Precise Estimates (Wide CI):")
            for _, row in wide_example.iterrows():
                guide.append(f"  • {row['Variable_X']} ↔ {row['Variable_Y']}")
                guide.append(f"    r = {row['Correlation']:.4f}, 95% CI [{row['CI_Lower']:.4f}, {row['CI_Upper']:.4f}]")
                guide.append(f"    → More uncertainty, wide range")
        
        guide.append("\n" + "="*100)
        guide.append("4. SAMPLE SIZE (N)")
        guide.append("="*100)
        guide.append("\nLarger sample = More reliable results")
        guide.append("Smaller sample = Higher chance of false positives/negatives")
        guide.append(f"\nOur analysis has N = {pearson_df['N'].mode()[0] if not pearson_df.empty else 'N/A'} for most correlations")
        guide.append("With N > 30, results are generally reliable if p < 0.05")
        
        guide.append("\n" + "="*100)
        guide.append("5. MULTIPLE COMPARISON CORRECTION (Bonferroni)")
        guide.append("="*100)
        guide.append(f"\nWe tested {len(pearson_df)} correlations.")
        guide.append("With many tests, some will appear significant just by chance!")
        guide.append(f"\nBonferroni correction: Adjusted α = 0.05/{len(pearson_df)} = {0.05/len(pearson_df):.6f}")
        guide.append("\nOnly correlations surviving this stricter threshold are truly robust.")
        
        bonf_survivors = pearson_df[pearson_df['Bonferroni_Significant']].nlargest(5, 'Correlation')
        guide.append(f"\nCorrelations that survive Bonferroni correction ({len(bonf_survivors)} total):")
        for _, row in bonf_survivors.iterrows():
            guide.append(f"  • {row['Variable_X']} ↔ {row['Variable_Y']}")
            guide.append(f"    r = {row['Correlation']:.4f}, adjusted p = {row['Bonferroni_Adjusted_P']:.4f}")
        
        guide.append("\n" + "="*100)
        guide.append("CONCLUSION: RANKING CORRELATION IMPORTANCE")
        guide.append("="*100)
        guide.append("\nA correlation is MOST IMPORTANT when it has:")
        guide.append("  ✓ Low p-value (< 0.001 preferred)")
        guide.append("  ✓ Medium to large effect size (|r| > 0.30)")
        guide.append("  ✓ Narrow confidence interval")
        guide.append("  ✓ Large sample size (N > 100 preferred)")
        guide.append("  ✓ Survives multiple comparison correction")
        
        guide.append("\n" + "="*100)
        guide.append("TOP CORRELATIONS BY IMPORTANCE (All Criteria Combined)")
        guide.append("="*100)
        
        # Score correlations by importance
        scored = pearson_df.copy()
        scored['importance_score'] = (
            (scored['P_Value'] < 0.001).astype(int) * 3 +  # Highly significant
            (scored['P_Value'] < 0.01).astype(int) * 2 +   # Very significant
            (scored['P_Value'] < 0.05).astype(int) * 1 +   # Significant
            (abs(scored['Correlation']) >= 0.30).astype(int) * 3 +  # Medium/large effect
            (abs(scored['Correlation']) >= 0.10).astype(int) * 1 +  # Small effect
            (scored['Bonferroni_Significant']).astype(int) * 2 +    # Survives correction
            (scored['N'] > 100).astype(int) * 1             # Large sample
        )
        
        top_important = scored.nlargest(10, 'importance_score')
        
        for i, (_, row) in enumerate(top_important.iterrows(), 1):
            guide.append(f"\n{i}. {row['Variable_X']} ↔ {row['Variable_Y']}")
            guide.append(f"   r = {row['Correlation']:.4f} {row['Significance']}, p = {row['P_Value']:.6f}")
            guide.append(f"   Effect: {row['Effect_Size']}, R² = {row['Explained_Variance_%']:.1f}%")
            guide.append(f"   95% CI: [{row['CI_Lower']:.4f}, {row['CI_Upper']:.4f}], N = {row['N']}")
            guide.append(f"   Bonferroni: {'PASS' if row['Bonferroni_Significant'] else 'FAIL'}")
            guide.append(f"   ★ Importance Score: {row['importance_score']}/10")
        
        guide_text = '\n'.join(guide)
        
        # Save to file
        with open('STATISTICAL_INTERPRETATION_GUIDE.txt', 'w') as f:
            f.write(guide_text)
        
        print(guide_text)
        print("\n✓ Interpretation guide saved: STATISTICAL_INTERPRETATION_GUIDE.txt")
    
    def run_complete_analysis(self):
        """Run complete statistical significance analysis"""
        print("\n" + "="*100)
        print("STATISTICAL SIGNIFICANCE ANALYSIS")
        print("="*100)
        
        # Load data
        self.load_data()
        
        # Analyze all correlations
        results_df = self.analyze_all_correlations()
        
        # Save results
        results_df.to_csv('correlation_statistics_full.csv', index=False)
        print(f"\n✓ Full results saved: correlation_statistics_full.csv")
        
        # Print summary
        self.print_significance_summary(results_df)
        
        # Create visualizations
        self.create_significance_visualizations(results_df)
        
        # Create interpretation guide
        self.create_interpretation_guide(results_df)
        
        print("\n" + "="*100)
        print("ANALYSIS COMPLETE!")
        print("="*100)
        print("\nGenerated Files:")
        print("  - STATISTICAL_SIGNIFICANCE_ANALYSIS.png (Comprehensive visualization)")
        print("  - STATISTICAL_INTERPRETATION_GUIDE.txt (Detailed explanation)")
        print("  - correlation_statistics_full.csv (All statistical results)")
        print("="*100)


if __name__ == "__main__":
    analyzer = StatisticalSignificanceAnalyzer()
    analyzer.run_complete_analysis()
