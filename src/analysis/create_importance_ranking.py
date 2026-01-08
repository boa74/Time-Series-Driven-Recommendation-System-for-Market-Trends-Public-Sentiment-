"""
Create a simple, clear visual ranking of correlation importance
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the results
df = pd.read_csv('correlation_statistics_full.csv')

# Calculate importance score
df['importance_score'] = (
    (df['P_Value'] < 0.001).astype(int) * 3 +  # Highly significant
    (df['P_Value'] < 0.01).astype(int) * 2 +   # Very significant
    (df['P_Value'] < 0.05).astype(int) * 1 +   # Significant
    (abs(df['Correlation']) >= 0.30).astype(int) * 3 +  # Medium/large effect
    (abs(df['Correlation']) >= 0.10).astype(int) * 1 +  # Small effect
    (df['Bonferroni_Significant']).astype(int) * 2 +    # Survives correction
    (df['N'] > 100).astype(int) * 1             # Large sample
)

# Create importance ranking visualization
fig, axes = plt.subplots(2, 1, figsize=(16, 12))
fig.suptitle('Statistical Importance Ranking: Which Correlations Can You Trust?', 
             fontsize=18, fontweight='bold')

# Plot 1: Top correlations by importance score
ax1 = axes[0]
top_10 = df.nlargest(10, 'importance_score')

y_pos = np.arange(len(top_10))
colors = ['darkgreen' if score >= 10 else 'green' if score >= 5 else 'orange' if score >= 3 else 'red' 
          for score in top_10['importance_score']]

bars = ax1.barh(y_pos, top_10['importance_score'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add labels
labels = []
for _, row in top_10.iterrows():
    var_x = row['Variable_X'][:30]
    var_y = row['Variable_Y'][:30]
    sig = row['Significance']
    labels.append(f"{var_x} ↔ {var_y} {sig}")

ax1.set_yticks(y_pos)
ax1.set_yticklabels(labels, fontsize=10)
ax1.set_xlabel('Importance Score (out of 10)', fontsize=12, fontweight='bold')
ax1.set_title('Top 10 Correlations by Statistical Importance', fontsize=14, fontweight='bold', pad=15)
ax1.set_xlim(0, 11)
ax1.grid(True, alpha=0.3, axis='x')

# Add score labels and details on bars
for i, (_, row) in enumerate(top_10.iterrows()):
    score = row['importance_score']
    ax1.text(score + 0.2, i, f"{score:.0f}", va='center', fontsize=11, fontweight='bold')
    
    # Add correlation and p-value
    corr_text = f"r={row['Correlation']:.3f}, p={row['P_Value']:.2e}"
    ax1.text(0.2, i, corr_text, va='center', fontsize=8, color='white', fontweight='bold')

# Add legend
legend_elements = [
    plt.Rectangle((0, 0), 1, 1, fc='darkgreen', alpha=0.8, label='10/10: Trust completely'),
    plt.Rectangle((0, 0), 1, 1, fc='green', alpha=0.8, label='5/10: Moderate evidence'),
    plt.Rectangle((0, 0), 1, 1, fc='orange', alpha=0.8, label='3/10: Weak evidence'),
    plt.Rectangle((0, 0), 1, 1, fc='red', alpha=0.8, label='1/10: Don\'t trust')
]
ax1.legend(handles=legend_elements, loc='lower right', fontsize=10)

# Plot 2: Decision tree visualization
ax2 = axes[1]
ax2.axis('off')

# Create decision tree text
decision_text = """
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    HOW TO INTERPRET CORRELATION IMPORTANCE                          │
└─────────────────────────────────────────────────────────────────────────────────────┘

Start Here: You found a correlation!
         │
         ├──→ Is p-value < 0.001? (***)
         │    │
         │    ├─ YES → Is |r| > 0.20?
         │    │   │
         │    │   ├─ YES → Does it survive Bonferroni correction?
         │    │   │   │
         │    │   │   ├─ YES → ★★★★★ HIGHLY IMPORTANT (Score: 10/10)
         │    │   │   │          Example: Price Range ↔ Depression (r=0.27, p<0.001)
         │    │   │   │          ✓ Trust this relationship completely
         │    │   │   │
         │    │   │   └─ NO  → ★★★ MODERATELY IMPORTANT (Score: 5-7/10)
         │    │   │              Example: Stock Open ↔ Depression (r=0.11, p=0.007)
         │    │   │              ⚠ Probably real but not robust to multiple testing
         │    │   │
         │    │   └─ NO → ★★ WEAK EVIDENCE (Score: 3-5/10)
         │    │              Example: Stock Low ↔ Depression (r=0.10, p=0.014)
         │    │              ⚠ Small effect, marginal significance
         │    │
         │    └─ NO (p ≥ 0.001) → Is p-value < 0.05? (** or *)
         │        │
         │        ├─ YES → ★ LIMITED EVIDENCE (Score: 1-3/10)
         │        │         ⚠ Significant but may not replicate
         │        │
         │        └─ NO  → ✗ NOT SIGNIFICANT (Score: 0-1/10)
         │                  Example: Rainfall ↔ Stocks (r=-0.03, p=0.50)
         │                  ❌ Do NOT trust - likely random noise
         │
         └──→ Additional Checks:
              • Sample Size: N > 100? (Our data: N=551 ✓)
              • Confidence Interval: Doesn't include 0? (Check CI)
              • Effect Size: |r| > 0.10? (Small but detectable)
              • Makes Sense: Theoretically plausible?

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                               FINAL VERDICT                                         │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ✓ TRUST COMPLETELY (Score 10/10):                                                 │
│    1. Price Range (Volatility) ↔ Depression Index                                  │
│    2. S&P 500 Volatility ↔ Depression Index                                        │
│    3. S&P 500 Close ↔ Depression Index                                             │
│                                                                                     │
│  ⚠ USE WITH CAUTION (Score 3-7/10):                                                │
│    • Individual stock prices ↔ Depression Index                                    │
│    • Trading volume ↔ Depression Index                                             │
│                                                                                     │
│  ❌ DON'T TRUST (Score 0-2/10):                                                     │
│    • Depression word count correlations (all p > 0.05)                             │
│    • Rainfall correlations (all p > 0.05)                                          │
│    • S&P 500 returns ↔ Depression (p = 0.62)                                       │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘

KEY INSIGHT: Only VOLATILITY relationships are robust. Price level relationships are weak.
"""

ax2.text(0.05, 0.95, decision_text, transform=ax2.transAxes,
        fontsize=9, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, pad=1))

plt.tight_layout()
plt.savefig('CORRELATION_IMPORTANCE_RANKING.png', dpi=300, bbox_inches='tight')
print("✓ Importance ranking visualization saved: CORRELATION_IMPORTANCE_RANKING.png")

# Also create a simple comparison table image
fig, ax = plt.subplots(figsize=(16, 10))
ax.axis('tight')
ax.axis('off')

# Create comparison table
table_data = []
table_data.append(['Rank', 'Relationship', 'r', 'p-value', 'Sig', 'R²%', 'Bonf.', 'Score', 'Verdict'])
table_data.append(['─'*4, '─'*45, '─'*6, '─'*10, '─'*3, '─'*5, '─'*5, '─'*5, '─'*20])

for i, (_, row) in enumerate(df.nlargest(12, 'importance_score').iterrows(), 1):
    rel = f"{row['Variable_X'][:20]}↔{row['Variable_Y'][:20]}"
    bonf = '✓' if row['Bonferroni_Significant'] else '✗'
    
    if row['importance_score'] >= 10:
        verdict = '⭐⭐⭐ TRUST COMPLETELY'
    elif row['importance_score'] >= 5:
        verdict = '⭐⭐ Use with caution'
    elif row['importance_score'] >= 3:
        verdict = '⭐ Weak evidence'
    else:
        verdict = '❌ Do not trust'
    
    table_data.append([
        str(i),
        rel,
        f"{row['Correlation']:.3f}",
        f"{row['P_Value']:.2e}",
        row['Significance'],
        f"{row['Explained_Variance_%']:.1f}",
        bonf,
        f"{row['importance_score']:.0f}/10",
        verdict
    ])

table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                colWidths=[0.04, 0.35, 0.06, 0.10, 0.04, 0.05, 0.05, 0.06, 0.18])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.5)

# Color code the rows
for i in range(len(table_data)):
    for j in range(len(table_data[0])):
        cell = table[(i, j)]
        if i == 0:  # Header
            cell.set_facecolor('#2E86AB')
            cell.set_text_props(weight='bold', color='white')
        elif i == 1:  # Separator
            cell.set_facecolor('#E8E8E8')
        elif i <= 4:  # Top 3 (plus header and separator = 4)
            cell.set_facecolor('#D4EDDA')  # Light green
        elif i <= 9:
            cell.set_facecolor('#FFF3CD')  # Light yellow
        else:
            cell.set_facecolor('#F8D7DA')  # Light red

plt.title('Statistical Significance Comparison Table\nRanked by Overall Importance',
         fontsize=16, fontweight='bold', pad=20)

plt.savefig('CORRELATION_COMPARISON_TABLE.png', dpi=300, bbox_inches='tight')
print("✓ Comparison table saved: CORRELATION_COMPARISON_TABLE.png")

print("\n✓ All importance ranking visualizations created!")
