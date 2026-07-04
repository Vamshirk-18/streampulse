"""
Milestone 2 - EDA & Feature Engineering Script
Domain: Entertainment (Video Streaming)
Task: Multi-class Classification → Low / Medium / High Engagement

AI Tool Prompt Used:
"Analyze this dataframe schema. Generate an EDA script using visualization
libraries to pinpoint feature importance and correlations. Write modular
functions to apply feature engineering pipelines dynamically."
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

# ── Style ────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="Set2")
LABEL_MAP = {0: 'Low', 1: 'Medium', 2: 'High'}
COLORS = ['#e74c3c', '#f39c12', '#2ecc71']

# ════════════════════════════════════════════════════════════
# STEP 1: LOAD DATA
# ════════════════════════════════════════════════════════════
df = pd.read_csv('/home/claude/streaming_engagement_raw.csv')
print("=" * 55)
print("MILESTONE 2 — EDA & FEATURE ENGINEERING")
print("=" * 55)
print(f"\n📦 Raw Data Shape: {df.shape}")
print(f"\n🔍 Data Types:\n{df.dtypes}")
print(f"\n📊 First 5 rows:\n{df.head()}")

# ════════════════════════════════════════════════════════════
# STEP 2: EDA — MISSING VALUES
# ════════════════════════════════════════════════════════════
def plot_missing_values(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    pct = (missing / len(df) * 100).round(2)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(missing.index, pct.values, color='#e74c3c', edgecolor='black')
    ax.set_title('Missing Values (%) Per Column', fontsize=14, fontweight='bold')
    ax.set_ylabel('Missing %')
    ax.set_ylim(0, 20)
    for bar, val in zip(bars, pct.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val}%', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig('/home/claude/plot_missing_values.png', dpi=150)
    plt.close()
    print(f"\n⚠️  Missing Values:\n{pct}")

plot_missing_values(df)

# ════════════════════════════════════════════════════════════
# STEP 3: EDA — CLASS IMBALANCE
# ════════════════════════════════════════════════════════════
def plot_class_distribution(df, col='engagement_level'):
    counts = df[col].value_counts().sort_index()
    labels = [LABEL_MAP[i] for i in counts.index]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts.values, color=COLORS, edgecolor='black')
    ax.set_title('Class Distribution (Engagement Level)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Count')
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                str(val), ha='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/claude/plot_class_distribution.png', dpi=150)
    plt.close()
    print(f"\n📊 Class Distribution:\n{counts.rename(LABEL_MAP)}")

plot_class_distribution(df)

# ════════════════════════════════════════════════════════════
# STEP 4: EDA — FEATURE DISTRIBUTIONS
# ════════════════════════════════════════════════════════════
def plot_feature_distributions(df):
    num_cols = ['weekly_watch_hours', 'skip_rate', 'avg_rating_given',
                'logins_per_week', 'days_since_last_login',
                'content_diversity_score', 'watch_streak_days']

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        for j, level in enumerate([0, 1, 2]):
            data = df[df['engagement_level'] == level][col].dropna()
            axes[i].hist(data, bins=25, alpha=0.6, color=COLORS[j],
                         label=LABEL_MAP[level], edgecolor='none')
        axes[i].set_title(col, fontsize=10, fontweight='bold')
        axes[i].legend(fontsize=8)

    for k in range(len(num_cols), len(axes)):
        axes[k].set_visible(False)

    fig.suptitle('Feature Distributions by Engagement Level', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/claude/plot_feature_distributions.png', dpi=150)
    plt.close()
    print("\n✅ Feature distribution plot saved.")

plot_feature_distributions(df)

# ════════════════════════════════════════════════════════════
# STEP 5: EDA — CORRELATION HEATMAP
# ════════════════════════════════════════════════════════════
def plot_correlation_heatmap(df):
    num_df = df.select_dtypes(include=np.number).dropna()
    corr = num_df.corr()

    fig, ax = plt.subplots(figsize=(10, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                ax=ax, linewidths=0.5, vmin=-1, vmax=1)
    ax.set_title('Feature Correlation Heatmap', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/claude/plot_correlation_heatmap.png', dpi=150)
    plt.close()
    print("\n✅ Correlation heatmap saved.")

plot_correlation_heatmap(df)

# ════════════════════════════════════════════════════════════
# STEP 6: EDA — BOXPLOTS (OUTLIER DETECTION)
# ════════════════════════════════════════════════════════════
def plot_boxplots(df):
    cols = ['weekly_watch_hours', 'skip_rate', 'logins_per_week',
            'notifications_clicked', 'watch_streak_days']

    fig, axes = plt.subplots(1, 5, figsize=(16, 5))
    for i, col in enumerate(cols):
        df.boxplot(column=col, by='engagement_level', ax=axes[i],
                   patch_artist=True)
        axes[i].set_title(col, fontsize=9)
        axes[i].set_xlabel('Engagement Level')
    fig.suptitle('Boxplots: Outlier Detection by Engagement Level',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/claude/plot_boxplots.png', dpi=150)
    plt.close()
    print("✅ Boxplot saved.")

plot_boxplots(df)

# ════════════════════════════════════════════════════════════
# STEP 7: CLEANING — HANDLE MISSING VALUES
# ════════════════════════════════════════════════════════════
def handle_missing_values(df):
    df = df.copy()
    # Fill numeric columns with median (robust to outliers)
    for col in ['weekly_watch_hours', 'skip_rate', 'avg_rating_given']:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"  Filled '{col}' missing values with median: {median_val:.2f}")
    return df

print("\n🔧 Handling Missing Values...")
df_clean = handle_missing_values(df)
print(f"  Missing after cleaning: {df_clean.isnull().sum().sum()}")

# ════════════════════════════════════════════════════════════
# STEP 8: CLEANING — HANDLE OUTLIERS (IQR Capping)
# ════════════════════════════════════════════════════════════
def cap_outliers(df, col):
    df = df.copy()
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    before = ((df[col] < lower) | (df[col] > upper)).sum()
    df[col] = df[col].clip(lower=lower, upper=upper)
    print(f"  '{col}': capped {before} outliers → range [{lower:.2f}, {upper:.2f}]")
    return df

print("\n🔧 Capping Outliers...")
df_clean = cap_outliers(df_clean, 'weekly_watch_hours')
df_clean = cap_outliers(df_clean, 'notifications_clicked')

# ════════════════════════════════════════════════════════════
# STEP 9: ENCODING — CATEGORICAL COLUMN
# ════════════════════════════════════════════════════════════
def encode_categoricals(df):
    df = df.copy()
    sub_map = {'Free': 0, 'Basic': 1, 'Premium': 2}
    df['subscription_type'] = df['subscription_type'].map(sub_map)
    print(f"\n  subscription_type encoded: Free=0, Basic=1, Premium=2")
    return df

print("\n🔧 Encoding Categorical Columns...")
df_clean = encode_categoricals(df_clean)

# ════════════════════════════════════════════════════════════
# STEP 10: FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════
def engineer_features(df):
    df = df.copy()

    # Feature 1: Engagement Ratio — how much they watch vs how often they skip
    df['engagement_ratio'] = df['weekly_watch_hours'] / (df['skip_rate'] + 0.01)

    # Feature 2: Activity Score — combines logins and streak
    df['activity_score'] = df['logins_per_week'] * 0.5 + df['watch_streak_days'] * 0.5

    # Feature 3: Interaction — Premium users who watch a lot
    df['premium_watch'] = df['subscription_type'] * df['weekly_watch_hours']

    # Feature 4: Recency flag — hasn't logged in for over 14 days
    df['inactive_flag'] = (df['days_since_last_login'] > 14).astype(int)

    # Feature 5: Notification responsiveness
    df['notification_rate'] = df['notifications_clicked'] / (df['logins_per_week'] + 1)

    print("\n✅ Engineered Features Added:")
    print("  - engagement_ratio (watch_hours / skip_rate)")
    print("  - activity_score (logins + streak)")
    print("  - premium_watch (subscription × watch_hours)")
    print("  - inactive_flag (days_since_login > 14)")
    print("  - notification_rate (clicks / logins)")
    return df

print("\n⚙️  Engineering Features...")
df_feat = engineer_features(df_clean)

# ════════════════════════════════════════════════════════════
# STEP 11: HANDLE CLASS IMBALANCE (Oversampling minority classes)
# ════════════════════════════════════════════════════════════
def balance_classes(df, target='engagement_level'):
    df = df.copy()
    counts = df[target].value_counts()
    max_count = counts.max()
    print(f"\n⚖️  Balancing Classes (Oversampling to {max_count} each)...")

    balanced_dfs = []
    for label in df[target].unique():
        subset = df[df[target] == label]
        if len(subset) < max_count:
            subset = resample(subset, replace=True,
                              n_samples=max_count, random_state=42)
        balanced_dfs.append(subset)
        print(f"  Class {label} ({LABEL_MAP[label]}): {counts[label]} → {max_count}")

    return pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

df_balanced = balance_classes(df_feat)
print(f"\n  Balanced dataset shape: {df_balanced.shape}")

# ════════════════════════════════════════════════════════════
# STEP 12: FEATURE SCALING
# ════════════════════════════════════════════════════════════
def scale_features(df, target='engagement_level'):
    df = df.copy()
    scaler = StandardScaler()
    feature_cols = [c for c in df.columns if c != target]
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    print(f"\n📐 Scaled {len(feature_cols)} features using StandardScaler.")
    return df, scaler

df_final, scaler = scale_features(df_balanced)

# ════════════════════════════════════════════════════════════
# STEP 13: FEATURE IMPORTANCE (Correlation with target)
# ════════════════════════════════════════════════════════════
def plot_feature_importance(df, target='engagement_level'):
    corr = df.corr()[target].drop(target).sort_values(key=abs, ascending=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in corr.values]
    ax.barh(corr.index, corr.values, color=colors, edgecolor='black')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title('Feature Correlation with Engagement Level\n(Feature Importance Proxy)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Correlation Coefficient')
    plt.tight_layout()
    plt.savefig('/home/claude/plot_feature_importance.png', dpi=150)
    plt.close()
    print("\n✅ Feature importance plot saved.")
    print(f"\n📌 Top Features:\n{corr.tail(5)[::-1]}")

plot_feature_importance(df_final)

# ════════════════════════════════════════════════════════════
# STEP 14: SAVE FINAL DATASET
# ════════════════════════════════════════════════════════════
df_final.to_csv('/home/claude/streaming_engagement_processed.csv', index=False)
print("\n" + "=" * 55)
print("✅ MILESTONE 2 COMPLETE!")
print(f"   Final processed dataset: streaming_engagement_processed.csv")
print(f"   Final shape: {df_final.shape}")
print(f"   Total features: {df_final.shape[1] - 1}")
print("=" * 55)
