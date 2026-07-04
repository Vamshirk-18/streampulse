"""
Milestone 1 - Synthetic Data Generation Script
Domain: Entertainment (Video Streaming)
Task: Multi-class Classification
Target: user_engagement_level → 0=Low, 1=Medium, 2=High

AI Tool Prompt Used (Cursor/Copilot):
"Generate a synthetic video streaming user dataset with 5000 rows.
Include features like age, weekly_watch_hours, skip_rate, avg_rating_given,
logins_per_week, subscription_type, content_diversity_score,
days_since_last_login, notifications_clicked, and a multi-class
engagement label (Low/Medium/High).
Inject real-world flaws: class imbalance (50% Low, 30% Medium, 20% High),
10-15% missing values in key columns, and correlated anomalies
where high skip_rate correlates with Low engagement."
"""

import pandas as pd
import numpy as np

np.random.seed(42)
N = 5000

# ── Base Features ──────────────────────────────────────────
age                     = np.random.randint(16, 65, N)
weekly_watch_hours      = np.round(np.random.exponential(scale=8, size=N), 2)
skip_rate               = np.round(np.random.uniform(0, 1, N), 2)
avg_rating_given        = np.round(np.random.uniform(1, 5, N), 1)
logins_per_week         = np.random.poisson(lam=4, size=N)
subscription_type       = np.random.choice(['Free', 'Basic', 'Premium'], N, p=[0.5, 0.3, 0.2])
content_diversity_score = np.round(np.random.uniform(0, 1, N), 2)  # genres explored
days_since_last_login   = np.random.randint(0, 60, N)
notifications_clicked   = np.random.poisson(lam=3, size=N)
watch_streak_days       = np.random.randint(0, 30, N)

# ── Target: Engagement Level (class imbalance 50/30/20) ────
# Base score driven by features
engagement_score = (
      weekly_watch_hours * 0.4
    + logins_per_week    * 0.3
    - skip_rate          * 5
    + avg_rating_given   * 0.5
    + content_diversity_score * 3
    + watch_streak_days  * 0.2
    - days_since_last_login * 0.1
    + notifications_clicked * 0.3
)

# Bin into 3 classes with target distribution
low_thresh    = np.percentile(engagement_score, 50)
medium_thresh = np.percentile(engagement_score, 80)

engagement_level = np.where(
    engagement_score < low_thresh, 0,          # Low    (50%)
    np.where(engagement_score < medium_thresh, 1, 2)   # Medium (30%), High (20%)
)

# ── Assemble DataFrame ──────────────────────────────────────
df = pd.DataFrame({
    'age':                      age,
    'weekly_watch_hours':       weekly_watch_hours,
    'skip_rate':                skip_rate,
    'avg_rating_given':         avg_rating_given,
    'logins_per_week':          logins_per_week,
    'subscription_type':        subscription_type,
    'content_diversity_score':  content_diversity_score,
    'days_since_last_login':    days_since_last_login,
    'notifications_clicked':    notifications_clicked,
    'watch_streak_days':        watch_streak_days,
    'engagement_level':         engagement_level
})

# ── Inject Real-World Flaws ─────────────────────────────────

# Flaw 1: Missing values (~12%) in 3 key columns
for col in ['weekly_watch_hours', 'skip_rate', 'avg_rating_given']:
    idx = np.random.choice(N, size=int(N * 0.12), replace=False)
    df.loc[idx, col] = np.nan

# Flaw 2: Correlated anomaly — Low engagement users have very high skip_rate
low_idx = df[df['engagement_level'] == 0].index
anomaly_idx = np.random.choice(low_idx, size=int(len(low_idx) * 0.2), replace=False)
df.loc[anomaly_idx, 'skip_rate'] = np.round(np.random.uniform(0.85, 1.0, len(anomaly_idx)), 2)

# Flaw 3: Outliers in weekly_watch_hours
outlier_idx = np.random.choice(N, size=30, replace=False)
df.loc[outlier_idx, 'weekly_watch_hours'] = np.round(np.random.uniform(80, 150, 30), 2)

# ── Save ────────────────────────────────────────────────────
df.to_csv('/home/claude/streaming_engagement_raw.csv', index=False)

print("✅ Dataset generated: streaming_engagement_raw.csv")
print(f"Shape: {df.shape}")
print(f"\nEngagement Level Distribution:")
print(df['engagement_level'].value_counts().rename({0:'Low',1:'Medium',2:'High'}))
print(f"\nMissing Values:\n{df.isnull().sum()}")
