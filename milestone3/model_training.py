"""
Milestone 3 - ML Modeling, Evaluation & Hyperparameter Tuning
Domain: Entertainment (Video Streaming)
Task: Multi-class Classification → Low / Medium / High Engagement

Models Trained:
  1. Logistic Regression     (Baseline - simple traditional model)
  2. Random Forest Classifier (Ensemble - more powerful model)

AI Tool Prompt Used (Claude Code / Aider):
"Write optimization routines for hyperparameter tuning using GridSearchCV
for both a Logistic Regression and Random Forest model. Automate the code
to serialize and save the final best model weights using joblib."
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
    accuracy_score, ConfusionMatrixDisplay
)

sns.set_theme(style="darkgrid", palette="Set2")
LABEL_MAP = {0: 'Low', 1: 'Medium', 2: 'High'}
COLORS    = ['#e74c3c', '#f39c12', '#2ecc71']

# ════════════════════════════════════════════════════════════
# STEP 1: LOAD PROCESSED DATA
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("   MILESTONE 3 — ML MODELING & EVALUATION")
print("=" * 60)

df = pd.read_csv('/home/claude/streaming_engagement_processed.csv')
print(f"\n📦 Loaded processed dataset: {df.shape}")

X = df.drop('engagement_level', axis=1)
y = df['engagement_level']

print(f"   Features (X): {X.shape[1]} columns")
print(f"   Target  (y): {y.value_counts().to_dict()}")

# ════════════════════════════════════════════════════════════
# STEP 2: TRAIN / TEST SPLIT (80% train, 20% test)
# ════════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n✂️  Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ════════════════════════════════════════════════════════════
# STEP 3: WHY NOT JUST USE ACCURACY?
# ════════════════════════════════════════════════════════════
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 WHY ACCURACY IS NOT THE BEST METRIC HERE:
   Our dataset originally had class imbalance (50% Low, 30% Medium,
   20% High). A dumb model that always predicts 'Low' would still
   get 50% accuracy — which looks decent but is useless!

   Instead we use:
   ✅ F1-Score (Macro) — balances precision & recall across all classes
   ✅ Precision        — of all predicted High, how many were correct?
   ✅ Recall           — of all actual High, how many did we catch?
   ✅ Confusion Matrix — see exactly where the model gets confused
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

# ════════════════════════════════════════════════════════════
# HELPER: Evaluate any model and print metrics
# ════════════════════════════════════════════════════════════
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average='macro')
    prec   = precision_score(y_test, y_pred, average='macro')
    rec    = recall_score(y_test, y_pred, average='macro')

    print(f"\n{'─'*50}")
    print(f"📊 {model_name} — Results")
    print(f"{'─'*50}")
    print(f"  Accuracy  : {acc:.4f}  ← not our main metric!")
    print(f"  F1 (Macro): {f1:.4f}  ✅ main metric")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"\n  Per-class report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Low','Medium','High']))
    return {'model': model_name, 'accuracy': acc,
            'f1': f1, 'precision': prec, 'recall': rec,
            'y_pred': y_pred}

# ════════════════════════════════════════════════════════════
# MODEL 1: LOGISTIC REGRESSION (Baseline)
# ════════════════════════════════════════════════════════════
print("\n🔷 MODEL 1: Logistic Regression (Baseline Traditional Model)")
print("   Simple model — finds a straight-line boundary between classes")

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_results = evaluate_model(lr, X_test, y_test, "Logistic Regression")

# ════════════════════════════════════════════════════════════
# MODEL 2: RANDOM FOREST (Ensemble)
# ════════════════════════════════════════════════════════════
print("\n🔶 MODEL 2: Random Forest (Ensemble Model)")
print("   Builds 100+ decision trees and votes on the final answer")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_results = evaluate_model(rf, X_test, y_test, "Random Forest")

# ════════════════════════════════════════════════════════════
# STEP 4: HYPERPARAMETER TUNING (Random Forest)
# ════════════════════════════════════════════════════════════
print("\n⚙️  Hyperparameter Tuning — Random Forest (GridSearchCV)")
print("   Trying different settings to find the best combination...")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth':    [None, 10, 20],
    'min_samples_split': [2, 5],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

print(f"\n✅ Best Parameters Found: {grid_search.best_params_}")
print(f"   Best CV F1 Score     : {grid_search.best_score_:.4f}")

best_rf = grid_search.best_estimator_
best_rf_results = evaluate_model(best_rf, X_test, y_test, "Random Forest (Tuned)")

# ════════════════════════════════════════════════════════════
# STEP 5: CROSS VALIDATION (checks model is not overfitting)
# ════════════════════════════════════════════════════════════
print("\n🔁 Cross Validation (5-Fold) — Checks consistency across data splits")
cv_scores = cross_val_score(best_rf, X, y, cv=5, scoring='f1_macro')
print(f"   CV F1 Scores : {[round(s,4) for s in cv_scores]}")
print(f"   Mean F1      : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ════════════════════════════════════════════════════════════
# STEP 6: PLOTS
# ════════════════════════════════════════════════════════════

# Plot 1: Model Comparison Bar Chart
def plot_model_comparison(lr_r, rf_r, best_rf_r):
    metrics = ['accuracy', 'f1', 'precision', 'recall']
    models  = ['Logistic Regression', 'Random Forest', 'RF (Tuned)']
    values  = [
        [lr_r[m] for m in metrics],
        [rf_r[m] for m in metrics],
        [best_rf_r[m] for m in metrics],
    ]

    x = np.arange(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (model, vals) in enumerate(zip(models, values)):
        bars = ax.bar(x + i*width, vals, width, label=model,
                      color=['#3498db','#e67e22','#2ecc71'][i],
                      edgecolor='black')

    ax.set_xticks(x + width)
    ax.set_xticklabels(['Accuracy', 'F1 (Macro)', 'Precision', 'Recall'],
                       fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
    ax.legend()
    ax.axhline(0.8, color='red', linestyle='--', alpha=0.5, label='0.8 threshold')
    plt.tight_layout()
    plt.savefig('/home/claude/plot_model_comparison.png', dpi=150)
    plt.close()
    print("\n✅ Model comparison plot saved.")

plot_model_comparison(lr_results, rf_results, best_rf_results)

# Plot 2: Confusion Matrices (side by side)
def plot_confusion_matrices(lr_r, best_rf_r, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    class_names = ['Low', 'Medium', 'High']

    for ax, results, title in zip(
        axes,
        [lr_r, best_rf_r],
        ['Logistic Regression', 'Random Forest (Tuned)']
    ):
        cm = confusion_matrix(y_test, results['y_pred'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=class_names)
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title(title, fontsize=12, fontweight='bold')

    fig.suptitle('Confusion Matrices — Predicted vs Actual',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/claude/plot_confusion_matrices.png', dpi=150)
    plt.close()
    print("✅ Confusion matrices plot saved.")

plot_confusion_matrices(lr_results, best_rf_results, y_test)

# Plot 3: Feature Importance from Random Forest
def plot_rf_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices     = np.argsort(importances)
    top_n       = 10
    indices     = indices[-top_n:]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(top_n),
            importances[indices],
            color='#3498db', edgecolor='black')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance Score')
    ax.set_title('Top 10 Feature Importances (Random Forest)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/claude/plot_rf_feature_importance.png', dpi=150)
    plt.close()
    print("✅ Feature importance plot saved.")

plot_rf_feature_importance(best_rf, list(X.columns))

# Plot 4: Cross-Validation Scores
def plot_cv_scores(cv_scores):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, 6), cv_scores, color='#9b59b6', edgecolor='black')
    ax.axhline(cv_scores.mean(), color='red', linestyle='--',
               label=f'Mean = {cv_scores.mean():.3f}')
    ax.set_xlabel('Fold')
    ax.set_ylabel('F1 Score (Macro)')
    ax.set_title('5-Fold Cross Validation — F1 Scores', fontsize=12, fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig('/home/claude/plot_cv_scores.png', dpi=150)
    plt.close()
    print("✅ Cross-validation plot saved.")

plot_cv_scores(cv_scores)

# ════════════════════════════════════════════════════════════
# STEP 7: SAVE BEST MODEL
# ════════════════════════════════════════════════════════════
joblib.dump(best_rf, '/home/claude/best_model.joblib')
print("\n💾 Best model saved → best_model.joblib")

# Verify it loads correctly
loaded_model = joblib.load('/home/claude/best_model.joblib')
verify_pred  = loaded_model.predict(X_test[:3])
print(f"   Verification prediction on 3 samples: {[LABEL_MAP[p] for p in verify_pred]}")

# ════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("✅ MILESTONE 3 COMPLETE — SUMMARY")
print("=" * 60)
print(f"""
  Model 1: Logistic Regression (Baseline)
    → F1 Score: {lr_results['f1']:.4f}

  Model 2: Random Forest (Tuned - BEST MODEL ✅)
    → F1 Score: {best_rf_results['f1']:.4f}
    → Best Params: {grid_search.best_params_}

  Cross Validation F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}

  Why Random Forest won:
    → Handles non-linear relationships better
    → Builds many trees → more robust predictions
    → Less affected by outliers and noisy features

  Saved model file: best_model.joblib
  Ready for Milestone 4 (API) ✅
""")
