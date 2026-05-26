"""
Credit Risk Modeling with SR 11-7 Compliance
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("="*80)
print("CREDIT RISK MODELING WITH SR 11-7 COMPLIANCE")
print("="*80)
print()

# =============================================================================
# DATA PREPARATION - Real German Credit Dataset
# =============================================================================
print("Loading German Credit Dataset (UCI ML Repository)...")

# Column names from UCI specification
GERMAN_COLS = [
    "checking_acct", "duration", "credit_history", "purpose", "credit_amount",
    "savings", "employment", "installment_rate", "personal_status", "other_debtors",
    "residence_since", "property", "age", "other_plans", "housing",
    "existing_credits", "job", "num_dependents", "telephone", "foreign_worker", "target"
]

# Load real German Credit data
df = pd.read_csv('german.data', sep=' ', header=None, names=GERMAN_COLS)

# Original coding: 1 = Good, 2 = Bad → recode to 0 (bad) / 1 (good)
df['target'] = (df['target'] == 1).astype(int)  # 1 = good credit

# Separate features and target
X = df.drop(columns=['target'])
y = df['target']

print(f" Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
print(f" Target distribution: Good={y.sum()} ({y.sum()/len(y)*100:.1f}%), Bad={len(y)-y.sum()} ({(len(y)-y.sum())/len(y)*100:.1f}%)")
print(f" Real German Credit data from UCI ML Repository")

# =============================================================================
# MODEL TRAINING
# =============================================================================
from hugiml import HUGIMLClassifierNative
from hugiml.calibration import evaluate_calibration
from hugiml.metrics import compute_all_metrics

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nData Split:")
print(f"  Training: {len(X_train)} samples ({y_train.sum()} good, {len(y_train)-y_train.sum()} bad)")
print(f"  Test: {len(X_test)} samples ({y_test.sum()} good, {len(y_test)-y_test.sum()} bad)")

# Initialize and train model with optimized hyperparameters
# B=15 for good discretization, L=1 for maximum interpretability, G=5e-4 for comprehensive patterns
clf = HUGIMLClassifierNative(B=15, L=1, G=5e-4, topK=100)
X_enc, y_enc = clf.prepareXy(X, y)
X_train_enc, X_test_enc, y_train_enc, y_test_enc = train_test_split(
    X_enc, y_enc, test_size=0.3, random_state=42, stratify=y_enc
)

clf.fit(X_train_enc, y_train_enc)
print(f"\n Model trained: {len(clf.get_hug_features())} patterns discovered")

# Predictions
y_pred = clf.predict(X_test_enc)
y_proba = clf.predict_proba(X_test_enc)[:, 1]
auc = roc_auc_score(y_test_enc, y_proba)

print(f" AUC-ROC: {auc:.4f}")

# =============================================================================
# GENERATE ALL VISUALIZATIONS
# =============================================================================
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 1. ROC Curve
print("\n1. ROC Curve...")
fpr, tpr, thresholds = roc_curve(y_test_enc, y_proba)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(fpr, tpr, 'b-', linewidth=2.5, label=f'HUG-IML (AUC = {auc:.4f})')
ax.plot([0, 1], [0, 1], 'r--', linewidth=1.5, label='Random Classifier')
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'go', markersize=12, 
        label=f'Optimal Threshold = {optimal_threshold:.3f}', markeredgecolor='black', markeredgewidth=1.5)
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('ROC Curve - Credit Risk Model', fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
plt.tight_layout()
plt.savefig('nb04_roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Calibration Plot
print("2. Calibration Plot...")
y_test_array = y_test_enc if isinstance(y_test_enc, np.ndarray) else y_test_enc.values
cal_result = evaluate_calibration(y_test_array, y_proba)

n_bins = 10
bins = np.linspace(0, 1, n_bins + 1)
bin_indices = np.digitize(y_proba, bins) - 1
bin_indices = np.clip(bin_indices, 0, n_bins - 1)
bin_sums = np.bincount(bin_indices, weights=y_test_array, minlength=n_bins)
bin_counts = np.bincount(bin_indices, minlength=n_bins)
empirical_probs = np.divide(bin_sums, bin_counts, where=bin_counts > 0, 
                            out=np.zeros_like(bin_sums, dtype=float))
predicted_probs = np.bincount(bin_indices, weights=y_proba, minlength=n_bins) / np.maximum(bin_counts, 1)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
mask = bin_counts > 0
ax.plot(predicted_probs[mask], empirical_probs[mask], 
        'bo-', linewidth=2.5, markersize=10, label='HUG-IML Model', 
        markeredgecolor='black', markeredgewidth=1.5)
ax2 = ax.twinx()
ax2.hist(y_proba, bins=bins, alpha=0.3, color='gray')
ax2.set_ylabel('Count', fontsize=11, color='gray')
ax2.tick_params(axis='y', labelcolor='gray')
ax.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
ax.set_ylabel('Empirical Probability', fontsize=12, fontweight='bold')
ax.set_title(f'Calibration Plot (ECE = {cal_result.ece:.4f})', fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
plt.tight_layout()
plt.savefig('nb04_calibration_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Confusion Matrix
print("3. Confusion Matrix...")
cm = confusion_matrix(y_test_enc, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, 
            square=True, linewidths=2, linecolor='black',
            xticklabels=['Bad Credit', 'Good Credit'],
            yticklabels=['Bad Credit', 'Good Credit'],
            annot_kws={'size': 16, 'weight': 'bold'})
ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax.set_title('Confusion Matrix - Credit Risk Model', fontsize=14, fontweight='bold', pad=15)
accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
plt.text(1, -0.15, f'Overall Accuracy: {accuracy:.2%}', 
         ha='center', va='top', fontsize=12, fontweight='bold',
         transform=ax.transAxes)
plt.tight_layout()
plt.savefig('nb04_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Pattern Importance
print("4. Pattern Importance...")
importances_df = clf.feature_importances()
top_15 = importances_df.nlargest(15, 'abs_coefficient')
fig, ax = plt.subplots(figsize=(9, 7))
colors = ['green' if c > 0 else 'red' for c in top_15['coefficient']]
ax.barh(range(len(top_15)), top_15['abs_coefficient'], color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
for i, (idx, row) in enumerate(top_15.iterrows()):
    ax.text(row['abs_coefficient'] + 0.05, i, f"{row['coefficient']:+.3f}", 
            va='center', fontsize=9, fontweight='bold')
ax.set_yticks(range(len(top_15)))
ax.set_yticklabels(top_15['pattern'], fontsize=9)
ax.set_xlabel('Absolute Coefficient Value', fontsize=12, fontweight='bold')
ax.set_title('Top 15 Credit Risk Patterns by Importance\n(Green=Approve, Red=Deny)', 
             fontsize=14, fontweight='bold', pad=15)
ax.grid(True, axis='x', alpha=0.3)
ax.invert_yaxis()
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='green', alpha=0.7, label='Positive (Approve)'),
                  Patch(facecolor='red', alpha=0.7, label='Negative (Deny)')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
plt.tight_layout()
plt.savefig('nb04_pattern_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Age Disparity
print("5. Age Disparity...")
X_test_with_age = X_test.copy()
X_test_with_age['prediction'] = y_pred
X_test_with_age['prob_good'] = y_proba

if 'age' in X_test_with_age.columns:
    X_test_with_age['age_group'] = pd.cut(
        X_test_with_age['age'], 
        bins=[18, 26, 35, 50, 100],  # Changed to match [19,26) pattern
        labels=['19-25', '26-35', '36-50', '50+'],
        right=False
    )
    
    age_analysis = X_test_with_age.groupby('age_group').agg({
        'prediction': ['mean', 'count'],
        'prob_good': 'mean'
    }).round(4)
    age_analysis.columns = ['Approval_Rate', 'Count', 'Avg_Score']
    
    overall_approval = y_pred.mean()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    age_groups = age_analysis.index.tolist()
    approval_rates = age_analysis['Approval_Rate'].values
    colors_disparity = ['red' if abs((r - overall_approval) / overall_approval) > 0.20 
                        else 'orange' if abs((r - overall_approval) / overall_approval) > 0.10
                        else 'green' for r in approval_rates]
    
    bars = ax1.bar(age_groups, approval_rates, color=colors_disparity, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.axhline(y=overall_approval, color='blue', linestyle='--', linewidth=2, label=f'Overall Rate ({overall_approval:.1%})')
    ax1.axhline(y=overall_approval * 0.8, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
    ax1.axhline(y=overall_approval * 1.2, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
    
    for bar, rate in zip(bars, approval_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_ylabel('Approval Rate', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Age Group', fontsize=12, fontweight='bold')
    ax1.set_title('Credit Approval Rates by Age Group', fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.set_ylim([0, max(approval_rates) * 1.15])
    
    counts = age_analysis['Count'].values
    ax2.bar(age_groups, counts, color='steelblue', alpha=0.7, edgecolor='black', linewidth=2)
    for i, (ag, count) in enumerate(zip(age_groups, counts)):
        ax2.text(i, count + 2, str(int(count)), ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    ax2.set_ylabel('Number of Applicants', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Age Group', fontsize=12, fontweight='bold')
    ax2.set_title('Sample Distribution by Age Group', fontsize=13, fontweight='bold', pad=10)
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nb04_age_disparity.png', dpi=300, bbox_inches='tight')
    plt.close()

# 6. Profile Bin Plots - Top 4 Features
print("6. Profile Bin Plots (Top 4 Features)...")

# Get top numerical features by importance
top_patterns = importances_df.nlargest(20, 'abs_coefficient')
top_features = []
for pattern in top_patterns['pattern']:
    # Extract feature name (before '=')
    if '=' in pattern:
        feat = pattern.split('=')[0]
        if feat not in top_features and feat in X.select_dtypes(include=[np.number]).columns:
            top_features.append(feat)
    if len(top_features) >= 4:
        break

# Ensure we have at least 4 features
if len(top_features) < 4:
    # Add more numerical features
    num_features = X.select_dtypes(include=[np.number]).columns.tolist()
    for feat in num_features:
        if feat not in top_features:
            top_features.append(feat)
        if len(top_features) >= 4:
            break

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for idx, feature in enumerate(top_features[:4]):
    ax = axes[idx]
    
    # Get feature-specific patterns and their coefficients
    feature_patterns = importances_df[importances_df['pattern'].str.startswith(f"{feature}=")].copy()
    
    if len(feature_patterns) > 0:
        # Sort by the bin range (extract lower bound)
        feature_patterns = feature_patterns.sort_index()
        
        # Plot
        x_pos = np.arange(len(feature_patterns))
        coefficients = feature_patterns['coefficient'].values
        colors = ['green' if c > 0 else 'red' for c in coefficients]
        
        bars = ax.bar(x_pos, coefficients, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_xticks(x_pos)
        
        # Extract cleaner labels
        labels = []
        for pattern in feature_patterns['pattern']:
            range_str = pattern.split('=')[1] if '=' in pattern else pattern
            # Truncate long ranges
            if len(range_str) > 12:
                range_str = range_str[:12] + '...'
            labels.append(range_str)
        
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Coefficient', fontsize=10, fontweight='bold')
        ax.set_title(f'{feature.replace("_", " ").title()}', fontsize=11, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        
        # Add coefficient values on bars
        for i, (bar, coef) in enumerate(zip(bars, coefficients)):
            height = bar.get_height()
            if abs(height) > 0.01:  # Only show if significant
                ax.text(bar.get_x() + bar.get_width()/2., 
                       height + 0.05 if height > 0 else height - 0.05,
                       f"{coef:+.2f}", ha='center', 
                       va='bottom' if height > 0 else 'top', 
                       fontsize=7, fontweight='bold')
    else:
        # No patterns found - show empty plot with message
        ax.text(0.5, 0.5, f'No patterns for\n{feature}', 
               ha='center', va='center', fontsize=10, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])

plt.suptitle('Feature Impact Analysis - Coefficient by Value Range\n(Green=Approve, Red=Deny)', 
             fontsize=13, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('nb04_profile_bin_plots.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n All visualizations generated successfully!")

# Save key metrics for HTML
metrics = {
    'auc': auc,
    'accuracy': accuracy,
    'n_patterns': len(clf.get_hug_features()),
    'ece': cal_result.ece,
    'cm': cm,
    'age_analysis': age_analysis,
    'overall_approval': overall_approval,
    'top_patterns': importances_df.nlargest(10, 'abs_coefficient')
}

import pickle
with open('nb04_metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)

print("\n Analysis complete!")
