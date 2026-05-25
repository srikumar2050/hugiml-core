#!/usr/bin/env python3
"""
Mobile Money Fraud Detection
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
from hugiml import HUGIMLClassifierNative

# Load dataset
df = pd.read_csv('nb06_mobile_money_fraud_data.csv')
print(f"Dataset loaded: {len(df):,} rows")

# Prepare features and target
X = df.drop(columns=['isFraud'])
y = df['isFraud'].astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Train HUG-IML model
clf = HUGIMLClassifierNative(B=10, L=1, G=1e-4, topK=100)
X_enc, y_enc = clf.prepareXy(X, y)
X_train_enc, X_test_enc, y_train_enc, y_test_enc = train_test_split(
    X_enc, y_enc, test_size=0.3, random_state=42, stratify=y_enc
)

clf.fit(X_train_enc, y_train_enc)

# Evaluate
y_pred_proba = clf.predict_proba(X_test_enc)[:, 1]
auc = roc_auc_score(y_test_enc, y_pred_proba)
n_patterns = len(clf.get_hug_features())

print(f"\nModel Performance:")
print(f"  Patterns: {n_patterns}")
print(f"  AUC-ROC: {auc:.4f}")

# Display top patterns
patterns = clf.feature_importances().nlargest(10, 'abs_coefficient')
print(f"\nTop 10 Patterns:")
for i, (_, row) in enumerate(patterns.iterrows(), 1):
    print(f"  {i}. {row['pattern']} ({row['coefficient']:+.4f})")
