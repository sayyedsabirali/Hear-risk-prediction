import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Load and clean data
df = pd.read_csv(r"F:\18. MAJOR PROJECT\Data\DATA_FOR_HEART_RISK_PRED.csv")

vital_cols = ['HR_max', 'NBPs_max', 'NBPd_max']
df_clean = df.dropna(subset=vital_cols)
df_clean = df_clean[
    (df_clean['HR_max'].between(30, 250)) &
    (df_clean['NBPs_max'].between(60, 250)) & 
    (df_clean['NBPd_max'].between(30, 150))
]

# Handle missing values
df_clean['creatinine_max'] = df_clean['creatinine_max'].fillna(1.0)
df_clean['glucose_max'] = df_clean['glucose_max'].fillna(100)

# Features and target
features = ['creatinine_max', 'glucose_max', 'HR_max', 'NBPs_max', 'NBPd_max', 'anchor_age']
X = df_clean[features]
y = df_clean['heart_flag']

print("📊 Current Class Distribution:")
print(y.value_counts())
print(f"Imbalance Ratio: {(y == 0).sum() / len(y):.2%} vs {(y == 1).sum() / len(y):.2%}")

# Strategy 1: Simple Class Weight
print("\n🔄 TRYING CLASS WEIGHT BALANCING...")
model = XGBClassifier(
    n_estimators=200,
    max_depth=6, 
    learning_rate=0.05,
    scale_pos_weight=(y == 0).sum() / (y == 1).sum(),  # Fix imbalance
    random_state=42,
    eval_metric='logloss'
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

print(f"✅ Balanced XGBoost Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")