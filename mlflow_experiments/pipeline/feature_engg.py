import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Load data
df = pd.read_csv(r"F:\18. MAJOR PROJECT\Data\DATA_FOR_HEART_RISK_PRED.csv")

# MIMIC-IV specific cleaning
vital_cols = ['HR_max', 'NBPs_max', 'NBPd_max']
df_clean = df.dropna(subset=vital_cols)
df_clean = df_clean[
    (df_clean['HR_max'].between(30, 250)) &
    (df_clean['NBPs_max'].between(60, 250)) & 
    (df_clean['NBPd_max'].between(30, 150))
]

# MIMIC-IV typical values for imputation
df_clean['creatinine_max'] = df_clean['creatinine_max'].fillna(1.2)  # ICU typical
df_clean['glucose_max'] = df_clean['glucose_max'].fillna(140)        # ICU typical
df_clean['ast_max'] = df_clean['ast_max'].fillna(40)                 # ICU typical  
df_clean['alt_max'] = df_clean['alt_max'].fillna(35)                 # ICU typical

print("🏥 MIMIC-IV SPECIFIC FEATURES...")

# ICU-Specific Features
features = ['creatinine_max', 'glucose_max', 'HR_max', 'NBPs_max', 'NBPd_max', 'anchor_age']

# 1. ICU Risk Scores
df_clean['sofa_like_score'] = (
    (df_clean['creatinine_max'] > 1.2).astype(int) +
    (df_clean['glucose_max'] > 180).astype(int) +
    ((df_clean['HR_max'] < 60) | (df_clean['HR_max'] > 120)).astype(int) +
    ((df_clean['NBPs_max'] < 90) | (df_clean['NBPs_max'] > 160)).astype(int)
)
features.append('sofa_like_score')

# 2. Organ Dysfunction Markers
df_clean['kidney_dysfunction'] = (df_clean['creatinine_max'] > 1.5).astype(int)
df_clean['glucose_dysregulation'] = (df_clean['glucose_max'] > 200).astype(int) 
df_clean['hemodynamic_instability'] = ((df_clean['NBPs_max'] < 100) | (df_clean['HR_max'] > 130)).astype(int)
features.extend(['kidney_dysfunction', 'glucose_dysregulation', 'hemodynamic_instability'])

# 3. Age groups for ICU
df_clean['icu_age_group'] = pd.cut(df_clean['anchor_age'], 
                                 bins=[18, 50, 65, 75, 100],
                                 labels=[0, 1, 2, 3])
features.append('icu_age_group')

X = df_clean[features]
y = df_clean['heart_flag']

print(f"📊 MIMIC-IV Features: {len(features)}")
print(f"🎯 Samples: {len(X)}")

# ICU-optimized model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

print(f"✅ MIMIC-IV Optimized: {scores.mean():.4f} ± {scores.std():.4f}")

# Check if we beat simple age rule
age_rule_accuracy = (df_clean['anchor_age'] > 65).astype(int) == y
age_accuracy = age_rule_accuracy.mean()
print(f"📊 Age > 65 Rule: {age_accuracy:.4f}")
print(f"📈 Improvement: {scores.mean() - age_accuracy:.4f}")