import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

def create_clean_baseline():
    """Create a clean baseline dataset and evaluate using XGBoost"""
    
    # Load dataset
    df = pd.read_csv(r"F:\18. MAJOR PROJECT\Data\DATA_FOR_HEART_RISK_PRED.csv")
    
    # ✅ Step 1: Remove rows missing vital signs
    vital_cols = ['HR_max', 'NBPs_max', 'NBPd_max']
    df_clean = df.dropna(subset=vital_cols)
    
    # ✅ Step 2: Remove extreme physiological outliers
    df_clean = df_clean[
        (df_clean['HR_max'].between(30, 250)) &
        (df_clean['NBPs_max'].between(60, 250)) &
        (df_clean['NBPd_max'].between(30, 150))
    ]
    
    print(f"🧹 Cleaned data shape: {df_clean.shape}")
    print(f"🎯 Target distribution:\n{df_clean['heart_flag'].value_counts()}")
    
    # ✅ Step 3: Handle missing numeric values
    df_clean['creatinine_max'] = df_clean['creatinine_max'].fillna(1.0)
    df_clean['glucose_max'] = df_clean['glucose_max'].fillna(100)
    
    # ✅ Step 4: Select reliable features
    features = ['creatinine_max', 'glucose_max', 'HR_max', 'NBPs_max', 'NBPd_max', 'anchor_age']
    X = df_clean[features]
    y = df_clean['heart_flag']
    
    # ✅ Step 5: Model - XGBoost Classifier
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    # ✅ Step 6: Stratified Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    
    print(f"\n📈 Baseline XGBoost Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    
    return df_clean

if __name__ == "__main__":
    create_clean_baseline()
