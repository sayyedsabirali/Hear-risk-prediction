import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

def analyze_cleaned_data():
    """Analyze why we're only getting 76% accuracy"""
    
    # Load and clean data (same as baseline)
    df = pd.read_csv(r"F:\18. MAJOR PROJECT\Data\DATA_FOR_HEART_RISK_PRED.csv")
    
    vital_cols = ['HR_max', 'NBPs_max', 'NBPd_max']
    df_clean = df.dropna(subset=vital_cols)
    df_clean = df_clean[
        (df_clean['HR_max'].between(30, 250)) &
        (df_clean['NBPs_max'].between(60, 250)) &
        (df_clean['NBPd_max'].between(30, 150))
    ]
    df_clean['creatinine_max'] = df_clean['creatinine_max'].fillna(1.0)
    df_clean['glucose_max'] = df_clean['glucose_max'].fillna(100)
    
    print("="*60)
    print("🔍 DEEP DATA ANALYSIS")
    print("="*60)
    
    # 1. Check feature correlations with target
    features = ['creatinine_max', 'glucose_max', 'HR_max', 'NBPs_max', 'NBPd_max', 'anchor_age']
    X = df_clean[features]
    y = df_clean['heart_flag']
    
    # Correlation analysis
    correlation_with_target = X.corrwith(y).abs().sort_values(ascending=False)
    print("\n📊 Correlation with Target (heart_flag):")
    print(correlation_with_target)
    
    # 2. Check if features have predictive power
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_df = pd.DataFrame({'feature': X.columns, 'mi_score': mi_scores})
    mi_df = mi_df.sort_values('mi_score', ascending=False)
    print("\n🎯 Mutual Information Scores:")
    print(mi_df)
    
    # 3. Check feature distributions by target
    print("\n📈 Feature Statistics by Heart Flag:")
    for feature in features:
        heart_0 = df_clean[df_clean['heart_flag'] == 0][feature]
        heart_1 = df_clean[df_clean['heart_flag'] == 1][feature]
        
        print(f"\n{feature}:")
        print(f"  Heart=0: mean={heart_0.mean():.2f}, std={heart_0.std():.2f}")
        print(f"  Heart=1: mean={heart_1.mean():.2f}, std={heart_1.std():.2f}")
        print(f"  T-test p-value: {ttest_ind(heart_0, heart_1, nan_policy='omit').pvalue:.4f}")
    
    # 4. Check data balance after cleaning
    print(f"\n⚖️ Data Balance:")
    print(f"Total samples: {len(df_clean)}")
    print(f"Heart Flag 0: {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
    print(f"Heart Flag 1: {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")
    
    return df_clean

if __name__ == "__main__":
    from scipy.stats import ttest_ind
    analyze_cleaned_data()

# op - class imbalance