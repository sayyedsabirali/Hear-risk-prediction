import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.data_cleaner import DataCleaner
from config.data_config import DataConfig
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import xgboost as xgb

def run_advanced_analysis():
    """Run advanced cleaning and modeling"""
    
    # Load data
    config = DataConfig()
    config.DATA_PATH = r"F:\18. MAJOR PROJECT\Data\DATA_FOR_HEART_RISK_PRED.csv"
    config.TARGET_COLUMN = "heart_flag"
    
    print("🚀 LOADING ORIGINAL DATA...")
    df = pd.read_csv(config.DATA_PATH)
    print(f"Original data shape: {df.shape}")
    
    # Clean data
    cleaner = DataCleaner()
    df_clean = cleaner.clean_dataset(df)
    
    print(f"\n🎯 FINAL CLEANED DATA: {df_clean.shape}")
    print(f"📊 Target distribution:\n{df_clean[config.TARGET_COLUMN].value_counts()}")
    
    # Prepare for modeling
    X = df_clean.drop(columns=[config.TARGET_COLUMN])
    y = df_clean[config.TARGET_COLUMN]
    
    # Handle categorical variables
    X = pd.get_dummies(X, drop_first=True)
    
    print(f"\n🤖 MODELING WITH {X.shape[1]} FEATURES...")
    
    # Advanced models
    models = {
        'XGBoost': xgb.XGBClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
    }
    
    # Stratified K-fold for reliable evaluation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {}
    for name, model in models.items():
        try:
            # Cross-validation
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            
            # Additional metrics
            from sklearn.model_selection import cross_validate
            scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring)
            
            results[name] = {
                'accuracy': scores.mean(),
                'accuracy_std': scores.std(),
                'precision': cv_results['test_precision'].mean(),
                'recall': cv_results['test_recall'].mean(),
                'f1': cv_results['test_f1'].mean(),
                'roc_auc': cv_results['test_roc_auc'].mean()
            }
            
            print(f"\n✅ {name}:")
            print(f"   Accuracy:  {scores.mean():.4f} ± {scores.std():.4f}")
            print(f"   Precision: {cv_results['test_precision'].mean():.4f}")
            print(f"   Recall:    {cv_results['test_recall'].mean():.4f}")
            print(f"   F1-Score:  {cv_results['test_f1'].mean():.4f}")
            print(f"   ROC-AUC:   {cv_results['test_roc_auc'].mean():.4f}")
            
        except Exception as e:
            print(f"❌ {name} failed: {str(e)}")
    
    # Find best model
    if results:
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\n🏆 BEST MODEL: {best_model[0]} with accuracy: {best_model[1]['accuracy']:.4f}")
        
        # Feature importance for best model
        if best_model[0] == 'XGBoost':
            model = xgb.XGBClassifier(random_state=42)
        else:
            model = best_model[1]
        
        model.fit(X, y)
        
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n📊 TOP 10 FEATURE IMPORTANCES:")
            print(feature_importance.head(10))

if __name__ == "__main__":
    run_advanced_analysis()