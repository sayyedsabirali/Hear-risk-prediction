import mlflow
import pandas as pd
import numpy as np
import os
import sys
import traceback
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the DataPreprocessor
from preprocessor import DataPreprocessor

class XGBoostFeatureEngineeringOptimizer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.preprocessor = DataPreprocessor()
        self.newly_created_features = []  # Track new features
        self.original_features = []  # Track original features
        self.setup_mlflow()
    
    def setup_mlflow(self):
        """Setup MLflow tracking"""
        mlflow.set_tracking_uri("mlruns")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        experiment_name = f"5_XGBoost_FeatureEngineering_{timestamp}"

        mlflow.set_experiment(experiment_name)
        print(f"MLflow setup completed - Experiment: {experiment_name}")
    
    def load_data(self):
        """Load the dataset and drop identifier columns"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"✅ Dataset loaded: {self.df.shape}")
        
        # Store original features (before dropping IDs)
        self.original_features = [col for col in self.df.columns 
                                 if col not in ['subject_id', 'hadm_id', 'heart_attack']]
        
        # Drop identifier columns
        cols_to_drop = ['subject_id', 'hadm_id']
        existing_cols = [col for col in cols_to_drop if col in self.df.columns]
        if existing_cols:
            self.df.drop(existing_cols, axis=1, inplace=True)
            print(f"✅ Dropped columns: {existing_cols}")
            print(f"✅ Data shape after dropping IDs: {self.df.shape}")
        
        return self.df
    
    def advanced_medical_feature_engineering(self, df):
        """Advanced medical feature engineering based on clinical knowledge"""
        print("\n🛠️ Advanced Medical Feature Engineering...")
        
        df_eng = df.copy()
        features_before = set(df_eng.columns)
        
        # 1. CARDIAC-SPECIFIC FEATURES
        print("   Creating cardiac-specific features...")
        
        # Cardiac biomarker combinations
        if all(col in df_eng.columns for col in ['troponin_t', 'creatine_kinase_mb']):
            df_eng['cardiac_biomarker_product'] = np.log1p(df_eng['troponin_t']) * np.log1p(df_eng['creatine_kinase_mb'])
            df_eng['cardiac_risk_index'] = (df_eng['troponin_t'] * 100) + (df_eng['creatine_kinase_mb'] * 10)
            df_eng['elevated_troponin'] = (df_eng['troponin_t'] > 0.1).astype(int)
            df_eng['elevated_ckmb'] = (df_eng['creatine_kinase_mb'] > 5).astype(int)
            df_eng['both_biomarkers_elevated'] = ((df_eng['troponin_t'] > 0.1) & (df_eng['creatine_kinase_mb'] > 5)).astype(int)
        
        # 2. HEMODYNAMIC STABILITY FEATURES
        print("   Creating hemodynamic features...")
        
        if all(col in df_eng.columns for col in ['bp_systolic', 'bp_diastolic', 'heart_rate']):
            df_eng['mean_arterial_pressure'] = (df_eng['bp_systolic'] + 2 * df_eng['bp_diastolic']) / 3
            df_eng['pulse_pressure'] = df_eng['bp_systolic'] - df_eng['bp_diastolic']
            df_eng['rate_pressure_product'] = df_eng['heart_rate'] * df_eng['bp_systolic'] / 100
            df_eng['shock_index'] = df_eng['heart_rate'] / (df_eng['bp_systolic'] + 0.1)
            df_eng['hemodynamic_instability'] = (
                (df_eng['bp_systolic'] < 90) | 
                (df_eng['heart_rate'] > 120) | 
                (df_eng['mean_arterial_pressure'] < 65)
            ).astype(int)
        
        # 3. OXYGENATION AND RESPIRATORY FEATURES
        print("   Creating oxygenation features...")
        
        if all(col in df_eng.columns for col in ['spo2', 'respiratory_rate', 'hemoglobin']):
            df_eng['oxygen_saturation_ratio'] = df_eng['spo2'] / 100
            df_eng['oxygen_content'] = (df_eng['hemoglobin'] * 1.34 * df_eng['spo2'] / 100) + (0.003 * df_eng['spo2'])
            df_eng['respiratory_distress'] = (
                (df_eng['spo2'] < 92) | (df_eng['respiratory_rate'] > 24)
            ).astype(int)
            df_eng['ventilation_perfusion_ratio'] = df_eng['respiratory_rate'] / (df_eng['spo2'] + 0.1)
        
        # 4. METABOLIC AND RENAL FEATURES
        print("   Creating metabolic features...")
        
        if all(col in df_eng.columns for col in ['creatinine', 'potassium', 'sodium', 'glucose']):
            df_eng['renal_function_score'] = df_eng['creatinine'] * df_eng['potassium']
            df_eng['electrolyte_imbalance'] = (
                (df_eng['sodium'] < 135) | (df_eng['sodium'] > 145) |
                (df_eng['potassium'] < 3.5) | (df_eng['potassium'] > 5.0)
            ).astype(int)
            df_eng['hyperglycemia'] = (df_eng['glucose'] > 180).astype(int)
            df_eng['hypoglycemia'] = (df_eng['glucose'] < 70).astype(int)
            df_eng['bun_creatinine_ratio'] = df_eng['glucose'] / (df_eng['creatinine'] + 0.1)
        
        # 5. INFLAMMATORY AND HEMATOLOGICAL FEATURES
        print("   Creating inflammatory features...")
        
        if 'white_blood_cells' in df_eng.columns:
            df_eng['wbc_elevated'] = (df_eng['white_blood_cells'] > 11).astype(int)
            df_eng['wbc_low'] = (df_eng['white_blood_cells'] < 4).astype(int)
            df_eng['leukocytosis'] = (df_eng['white_blood_cells'] > 15).astype(int)
        
        if 'hemoglobin' in df_eng.columns:
            df_eng['anemia'] = (df_eng['hemoglobin'] < 12).astype(int)
            df_eng['severe_anemia'] = (df_eng['hemoglobin'] < 8).astype(int)
            df_eng['polycythemia'] = (df_eng['hemoglobin'] > 16).astype(int)
        
        # 6. AGE AND COMORBIDITY FEATURES
        print("   Creating age-comorbidity features...")
        
        if 'anchor_age' in df_eng.columns:
            df_eng['age_squared'] = df_eng['anchor_age'] ** 2
            df_eng['age_cubed'] = df_eng['anchor_age'] ** 3
            df_eng['elderly'] = (df_eng['anchor_age'] >= 65).astype(int)
            df_eng['very_elderly'] = (df_eng['anchor_age'] >= 80).astype(int)
            df_eng['young_adult'] = ((df_eng['anchor_age'] >= 18) & (df_eng['anchor_age'] <= 40)).astype(int)
            
            # Age-adjusted thresholds
            df_eng['age_adjusted_hr_max'] = 220 - df_eng['anchor_age']
            if 'heart_rate' in df_eng.columns:
                df_eng['hr_percentage_max'] = df_eng['heart_rate'] / (df_eng['age_adjusted_hr_max'] + 0.1)
        
        # 7. VITAL SIGN INTERACTIONS
        print("   Creating vital sign interactions...")
        
        vital_cols = ['heart_rate', 'respiratory_rate', 'spo2', 'temperature']
        available_vitals = [col for col in vital_cols if col in df_eng.columns]
        
        if len(available_vitals) >= 2:
            df_eng['vital_sign_mean'] = df_eng[available_vitals].mean(axis=1)
            df_eng['vital_sign_std'] = df_eng[available_vitals].std(axis=1)
            df_eng['vital_sign_range'] = df_eng[available_vitals].max(axis=1) - df_eng[available_vitals].min(axis=1)
            
            # Early Warning Score-like features
            df_eng['ews_score'] = 0
            if 'heart_rate' in df_eng.columns:
                df_eng['ews_score'] += ((df_eng['heart_rate'] < 50) | (df_eng['heart_rate'] > 100)).astype(int)
            if 'respiratory_rate' in df_eng.columns:
                df_eng['ews_score'] += ((df_eng['respiratory_rate'] < 12) | (df_eng['respiratory_rate'] > 20)).astype(int)
            if 'spo2' in df_eng.columns:
                df_eng['ews_score'] += (df_eng['spo2'] < 95).astype(int)
            if 'temperature' in df_eng.columns:
                df_eng['ews_score'] += ((df_eng['temperature'] < 36) | (df_eng['temperature'] > 38)).astype(int)
        
        # 8. CLINICAL RISK SCORES
        print("   Creating clinical risk scores...")
        
        # Simplified TIMI-like score
        df_eng['timi_like_score'] = 0
        if 'anchor_age' in df_eng.columns:
            df_eng['timi_like_score'] += (df_eng['anchor_age'] > 65).astype(int)
        if all(col in df_eng.columns for col in ['troponin_t', 'creatine_kinase_mb']):
            df_eng['timi_like_score'] += ((df_eng['troponin_t'] > 0.1) | (df_eng['creatine_kinase_mb'] > 5)).astype(int)
        if 'heart_rate' in df_eng.columns:
            df_eng['timi_like_score'] += (df_eng['heart_rate'] > 100).astype(int)
        if 'bp_systolic' in df_eng.columns:
            df_eng['timi_like_score'] += (df_eng['bp_systolic'] < 100).astype(int)
        
        # 9. POLYNOMIAL AND INTERACTION FEATURES
        print("   Creating polynomial features...")
        
        key_continuous = ['troponin_t', 'creatine_kinase_mb', 'hemoglobin', 'heart_rate', 'anchor_age']
        for col in key_continuous:
            if col in df_eng.columns:
                df_eng[f'{col}_squared'] = df_eng[col] ** 2
                df_eng[f'{col}_cubed'] = df_eng[col] ** 3
                df_eng[f'{col}_log'] = np.log1p(df_eng[col])
                df_eng[f'{col}_sqrt'] = np.sqrt(df_eng[col] + 0.1)
        
        # 10. RATIO FEATURES
        print("   Creating ratio features...")
        
        if all(col in df_eng.columns for col in ['hemoglobin', 'white_blood_cells']):
            df_eng['hgb_wbc_ratio'] = df_eng['hemoglobin'] / (df_eng['white_blood_cells'] + 0.1)
        
        if all(col in df_eng.columns for col in ['heart_rate', 'respiratory_rate']):
            df_eng['hr_rr_ratio'] = df_eng['heart_rate'] / (df_eng['respiratory_rate'] + 0.1)
        
        if all(col in df_eng.columns for col in ['spo2', 'respiratory_rate']):
            df_eng['spo2_rr_ratio'] = df_eng['spo2'] / (df_eng['respiratory_rate'] + 0.1)
        
        # Track newly created features
        features_after = set(df_eng.columns)
        self.newly_created_features = list(features_after - features_before - {'heart_attack'})
        
        print(f"✅ Advanced feature engineering completed: {len(self.newly_created_features)} new features")
        print(f"   Total features: {len(df_eng.columns)}")
        
        return df_eng
    
    def custom_preprocessing(self, df):
        """
        Custom preprocessing matching the tuned XGBoost parameters:
        - missing_categorical: 'unknown'
        - missing_numerical: 'median'
        - outlier_method: 'cap'
        - outlier_threshold: 3.0
        - scaling_method: 'none'
        - skewness_method: 'log'
        """
        print("\n🔄 Custom Preprocessing (Matching Tuned XGBoost)...")
        
        df_processed = df.copy()
        
        # Identify column types
        self.preprocessor.identify_columns(df_processed)
        
        # 1. Handle missing values
        print("   Handling missing values...")
        # Categorical: fill with 'unknown'
        for col in self.preprocessor.categorical_cols:
            if col in df_processed.columns and df_processed[col].isnull().any():
                df_processed[col].fillna('unknown', inplace=True)
        
        # Numerical: fill with median
        for col in self.preprocessor.numerical_cols:
            if col in df_processed.columns and df_processed[col].isnull().any():
                median_val = df_processed[col].median()
                df_processed[col].fillna(median_val, inplace=True)
        
        # 2. Handle outliers using cap method with threshold 3.0
        print("   Handling outliers (cap method, threshold=3.0)...")
        for col in self.preprocessor.numerical_cols:
            if col in df_processed.columns and col != 'heart_attack':
                Q1 = df_processed[col].quantile(0.25)
                Q3 = df_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3.0 * IQR
                upper_bound = Q3 + 3.0 * IQR
                
                df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
        
        # 3. Reduce skewness using log transformation
        print("   Reducing skewness (log method)...")
        for col in self.preprocessor.numerical_cols:
            if col in df_processed.columns and col != 'heart_attack':
                skewness = df_processed[col].skew()
                if abs(skewness) > 0.5:
                    # Apply log transformation (add 1 to handle zeros)
                    min_val = df_processed[col].min()
                    if min_val <= 0:
                        df_processed[col] = np.log1p(df_processed[col] - min_val + 1)
                    else:
                        df_processed[col] = np.log1p(df_processed[col])
        
        # 4. NO SCALING (scaling_method: 'none')
        print("   Scaling: None (as per tuned parameters)")
        
        # 5. Encode categorical variables
        print("   Encoding categorical variables...")
        df_processed = self.preprocessor.encode_categorical(df_processed)
        
        print(f"✅ Preprocessing completed. Shape: {df_processed.shape}")
        return df_processed
    
    def save_confusion_matrix(self, y_true, y_pred, run_id):
        """Generate and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Heart Attack', 'Heart Attack'],
                    yticklabels=['No Heart Attack', 'Heart Attack'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        cm_path = f"confusion_matrix_{run_id}.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return cm_path, cm
    
    def save_feature_importance(self, model, feature_names, run_id):
        """Generate and save feature importance plot"""
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Plot top 20 features
        plt.figure(figsize=(10, 8))
        top_20 = importance_df.head(20)
        plt.barh(range(len(top_20)), top_20['importance'])
        plt.yticks(range(len(top_20)), top_20['feature'])
        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importance')
        plt.gca().invert_yaxis()
        
        # Save plot
        fi_path = f"feature_importance_{run_id}.png"
        plt.savefig(fi_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return fi_path, importance_df
    
    def train_tuned_xgboost(self, X, y):
        """
        Train XGBoost with exact tuned parameters from your document
        """
        print("\n🤖 Training Tuned XGBoost...")
        
        # Ensure data is clean
        X_clean = X.copy()
        for col in X_clean.columns:
            if X_clean[col].dtype == 'object':
                X_clean[col] = pd.factorize(X_clean[col])[0]
        X_clean = X_clean.fillna(X_clean.median())
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   Training on {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"   Testing on {X_test.shape[0]} samples")
        
        # Tuned XGBoost parameters
        xgb_params = {
            'n_estimators': 400,
            'max_depth': 10,
            'learning_rate': 0.1,
            'colsample_bytree': 0.7,
            'subsample': 0.8,
            'gamma': 0.5,
            'reg_alpha': 0.1,
            'reg_lambda': 0.5,
            'min_child_weight': 5,
            'random_state': 42,
            'eval_metric': 'logloss',
            'use_label_encoder': False
        }
        
        # Train model
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_train, y_train, 
                 eval_set=[(X_test, y_test)],
                 verbose=False)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation
        print("   Performing cross-validation...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_clean, y, cv=cv, scoring='accuracy')
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_clean.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc_score,
            'cv_mean_accuracy': cv_scores.mean(),
            'cv_std_accuracy': cv_scores.std(),
            'feature_importance': feature_importance,
            'model': model,
            'y_test': y_test,
            'y_pred': y_pred,
            'X_test': X_test
        }
        
        return results
    
    def run_optimization(self):
        """Run the complete optimization pipeline"""
        print("\n" + "="*80)
        print("🚀 XGBOOST WITH ADVANCED FEATURE ENGINEERING")
        print("="*80)
        
        # Load data
        self.load_data()
        print(f"\n📊 Starting with: {len(self.df)} patients, {len(self.df.columns)} features")
        
        # Advanced feature engineering
        df_engineered = self.advanced_medical_feature_engineering(self.df)
        
        # Custom preprocessing (matching tuned XGBoost parameters)
        df_processed = self.custom_preprocessing(df_engineered)
        
        print(f"\n✅ Processing completed:")
        print(f"   Patients: {len(df_processed)}")
        print(f"   Total features: {len(df_processed.columns)}")
        print(f"   Original features: {len(self.original_features)}")
        print(f"   Newly created features: {len(self.newly_created_features)}")
        
        if 'heart_attack' not in df_processed.columns:
            print("❌ heart_attack column not found")
            return
        
        # Prepare data
        X = df_processed.drop(columns=['heart_attack'])
        y = df_processed['heart_attack']
        
        print(f"\n🎯 Final dataset: {X.shape[1]} features, {len(y)} patients")
        print(f"   Class distribution: {y.value_counts().to_dict()}")
        
        # Start MLflow run
        with mlflow.start_run(run_name="XGBoost_with_Feature_Engineering"):
            try:
                # Log configuration
                mlflow.log_params({
                    "model_type": "XGBoost",
                    "n_estimators": 400,
                    "max_depth": 10,
                    "learning_rate": 0.1,
                    "colsample_bytree": 0.7,
                    "subsample": 0.8,
                    "gamma": 0.5,
                    "reg_alpha": 0.1,
                    "reg_lambda": 0.5,
                    "min_child_weight": 5,
                    "preprocessing_missing_categorical": "unknown",
                    "preprocessing_missing_numerical": "median",
                    "preprocessing_outlier_method": "cap",
                    "preprocessing_outlier_threshold": 3.0,
                    "preprocessing_scaling_method": "none",
                    "preprocessing_skewness_method": "log",
                    "total_features": X.shape[1],
                    "original_features": len(self.original_features),
                    "newly_created_features": len(self.newly_created_features),
                    "total_patients": len(df_processed),
                    "feature_engineering": "advanced_medical"
                })
                
                # Train model
                results = self.train_tuned_xgboost(X, y)
                
                if results is None:
                    print("❌ Training failed")
                    return
                
                # Get run ID for file naming
                run_id = mlflow.active_run().info.run_id
                
                # Log metrics
                mlflow.log_metrics({
                    "accuracy": results['accuracy'],
                    "precision": results['precision'],
                    "recall": results['recall'],
                    "f1_score": results['f1_score'],
                    "auc_score": results['auc_score'],
                    "cv_mean_accuracy": results['cv_mean_accuracy'],
                    "cv_std_accuracy": results['cv_std_accuracy']
                })
                
                # Save and log confusion matrix
                print("\n📊 Saving confusion matrix...")
                cm_path, cm = self.save_confusion_matrix(
                    results['y_test'], 
                    results['y_pred'], 
                    run_id
                )
                mlflow.log_artifact(cm_path)
                
                # Log confusion matrix values
                mlflow.log_metrics({
                    "cm_true_negative": int(cm[0][0]),
                    "cm_false_positive": int(cm[0][1]),
                    "cm_false_negative": int(cm[1][0]),
                    "cm_true_positive": int(cm[1][1])
                })
                
                # Save and log feature importance
                print("📊 Saving feature importance...")
                fi_path, importance_df = self.save_feature_importance(
                    results['model'],
                    X.columns,
                    run_id
                )
                mlflow.log_artifact(fi_path)
                
                # Save feature importance CSV
                fi_csv_path = f"feature_importance_{run_id}.csv"
                importance_df.to_csv(fi_csv_path, index=False)
                mlflow.log_artifact(fi_csv_path)
                
                # Log top 10 features
                top_10 = importance_df.head(10)
                for i, row in top_10.iterrows():
                    mlflow.log_param(f"top_feature_{i+1}_name", row['feature'])
                    mlflow.log_metric(f"top_feature_{i+1}_importance", float(row['importance']))
                
                # Save newly created features list
                print("Saving newly created features list...")
                new_features_path = f"newly_created_features_{run_id}.json"
                with open(new_features_path, 'w') as f:
                    json.dump({
                        'newly_created_features': self.newly_created_features,
                        'count': len(self.newly_created_features),
                        'original_features': self.original_features,
                        'original_count': len(self.original_features)
                    }, f, indent=2)
                mlflow.log_artifact(new_features_path)
                
                # Save detailed features breakdown
                features_breakdown_path = f"features_breakdown_{run_id}.txt"
                with open(features_breakdown_path, 'w') as f:
                    f.write("FEATURES BREAKDOWN\n")
                    f.write("="*80 + "\n\n")
                    f.write(f"Original Features ({len(self.original_features)}):\n")
                    f.write("-"*80 + "\n")
                    for feat in sorted(self.original_features):
                        f.write(f"  - {feat}\n")
                    f.write(f"\n\nNewly Created Features ({len(self.newly_created_features)}):\n")
                    f.write("-"*80 + "\n")
                    for feat in sorted(self.newly_created_features):
                        f.write(f"  - {feat}\n")
                mlflow.log_artifact(features_breakdown_path)
                
                # Print results
                print("\n" + "="*80)
                print("📊 RESULTS")
                print("="*80)
                print(f"\n🎯 Performance Metrics:")
                print(f"   Accuracy:  {results['accuracy']:.4f}")
                print(f"   Precision: {results['precision']:.4f}")
                print(f"   Recall:    {results['recall']:.4f}")
                print(f"   F1-Score:  {results['f1_score']:.4f}")
                print(f"   AUC:       {results['auc_score']:.4f}")
                print(f"   CV Accuracy: {results['cv_mean_accuracy']:.4f} ± {results['cv_std_accuracy']:.4f}")
                
                print(f"\n Top 10 Most Important Features:")
                for i, row in top_10.iterrows():
                    is_new = "" if row['feature'] in self.newly_created_features else "   "
                    print(f"   {is_new} {row['feature']}: {row['importance']:.4f}")
                
                print(f"\n Feature Engineering Impact:")
                print(f"   Original features: {len(self.original_features)}")
                print(f"   New features created: {len(self.newly_created_features)}")
                print(f"   Total features: {X.shape[1]}")
                
                # Count how many new features are in top 10
                new_in_top_10 = sum(1 for _, row in top_10.iterrows() 
                                   if row['feature'] in self.newly_created_features)
                print(f"   New features in top 10: {new_in_top_10}/10")
                
                print(f"\nArtifacts saved:")
                print(f"   ✓ Confusion Matrix: {cm_path}")
                print(f"   ✓ Feature Importance Plot: {fi_path}")
                print(f"   ✓ Feature Importance CSV: {fi_csv_path}")
                print(f"   ✓ Newly Created Features: {new_features_path}")
                print(f"   ✓ Features Breakdown: {features_breakdown_path}")
                
                # Comparison with baseline
                print(f"\nComparison with Baseline (without feature engineering):")
                print(f"   Baseline Accuracy: 0.9412")
                print(f"   Current Accuracy:  {results['accuracy']:.4f}")
                improvement = (results['accuracy'] - 0.9412) * 100
                if improvement > 0:
                    print(f"   Improvement: +{improvement:.2f}% 🎉")
                else:
                    print(f"   Change: {improvement:.2f}%")
                
                print("\n" + "="*80)
                print("✅ OPTIMIZATION COMPLETED SUCCESSFULLY!")
                print("="*80)
                print(f"\nMLflow UI: mlflow ui --port 5000")
                print(f"Run ID: {run_id}")
                
                # Clean up temporary files
                import os
                for file in [cm_path, fi_path, fi_csv_path, new_features_path, features_breakdown_path]:
                    if os.path.exists(file):
                        try:
                            os.remove(file)
                        except:
                            pass
                
            except Exception as e:
                print(f"\n❌ Error during optimization: {str(e)}")
                traceback.print_exc()


if __name__ == "__main__":
    data_path = "F:\\18. MAJOR PROJECT\\Heart-related-content\\heart_risk_complete_dataset.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print(f"Please update the data_path variable with the correct path to your dataset.")
    else:
        optimizer = XGBoostFeatureEngineeringOptimizer(data_path)
        optimizer.run_optimization()