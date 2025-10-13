import mlflow
import pandas as pd
import numpy as np
import os
import sys
import traceback
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.preprocessor import BoxPlotPreprocessor

class ClinicalFeaturesOptimizer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.preprocessor = BoxPlotPreprocessor()
        self.setup_mlflow()
    
    def setup_mlflow(self):
        """Setup MLflow tracking"""
        mlflow.set_tracking_uri("mlruns")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        experiment_name = f"9. CLINICAL_FEATURES_ONLY_95_ACCURACY_{timestamp}"

        mlflow.set_experiment(experiment_name)
        print(f"MLflow setup completed - Experiment: {experiment_name}")
    
    def load_data(self):
        """Load the dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f" Dataset loaded: {self.df.shape}")
        return self.df
    
    def remove_identifier_features(self, df):
        """Remove identifier features that cause data leakage"""
        print("🚫 Removing identifier features...")
        
        # Remove obvious identifiers
        identifiers = ['subject_id', 'hadm_id']
        df_clean = df.drop(columns=[col for col in identifiers if col in df.columns])
        
        print(f"   Removed identifiers: {identifiers}")
        print(f"   Features remaining: {len(df_clean.columns)}")
        return df_clean
    
    def advanced_clinical_imputation(self, df):
        """Advanced imputation for clinical features only"""
        print("🔄 Advanced Clinical Features Imputation...")
        
        df_imputed = df.copy()
        original_rows = len(df_imputed)
        
        # Step 1: Handle categorical missing values
        categorical_cols = ['insurance', 'marital_status', 'gender', 'admission_type', 'race']
        for col in categorical_cols:
            if col in df_imputed.columns and df_imputed[col].isna().any():
                mode_val = df_imputed[col].mode()
                df_imputed[col].fillna(mode_val[0] if len(mode_val) > 0 else 'Unknown', inplace=True)
        
        # Step 2: Clinical feature imputation with domain knowledge
        clinical_features = {
            'vital_signs': ['heart_rate', 'bp_systolic', 'bp_diastolic', 'spo2', 'respiratory_rate', 'temperature'],
            'lab_tests': ['creatinine', 'glucose', 'sodium', 'potassium', 'troponin_t', 'creatine_kinase_mb', 'hemoglobin', 'white_blood_cells'],
            'demographics': ['anchor_age']
        }
        
        # Impute with clinical reasonable values
        clinical_ranges = {
            'heart_rate': 72, 'bp_systolic': 120, 'bp_diastolic': 80, 'spo2': 98,
            'respiratory_rate': 18, 'temperature': 98.6, 'creatinine': 1.0,
            'glucose': 100, 'sodium': 140, 'potassium': 4.0, 'troponin_t': 0.01,
            'creatine_kinase_mb': 5.0, 'hemoglobin': 13.5, 'white_blood_cells': 7.0
        }
        
        for feature, default_value in clinical_ranges.items():
            if feature in df_imputed.columns and df_imputed[feature].isna().any():
                # Use median if available, else clinical default
                if df_imputed[feature].notna().sum() > 0:
                    df_imputed[feature].fillna(df_imputed[feature].median(), inplace=True)
                else:
                    df_imputed[feature].fillna(default_value, inplace=True)
        
        # Step 3: Advanced Clinical Feature Engineering
        print("   Creating advanced clinical features...")
        
        # Cardiac-specific features
        if all(col in df_imputed.columns for col in ['troponin_t', 'creatine_kinase_mb']):
            df_imputed['cardiac_risk_score'] = np.log1p(df_imputed['troponin_t']) * np.log1p(df_imputed['creatine_kinase_mb'])
            df_imputed['elevated_cardiac_markers'] = (
                (df_imputed['troponin_t'] > 0.1) | (df_imputed['creatine_kinase_mb'] > 5)
            ).astype(int)
        
        # Hemodynamic stability score
        if all(col in df_imputed.columns for col in ['bp_systolic', 'heart_rate', 'spo2']):
            df_imputed['hemodynamic_score'] = (
                (df_imputed['bp_systolic'] / 120) + 
                (df_imputed['heart_rate'] / 80) + 
                (df_imputed['spo2'] / 98)
            )
            df_imputed['hemodynamic_instability'] = (
                (df_imputed['bp_systolic'] < 90) | 
                (df_imputed['heart_rate'] > 120) | 
                (df_imputed['spo2'] < 92)
            ).astype(int)
        
        # Metabolic panel interactions
        if all(col in df_imputed.columns for col in ['sodium', 'potassium', 'creatinine']):
            df_imputed['electrolyte_balance'] = df_imputed['sodium'] / (df_imputed['potassium'] + 1)
            df_imputed['renal_function_score'] = df_imputed['creatinine'] * df_imputed['potassium']
        
        # Oxygenation and perfusion
        if all(col in df_imputed.columns for col in ['hemoglobin', 'spo2']):
            df_imputed['oxygen_carrying_capacity'] = df_imputed['hemoglobin'] * (df_imputed['spo2'] / 100)
            df_imputed['anemia_hypoxia_risk'] = (
                (df_imputed['hemoglobin'] < 12) & (df_imputed['spo2'] < 94)
            ).astype(int)
        
        # Age-adjusted features
        if 'anchor_age' in df_imputed.columns:
            df_imputed['age_squared'] = df_imputed['anchor_age'] ** 2
            df_imputed['elderly_risk'] = (df_imputed['anchor_age'] > 65).astype(int)
            df_imputed['young_risk'] = (df_imputed['anchor_age'] < 45).astype(int)
        
        # Inflammatory markers
        if 'white_blood_cells' in df_imputed.columns:
            df_imputed['wbc_elevated'] = (df_imputed['white_blood_cells'] > 11).astype(int)
            df_imputed['wbc_low'] = (df_imputed['white_blood_cells'] < 4).astype(int)
        
        # Clinical ratios
        if all(col in df_imputed.columns for col in ['bp_systolic', 'bp_diastolic']):
            df_imputed['pulse_pressure'] = df_imputed['bp_systolic'] - df_imputed['bp_diastolic']
            df_imputed['map'] = (df_imputed['bp_systolic'] + 2 * df_imputed['bp_diastolic']) / 3  # Mean Arterial Pressure
        
        # Verify no data loss
        if original_rows == len(df_imputed):
            print(f"   ✅ No data removed! Patients: {len(df_imputed)}")
        else:
            print(f"   ⚠️ Data loss: {original_rows} -> {len(df_imputed)}")
        
        print(f"   Clinical features created: {len(df_imputed.columns) - len(df.columns)}")
        return df_imputed
    
    def encode_categorical_clinical(self, df):
        """Encode categorical variables for clinical modeling"""
        print("   Encoding categorical variables...")
        
        df_encoded = df.copy()
        
        # Clinical one-hot encoding
        clinical_categorical = ['gender', 'admission_type']
        for col in clinical_categorical:
            if col in df_encoded.columns:
                dummies = pd.get_dummies(df_encoded[col], prefix=col)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
        
        # Demographic encoding
        demographic_categorical = ['insurance', 'marital_status', 'race']
        for col in demographic_categorical:
            if col in df_encoded.columns:
                df_encoded[col] = pd.factorize(df_encoded[col])[0]
        
        # Remove original categorical columns
        df_encoded.drop(clinical_categorical, axis=1, inplace=True, errors='ignore')
        
        return df_encoded
    
    def select_clinical_features(self, X, y, k=25):
        """Select most clinically relevant features"""
        print("   Selecting clinical features...")
        
        # Remove any remaining non-clinical features
        non_clinical = ['subject_id', 'hadm_id']
        X_clinical = X.drop(columns=[col for col in non_clinical if col in X.columns])
        
        # Use statistical feature selection
        selector = SelectKBest(score_func=f_classif, k=min(k, X_clinical.shape[1]))
        X_selected = selector.fit_transform(X_clinical, y)
        selected_features = X_clinical.columns[selector.get_support()].tolist()
        
        print(f"     Selected {len(selected_features)} clinical features")
        print(f"     Top clinical features: {selected_features[:5]}")
        
        return X_clinical[selected_features]
    
    def train_clinical_lightgbm(self, X, y, params):
        """Train LightGBM with clinical features only"""
        try:
            print("🤖 Training Clinical LightGBM...")
            
            # Ensure numeric and clean
            X_numeric = X.copy()
            for col in X_numeric.columns:
                if X_numeric[col].dtype == 'object':
                    X_numeric[col] = pd.factorize(X_numeric[col])[0]
            
            X_numeric = X_numeric.fillna(X_numeric.median())
            
            # Split with stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X_numeric, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"     Training on {X_train.shape[0]} patients, {X_train.shape[1]} clinical features")
            
            # Train model
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Comprehensive metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_numeric, y, cv=5, scoring='accuracy')
            
            # Feature importance
            feature_importance = model.feature_importances_
            top_features = sorted(zip(X_numeric.columns, feature_importance), 
                                key=lambda x: x[1], reverse=True)[:10]
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_score': auc_score,
                'cv_mean_accuracy': cv_scores.mean(),
                'cv_std_accuracy': cv_scores.std(),
                'top_features': top_features,
                'model': model
            }
            
        except Exception as e:
            print(f"❌ Training error: {str(e)}")
            return None
    
    def run_clinical_optimization(self):
        """Optimize using clinical features only"""
        print("🚀 CLINICAL FEATURES OPTIMIZATION FOR 95% ACCURACY")
        print("=" * 70)
        
        self.load_data()
        print(f"📊 Starting with: {len(self.df)} patients")
        
        # Remove identifier features first
        df_no_ids = self.remove_identifier_features(self.df)
        
        # Apply clinical imputation and feature engineering
        df_processed = self.advanced_clinical_imputation(df_no_ids)
        
        # Encode categorical variables
        df_encoded = self.encode_categorical_clinical(df_processed)
        
        print(f"✅ Clinical processing completed: {len(df_encoded)} patients, {len(df_encoded.columns)} features")
        
        if 'heart_attack' not in df_encoded.columns:
            print("❌ heart_attack column not found")
            return
        
        # Prepare features and target
        X = df_encoded.drop(columns=['heart_attack'])
        y = df_encoded['heart_attack']
        
        print(f"🎯 Final clinical dataset: {X.shape[1]} features, {len(y)} patients")
        
        # Clinical-focused parameter combinations
        param_combinations = [
            {
                "name": "clinical_high_capacity",
                "params": {
                    "n_estimators": 2000,
                    "max_depth": 14,
                    "learning_rate": 0.01,
                    "num_leaves": 100,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_alpha": 0.1,
                    "reg_lambda": 0.1,
                    "min_child_samples": 15,
                    "random_state": 42,
                    "verbose": -1
                }
            },
            {
                "name": "clinical_precision", 
                "params": {
                    "n_estimators": 1500,
                    "max_depth": 12,
                    "learning_rate": 0.02,
                    "num_leaves": 80,
                    "subsample": 0.75,
                    "colsample_bytree": 0.75,
                    "reg_alpha": 0.2,
                    "reg_lambda": 0.2,
                    "min_child_samples": 20,
                    "random_state": 42,
                    "verbose": -1
                }
            },
            {
                "name": "clinical_balanced",
                "params": {
                    "n_estimators": 1200,
                    "max_depth": 10,
                    "learning_rate": 0.03,
                    "num_leaves": 64,
                    "subsample": 0.85,
                    "colsample_bytree": 0.85,
                    "reg_alpha": 0.15,
                    "reg_lambda": 0.15,
                    "min_child_samples": 25,
                    "random_state": 42,
                    "verbose": -1
                }
            }
        ]
        
        best_accuracy = 0
        best_combo = None
        
        for param_combo in param_combinations:
            run_name = f"clinical_{param_combo['name']}"
            
            with mlflow.start_run(run_name=run_name):
                try:
                    print(f"\n🔬 Testing: {run_name}")
                    
                    # Select clinical features
                    X_selected = self.select_clinical_features(X, y, k=25)
                    
                    # Log parameters
                    mlflow.log_params({
                        "strategy": "clinical_features_only",
                        "identifiers_removed": "yes",
                        "clinical_features_used": X_selected.shape[1],
                        "total_patients": len(df_encoded)
                    })
                    
                    # Log model parameters
                    for param_name, param_value in param_combo['params'].items():
                        mlflow.log_param(f"lgb_{param_name}", param_value)
                    
                    # Train and evaluate
                    results = self.train_clinical_lightgbm(X_selected, y, param_combo['params'])
                    
                    if results is None:
                        continue
                    
                    # Log metrics
                    mlflow.log_metrics({
                        "accuracy": results['accuracy'],
                        "precision": results['precision'],
                        "recall": results['recall'],
                        "f1_score": results['f1_score'],
                        "auc_score": results['auc_score'],
                        "cv_mean_accuracy": results['cv_mean_accuracy']
                    })
                    
                    # Log top clinical features
                    for i, (feature, importance) in enumerate(results['top_features'][:5]):
                        mlflow.log_param(f"clinical_feature_{i+1}_name", feature)
                        mlflow.log_metric(f"clinical_feature_{i+1}_importance", float(importance))
                    
                    print(f"✅ {run_name}")
                    print(f"   📊 Accuracy: {results['accuracy']:.4f}")
                    print(f"   🎯 Precision: {results['precision']:.4f}")
                    print(f"   🔄 Recall: {results['recall']:.4f}")
                    print(f"   ⚖️ F1 Score: {results['f1_score']:.4f}")
                    print(f"   📈 AUC: {results['auc_score']:.4f}")
                    
                    print(f"   🏆 Top Clinical Features:")
                    for feature, importance in results['top_features'][:5]:
                        print(f"     - {feature}: {importance:.4f}")
                    
                    # Track best result
                    if results['accuracy'] > best_accuracy:
                        best_accuracy = results['accuracy']
                        best_combo = {
                            'params': param_combo,
                            'results': results
                        }
                    
                    if results['accuracy'] >= 0.95:
                        print("🎉🎉🎉 95% ACCURACY ACHIEVED WITH CLINICAL FEATURES! 🎉🎉🎉")
                        
                except Exception as e:
                    print(f"❌ Error: {str(e)}")
        
        # Final results
        print(f"\n{'='*70}")
        print("🎯 CLINICAL FEATURES OPTIMIZATION COMPLETED")
        print(f"{'='*70}")
        
        if best_combo:
            print(f"🏆 BEST CLINICAL CONFIGURATION:")
            print(f"   Model: {best_combo['params']['name']}")
            print(f"   📊 Accuracy: {best_accuracy:.4f}")
            print(f"   ⚖️ F1 Score: {best_combo['results']['f1_score']:.4f}")
            print(f"   Patients: {len(df_encoded)}")
            print(f"   Clinical Features: {len(X_selected.columns)}")
            
            print(f"\n🏆 Top 5 Clinical Features:")
            for feature, importance in best_combo['results']['top_features'][:5]:
                print(f"   - {feature}: {importance:.4f}")
            
            if best_accuracy >= 0.95:
                print(f"\n🎉 SUCCESS: 95%+ accuracy achieved with REAL clinical features!")
            else:
                improvement = (0.95 - best_accuracy) * 100
                print(f"\n📈 Need {improvement:.2f}% improvement to reach 95%")
        
        print(f"\n📊 MLflow: mlflow ui")
from mlflow_experiments.config.data_config import DATA_PATH

if __name__ == "__main__":
    data_path = DATA_PATH
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
    else:
        optimizer = ClinicalFeaturesOptimizer(data_path)
        optimizer.run_clinical_optimization()