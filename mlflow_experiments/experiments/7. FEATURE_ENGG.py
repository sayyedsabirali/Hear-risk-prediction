import mlflow
import pandas as pd
import numpy as np
import os
import sys
import traceback
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.preprocessor import BoxPlotPreprocessor

class NoDataRemovalOptimizer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.preprocessor = BoxPlotPreprocessor()
        self.setup_mlflow()
    
    def setup_mlflow(self):
        """Setup MLflow tracking"""
        mlflow.set_tracking_uri("mlruns")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        experiment_name = f"8. NO_DATA_REMOVAL_95_ACCURACY_{timestamp}"

        mlflow.set_experiment(experiment_name)
        print(f"MLflow setup completed - Experiment: {experiment_name}")
    
    def load_data(self):
        """Load the dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f" Dataset loaded: {self.df.shape}")
        return self.df
    
    def advanced_imputation_only(self, df):
        """Apply advanced imputation WITHOUT removing any data"""
        print("🔄 Applying Advanced Imputation (NO DATA REMOVAL)...")
        
        df_imputed = df.copy()
        original_rows = len(df_imputed)
        
        # Step 1: Handle categorical missing values
        print("   Handling categorical missing values...")
        categorical_cols = ['insurance', 'marital_status', 'gender', 'admission_type', 'race']
        
        for col in categorical_cols:
            if col in df_imputed.columns and df_imputed[col].isna().any():
                mode_val = df_imputed[col].mode()
                if len(mode_val) > 0:
                    df_imputed[col].fillna(mode_val[0], inplace=True)
                else:
                    df_imputed[col].fillna('Unknown', inplace=True)
        
        # Step 2: Handle numerical missing values with advanced imputation
        print("   Handling numerical missing values with MICE imputation...")
        numerical_cols = [
            'creatinine', 'glucose', 'sodium', 'potassium', 'troponin_t', 
            'creatine_kinase_mb', 'hemoglobin', 'white_blood_cells', 
            'heart_rate', 'bp_systolic', 'bp_diastolic', 'spo2', 
            'respiratory_rate', 'temperature'
        ]
        
        # First fill with median for quick imputation
        for col in numerical_cols:
            if col in df_imputed.columns and df_imputed[col].isna().any():
                median_val = df_imputed[col].median()
                df_imputed[col].fillna(median_val, inplace=True)
        
        # Step 3: Use MICE (Multiple Imputation by Chained Equations) for better imputation
        try:
            numerical_data = df_imputed[numerical_cols]
            imputer = IterativeImputer(
                max_iter=10, 
                random_state=42,
                skip_complete=True
            )
            imputed_data = imputer.fit_transform(numerical_data)
            df_imputed[numerical_cols] = imputed_data
            print("   ✅ MICE imputation completed successfully")
        except Exception as e:
            print(f"   ⚠️ MICE imputation failed, using median: {e}")
            # Fallback to median imputation
            for col in numerical_cols:
                if col in df_imputed.columns and df_imputed[col].isna().any():
                    median_val = df_imputed[col].median()
                    df_imputed[col].fillna(median_val, inplace=True)
        
        # Step 4: Smart Feature Engineering (NO DATA LOSS)
        print("   Applying smart feature engineering...")
        
        # Clinical ratios and interactions
        if all(col in df_imputed.columns for col in ['hemoglobin', 'spo2']):
            df_imputed['hgb_spo2_ratio'] = df_imputed['hemoglobin'] / (df_imputed['spo2'] + 0.1)
            df_imputed['hgb_spo2_product'] = df_imputed['hemoglobin'] * df_imputed['spo2']
        
        if all(col in df_imputed.columns for col in ['bp_systolic', 'bp_diastolic']):
            df_imputed['map'] = (df_imputed['bp_systolic'] + 2 * df_imputed['bp_diastolic']) / 3  # Mean Arterial Pressure
            df_imputed['pulse_pressure'] = df_imputed['bp_systolic'] - df_imputed['bp_diastolic']
        
        # Vital signs combinations
        if all(col in df_imputed.columns for col in ['heart_rate', 'respiratory_rate', 'spo2']):
            df_imputed['vital_score'] = (
                (df_imputed['heart_rate'] / 100) + 
                (df_imputed['respiratory_rate'] / 30) + 
                ((100 - df_imputed['spo2']) / 10)
            )
        
        # Lab value interactions
        if all(col in df_imputed.columns for col in ['creatinine', 'potassium']):
            df_imputed['renal_risk'] = df_imputed['creatinine'] * df_imputed['potassium']
        
        if all(col in df_imputed.columns for col in ['troponin_t', 'creatine_kinase_mb']):
            df_imputed['cardiac_risk'] = df_imputed['troponin_t'] * df_imputed['creatine_kinase_mb']
        
        # Age-based features
        if 'anchor_age' in df_imputed.columns:
            df_imputed['age_decade'] = (df_imputed['anchor_age'] // 10) * 10
            df_imputed['is_elderly'] = (df_imputed['anchor_age'] > 65).astype(int)
        
        # Polynomial features for key variables
        key_features = ['hemoglobin', 'troponin_t', 'creatine_kinase_mb', 'spo2']
        for col in key_features:
            if col in df_imputed.columns:
                df_imputed[f'{col}_squared'] = df_imputed[col] ** 2
                df_imputed[f'{col}_log'] = np.log1p(df_imputed[col])
        
        # Verify no data loss
        final_rows = len(df_imputed)
        if original_rows == final_rows:
            print(f"   ✅ SUCCESS: No data removed! Original: {original_rows}, Final: {final_rows}")
        else:
            print(f"   ⚠️ Data loss detected: Original: {original_rows}, Final: {final_rows}")
        
        print(f"   Features before: {len(df.columns)}, Features after: {len(df_imputed.columns)}")
        return df_imputed
        
    def encode_categorical_smart(self, df):
        """Smart encoding without data loss"""
        print("   Encoding categorical variables...")
        
        df_encoded = df.copy()
        
        # One-hot encoding for low cardinality features
        low_cardinality = ['gender', 'admission_type']
        for col in low_cardinality:
            if col in df_encoded.columns:
                dummies = pd.get_dummies(df_encoded[col], prefix=col)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                df_encoded.drop(col, axis=1, inplace=True)
        
        # Label encoding for high cardinality features
        high_cardinality = ['insurance', 'marital_status', 'race']
        for col in high_cardinality:
            if col in df_encoded.columns:
                df_encoded[col] = pd.factorize(df_encoded[col])[0]
        
        return df_encoded
    
    def safe_feature_selection(self, X, y, method='lgb', k=30):
        """Safe feature selection without data loss"""
        print(f"   Applying {method} feature selection...")
        
        X_selected = X.copy()
        
        if method == 'lgb':
            # LightGBM based feature importance
            lgb_selector = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
            lgb_selector.fit(X_selected, y)
            importance = pd.Series(lgb_selector.feature_importances_, index=X_selected.columns)
            top_features = importance.nlargest(min(k, len(X_selected.columns))).index.tolist()
            print(f"     Selected {len(top_features)} features using LightGBM")
            return X_selected[top_features]
        
        elif method == 'correlation':
            # Correlation based selection
            correlations = X_selected.corrwith(y).abs()
            top_features = correlations.nlargest(min(k, len(X_selected.columns))).index.tolist()
            print(f"     Selected {len(top_features)} features using correlation")
            return X_selected[top_features]
        
        else:
            print(f"     No feature selection applied, using all {len(X_selected.columns)} features")
            return X_selected
    
    def train_lightgbm_advanced(self, X, y, params):
        """Train LightGBM with advanced techniques"""
        try:
            print("🤖 Training Advanced LightGBM...")
            
            # Ensure all data is numeric
            X_numeric = X.copy()
            for col in X_numeric.columns:
                if X_numeric[col].dtype == 'object':
                    X_numeric[col] = pd.factorize(X_numeric[col])[0]
            
            # Handle any remaining NaN
            X_numeric = X_numeric.fillna(X_numeric.median())
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_numeric, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"     Training on {X_train.shape[0]} samples, {X_train.shape[1]} features")
            
            # Train model
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train)
            
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
                'model': model,
                'feature_names': X_numeric.columns.tolist()
            }
            
        except Exception as e:
            print(f"❌ Training error: {str(e)}")
            traceback.print_exc()
            return None
    
    def run_no_data_removal_optimization(self):
        """Main optimization with NO DATA REMOVAL"""
        print("🚀 ADVANCED OPTIMIZATION WITH NO DATA REMOVAL")
        print("=" * 70)
        
        self.load_data()
        
        print(f"📊 Starting with: {len(self.df)} patients")
        
        # Apply advanced imputation WITHOUT data removal
        df_processed = self.advanced_imputation_only(self.df)
        
        # Encode categorical variables
        df_encoded = self.encode_categorical_smart(df_processed)
        
        print(f"✅ Processing completed: {len(df_encoded)} patients, {len(df_encoded.columns)} features")
        
        if 'heart_attack' not in df_encoded.columns:
            print("❌ heart_attack column not found")
            return
        
        # Prepare features and target
        X = df_encoded.drop(columns=['heart_attack'])
        y = df_encoded['heart_attack']
        
        print(f"🎯 Final dataset: {X.shape[1]} features, {len(y)} patients")
        print(f"📈 Target distribution: {y.value_counts().to_dict()}")
        
        # Optimized LightGBM parameters
        param_combinations = [
            {
                "name": "high_capacity",
                "params": {
                    "n_estimators": 2000,
                    "max_depth": 16,
                    "learning_rate": 0.01,
                    "num_leaves": 128,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_alpha": 0.1,
                    "reg_lambda": 0.1,
                    "min_child_samples": 20,
                    "random_state": 42,
                    "verbose": -1
                }
            },
            {
                "name": "balanced", 
                "params": {
                    "n_estimators": 1500,
                    "max_depth": 12,
                    "learning_rate": 0.02,
                    "num_leaves": 96,
                    "subsample": 0.85,
                    "colsample_bytree": 0.85,
                    "reg_alpha": 0.15,
                    "reg_lambda": 0.15,
                    "min_child_samples": 15,
                    "random_state": 42,
                    "verbose": -1
                }
            },
            {
                "name": "fast_convergence",
                "params": {
                    "n_estimators": 1000,
                    "max_depth": 10,
                    "learning_rate": 0.05,
                    "num_leaves": 64,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                    "reg_alpha": 0.2,
                    "reg_lambda": 0.2,
                    "min_child_samples": 10,
                    "random_state": 42,
                    "verbose": -1
                }
            }
        ]
        
        # Feature selection methods
        fs_methods = [
            {'name': 'lgb', 'k': 35},
            {'name': 'correlation', 'k': 30},
            {'name': 'none', 'k': None}
        ]
        
        best_accuracy = 0
        best_combo = None
        successful_runs = 0
        
        for fs_method in fs_methods:
            print(f"\n🔍 Feature Selection: {fs_method['name']}")
            
            X_selected = self.safe_feature_selection(X, y, method=fs_method['name'], k=fs_method.get('k'))
            
            for param_combo in param_combinations:
                run_name = f"no_remove_{fs_method['name']}_{param_combo['name']}"
                
                with mlflow.start_run(run_name=run_name):
                    try:
                        print(f"\n🔬 Testing: {run_name}")
                        
                        # Log parameters
                        mlflow.log_params({
                            "data_strategy": "no_removal",
                            "imputation": "advanced_mice",
                            "feature_selection": fs_method['name'],
                            "features_used": X_selected.shape[1],
                            "total_patients": len(df_encoded),
                            "original_patients": len(self.df)
                        })
                        
                        # Log model parameters
                        for param_name, param_value in param_combo['params'].items():
                            mlflow.log_param(f"lgb_{param_name}", param_value)
                        
                        # Train and evaluate
                        results = self.train_lightgbm_advanced(X_selected, y, param_combo['params'])
                        
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
                        
                        # Log top features
                        for i, (feature, importance) in enumerate(results['top_features'][:5]):
                            mlflow.log_param(f"top_feature_{i+1}_name", feature)
                            mlflow.log_metric(f"top_feature_{i+1}_importance", float(importance))
                        
                        print(f"✅ {run_name}")
                        print(f"   📊 Accuracy: {results['accuracy']:.4f}")
                        print(f"   🎯 Precision: {results['precision']:.4f}")
                        print(f"   🔄 Recall: {results['recall']:.4f}")
                        print(f"   ⚖️ F1 Score: {results['f1_score']:.4f}")
                        print(f"   📈 AUC: {results['auc_score']:.4f}")
                        
                        print(f"   🏆 Top 3 Features:")
                        for feature, importance in results['top_features'][:3]:
                            print(f"     - {feature}: {importance:.4f}")
                        
                        # Track best result
                        if results['accuracy'] > best_accuracy:
                            best_accuracy = results['accuracy']
                            best_combo = {
                                'fs_method': fs_method,
                                'params': param_combo,
                                'results': results
                            }
                        
                        if results['accuracy'] >= 0.95:
                            successful_runs += 1
                            print("🎉🎉🎉 95% ACCURACY ACHIEVED! 🎉🎉🎉")
                            
                    except Exception as e:
                        print(f"❌ Error: {str(e)}")
        
        # Final results
        print(f"\n{'='*70}")
        print("🎯 OPTIMIZATION COMPLETED - NO DATA REMOVAL")
        print(f"{'='*70}")
        
        if best_combo:
            print(f"🏆 BEST CONFIGURATION:")
            print(f"   Feature Selection: {best_combo['fs_method']['name']}")
            print(f"   Model: {best_combo['params']['name']}")
            print(f"   📊 Accuracy: {best_accuracy:.4f}")
            print(f"   ⚖️ F1 Score: {best_combo['results']['f1_score']:.4f}")
            print(f"   Patients Used: {len(df_encoded)}/{len(self.df)} (100% data preserved)")
            
            print(f"\n🏆 Top 5 Features:")
            for feature, importance in best_combo['results']['top_features'][:5]:
                print(f"   - {feature}: {importance:.4f}")
            
            if successful_runs > 0:
                print(f"\n🎉 SUCCESS: {successful_runs} runs achieved 95%+ accuracy!")
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
        optimizer = NoDataRemovalOptimizer(data_path)
        optimizer.run_no_data_removal_optimization()