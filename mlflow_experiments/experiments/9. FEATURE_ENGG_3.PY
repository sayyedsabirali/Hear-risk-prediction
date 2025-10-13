import mlflow
import pandas as pd
import numpy as np
import os
import sys
import traceback
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.preprocessor import BoxPlotPreprocessor

class AdvancedEnsembleOptimizer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.preprocessor = BoxPlotPreprocessor()
        self.setup_mlflow()
    
    def setup_mlflow(self):
        """Setup MLflow tracking"""
        mlflow.set_tracking_uri("mlruns")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        experiment_name = f"10. ADVANCED_ENSEMBLE_95_ACCURACY_{timestamp}"

        mlflow.set_experiment(experiment_name)
        print(f"MLflow setup completed - Experiment: {experiment_name}")
    
    def load_data(self):
        """Load the dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f" Dataset loaded: {self.df.shape}")
        return self.df
    
    def advanced_medical_feature_engineering(self, df):
        """Advanced medical feature engineering based on clinical knowledge"""
        print("🛠️ Advanced Medical Feature Engineering...")
        
        df_eng = df.copy()
        original_features = len(df_eng.columns)
        
        # Remove identifiers but keep for temporary calculations
        temp_df = df_eng.copy()
        if 'subject_id' in temp_df.columns:
            temp_df.drop('subject_id', axis=1, inplace=True)
        if 'hadm_id' in temp_df.columns:
            temp_df.drop('hadm_id', axis=1, inplace=True)
        
        # 1. CARDIAC-SPECIFIC FEATURES
        print("   Creating cardiac-specific features...")
        
        # Cardiac biomarker combinations
        if all(col in temp_df.columns for col in ['troponin_t', 'creatine_kinase_mb']):
            temp_df['cardiac_biomarker_product'] = np.log1p(temp_df['troponin_t']) * np.log1p(temp_df['creatine_kinase_mb'])
            temp_df['cardiac_risk_index'] = (temp_df['troponin_t'] * 100) + (temp_df['creatine_kinase_mb'] * 10)
            temp_df['elevated_troponin'] = (temp_df['troponin_t'] > 0.1).astype(int)
            temp_df['elevated_ckmb'] = (temp_df['creatine_kinase_mb'] > 5).astype(int)
            temp_df['both_biomarkers_elevated'] = ((temp_df['troponin_t'] > 0.1) & (temp_df['creatine_kinase_mb'] > 5)).astype(int)
        
        # 2. HEMODYNAMIC STABILITY FEATURES
        print("   Creating hemodynamic features...")
        
        if all(col in temp_df.columns for col in ['bp_systolic', 'bp_diastolic', 'heart_rate']):
            temp_df['mean_arterial_pressure'] = (temp_df['bp_systolic'] + 2 * temp_df['bp_diastolic']) / 3
            temp_df['pulse_pressure'] = temp_df['bp_systolic'] - temp_df['bp_diastolic']
            temp_df['rate_pressure_product'] = temp_df['heart_rate'] * temp_df['bp_systolic'] / 100
            temp_df['shock_index'] = temp_df['heart_rate'] / temp_df['bp_systolic']
            temp_df['hemodynamic_instability'] = (
                (temp_df['bp_systolic'] < 90) | 
                (temp_df['heart_rate'] > 120) | 
                (temp_df['mean_arterial_pressure'] < 65)
            ).astype(int)
        
        # 3. OXYGENATION AND RESPIRATORY FEATURES
        print("   Creating oxygenation features...")
        
        if all(col in temp_df.columns for col in ['spo2', 'respiratory_rate', 'hemoglobin']):
            temp_df['oxygen_saturation_ratio'] = temp_df['spo2'] / 100
            temp_df['oxygen_content'] = (temp_df['hemoglobin'] * 1.34 * temp_df['spo2'] / 100) + (0.003 * temp_df['spo2'])
            temp_df['respiratory_distress'] = (
                (temp_df['spo2'] < 92) | (temp_df['respiratory_rate'] > 24)
            ).astype(int)
            temp_df['ventilation_perfusion_ratio'] = temp_df['respiratory_rate'] / (temp_df['spo2'] + 0.1)
        
        # 4. METABOLIC AND RENAL FEATURES
        print("   Creating metabolic features...")
        
        if all(col in temp_df.columns for col in ['creatinine', 'potassium', 'sodium', 'glucose']):
            temp_df['renal_function_score'] = temp_df['creatinine'] * temp_df['potassium']
            temp_df['electrolyte_imbalance'] = (
                (temp_df['sodium'] < 135) | (temp_df['sodium'] > 145) |
                (temp_df['potassium'] < 3.5) | (temp_df['potassium'] > 5.0)
            ).astype(int)
            temp_df['hyperglycemia'] = (temp_df['glucose'] > 180).astype(int)
            temp_df['hypoglycemia'] = (temp_df['glucose'] < 70).astype(int)
            temp_df['bun_creatinine_ratio'] = temp_df['glucose'] / (temp_df['creatinine'] + 0.1)  # Approximate
        
        # 5. INFLAMMATORY AND HEMATOLOGICAL FEATURES
        print("   Creating inflammatory features...")
        
        if 'white_blood_cells' in temp_df.columns:
            temp_df['wbc_elevated'] = (temp_df['white_blood_cells'] > 11).astype(int)
            temp_df['wbc_low'] = (temp_df['white_blood_cells'] < 4).astype(int)
            temp_df['leukocytosis'] = (temp_df['white_blood_cells'] > 15).astype(int)
        
        if 'hemoglobin' in temp_df.columns:
            temp_df['anemia'] = (temp_df['hemoglobin'] < 12).astype(int)
            temp_df['severe_anemia'] = (temp_df['hemoglobin'] < 8).astype(int)
            temp_df['polycythemia'] = (temp_df['hemoglobin'] > 16).astype(int)
        
        # 6. AGE AND COMORBIDITY FEATURES
        print("   Creating age-comorbidity features...")
        
        if 'anchor_age' in temp_df.columns:
            temp_df['age_squared'] = temp_df['anchor_age'] ** 2
            temp_df['age_cubed'] = temp_df['anchor_age'] ** 3
            temp_df['elderly'] = (temp_df['anchor_age'] >= 65).astype(int)
            temp_df['very_elderly'] = (temp_df['anchor_age'] >= 80).astype(int)
            temp_df['young_adult'] = ((temp_df['anchor_age'] >= 18) & (temp_df['anchor_age'] <= 40)).astype(int)
            
            # Age-adjusted thresholds
            temp_df['age_adjusted_hr_max'] = 220 - temp_df['anchor_age']
            temp_df['hr_percentage_max'] = temp_df['heart_rate'] / temp_df['age_adjusted_hr_max'] if 'heart_rate' in temp_df.columns else 0
        
        # 7. VITAL SIGN INTERACTIONS
        print("   Creating vital sign interactions...")
        
        vital_cols = ['heart_rate', 'respiratory_rate', 'spo2', 'temperature']
        available_vitals = [col for col in vital_cols if col in temp_df.columns]
        
        if len(available_vitals) >= 2:
            temp_df['vital_sign_mean'] = temp_df[available_vitals].mean(axis=1)
            temp_df['vital_sign_std'] = temp_df[available_vitals].std(axis=1)
            temp_df['vital_sign_range'] = temp_df[available_vitals].max(axis=1) - temp_df[available_vitals].min(axis=1)
            
            # Early Warning Score-like features
            temp_df['ews_score'] = 0
            if 'heart_rate' in temp_df.columns:
                temp_df['ews_score'] += ((temp_df['heart_rate'] < 50) | (temp_df['heart_rate'] > 100)).astype(int)
            if 'respiratory_rate' in temp_df.columns:
                temp_df['ews_score'] += ((temp_df['respiratory_rate'] < 12) | (temp_df['respiratory_rate'] > 20)).astype(int)
            if 'spo2' in temp_df.columns:
                temp_df['ews_score'] += (temp_df['spo2'] < 95).astype(int)
            if 'temperature' in temp_df.columns:
                temp_df['ews_score'] += ((temp_df['temperature'] < 36) | (temp_df['temperature'] > 38)).astype(int)
        
        # 8. CLINICAL RISK SCORES
        print("   Creating clinical risk scores...")
        
        # Simplified TIMI-like score
        temp_df['timi_like_score'] = 0
        if 'anchor_age' in temp_df.columns:
            temp_df['timi_like_score'] += (temp_df['anchor_age'] > 65).astype(int)
        if all(col in temp_df.columns for col in ['troponin_t', 'creatine_kinase_mb']):
            temp_df['timi_like_score'] += ((temp_df['troponin_t'] > 0.1) | (temp_df['creatine_kinase_mb'] > 5)).astype(int)
        if 'heart_rate' in temp_df.columns:
            temp_df['timi_like_score'] += (temp_df['heart_rate'] > 100).astype(int)
        if 'bp_systolic' in temp_df.columns:
            temp_df['timi_like_score'] += (temp_df['bp_systolic'] < 100).astype(int)
        
        # 9. POLYNOMIAL AND INTERACTION FEATURES
        print("   Creating polynomial features...")
        
        key_continuous = ['troponin_t', 'creatine_kinase_mb', 'hemoglobin', 'heart_rate', 'anchor_age']
        for col in key_continuous:
            if col in temp_df.columns:
                temp_df[f'{col}_squared'] = temp_df[col] ** 2
                temp_df[f'{col}_cubed'] = temp_df[col] ** 3
                temp_df[f'{col}_log'] = np.log1p(temp_df[col])
                temp_df[f'{col}_sqrt'] = np.sqrt(temp_df[col] + 0.1)
        
        # 10. RATIO FEATURES
        print("   Creating ratio features...")
        
        if all(col in temp_df.columns for col in ['hemoglobin', 'white_blood_cells']):
            temp_df['hgb_wbc_ratio'] = temp_df['hemoglobin'] / (temp_df['white_blood_cells'] + 0.1)
        
        if all(col in temp_df.columns for col in ['heart_rate', 'respiratory_rate']):
            temp_df['hr_rr_ratio'] = temp_df['heart_rate'] / (temp_df['respiratory_rate'] + 0.1)
        
        if all(col in temp_df.columns for col in ['spo2', 'respiratory_rate']):
            temp_df['spo2_rr_ratio'] = temp_df['spo2'] / (temp_df['respiratory_rate'] + 0.1)
        
        new_features = len(temp_df.columns) - original_features
        print(f"✅ Advanced feature engineering completed: {new_features} new features")
        print(f"   Total features: {len(temp_df.columns)}")
        
        return temp_df
    
    def handle_missing_values_advanced(self, df):
        """Advanced missing value handling without data removal"""
        print("🔄 Advanced Missing Value Handling...")
        
        df_imputed = df.copy()
        
        # Categorical columns
        categorical_cols = ['insurance', 'marital_status', 'gender', 'admission_type', 'race']
        for col in categorical_cols:
            if col in df_imputed.columns and df_imputed[col].isna().any():
                mode_val = df_imputed[col].mode()
                df_imputed[col].fillna(mode_val[0] if len(mode_val) > 0 else 'Unknown', inplace=True)
        
        # Numerical columns - use advanced imputation
        numerical_cols = df_imputed.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_imputed[col].isna().any():
                # Use median for basic imputation
                df_imputed[col].fillna(df_imputed[col].median(), inplace=True)
        
        print(f"✅ Missing values handled. Data shape: {df_imputed.shape}")
        return df_imputed
    
    def encode_features_advanced(self, df):
        """Advanced feature encoding"""
        print("🔡 Advanced Feature Encoding...")
        
        df_encoded = df.copy()
        
        # One-hot encoding for low cardinality
        low_cardinality = ['gender', 'admission_type']
        for col in low_cardinality:
            if col in df_encoded.columns:
                dummies = pd.get_dummies(df_encoded[col], prefix=col)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
        
        # Label encoding for high cardinality
        high_cardinality = ['insurance', 'marital_status', 'race']
        for col in high_cardinality:
            if col in df_encoded.columns:
                df_encoded[col] = pd.factorize(df_encoded[col])[0]
        
        # Remove original categorical columns
        df_encoded.drop(low_cardinality, axis=1, inplace=True, errors='ignore')
        
        print(f"✅ Encoding completed. Features: {len(df_encoded.columns)}")
        return df_encoded
    
    def advanced_feature_selection(self, X, y, method='ensemble', n_features=30):
        """Advanced feature selection methods"""
        print(f"🎯 Advanced Feature Selection: {method}")
        
        if method == 'ensemble':
            # Ensemble of multiple methods
            selected_features = set()
            
            # Method 1: LightGBM importance
            lgb_selector = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
            lgb_selector.fit(X, y)
            lgb_importance = pd.Series(lgb_selector.feature_importances_, index=X.columns)
            selected_features.update(lgb_importance.nlargest(n_features//2).index.tolist())
            
            # Method 2: Correlation
            correlations = X.corrwith(y).abs()
            selected_features.update(correlations.nlargest(n_features//2).index.tolist())
            
            # Method 3: Statistical
            selector = SelectKBest(score_func=f_classif, k=min(n_features//2, X.shape[1]))
            selector.fit(X, y)
            statistical_scores = pd.Series(selector.scores_, index=X.columns)
            selected_features.update(statistical_scores.nlargest(n_features//2).index.tolist())
            
            final_features = list(selected_features)[:n_features]
            
        elif method == 'lgb_rfe':
            # Recursive Feature Elimination with LightGBM
            estimator = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
            selector = RFE(estimator, n_features_to_select=n_features, step=5)
            selector.fit(X, y)
            final_features = X.columns[selector.support_].tolist()
            
        elif method == 'correlation_only':
            # Pure correlation-based
            correlations = X.corrwith(y).abs()
            final_features = correlations.nlargest(n_features).index.tolist()
            
        else:
            final_features = X.columns.tolist()
        
        print(f"✅ Selected {len(final_features)} features using {method}")
        return X[final_features]
    
    def train_advanced_models(self, X, y, model_type='lightgbm', params=None):
        """Train different advanced models"""
        try:
            print(f"🤖 Training {model_type}...")
            
            # Ensure data is clean
            X_clean = X.copy()
            for col in X_clean.columns:
                if X_clean[col].dtype == 'object':
                    X_clean[col] = pd.factorize(X_clean[col])[0]
            X_clean = X_clean.fillna(X_clean.median())
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"   Training on {X_train.shape[0]} samples, {X_train.shape[1]} features")
            
            # Train different models
            if model_type == 'lightgbm':
                model = lgb.LGBMClassifier(**params)
            elif model_type == 'random_forest':
                model = RandomForestClassifier(**params)
            elif model_type == 'gradient_boosting':
                model = GradientBoostingClassifier(**params)
            elif model_type == 'svm':
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                model = SVC(**params, probability=True)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model = LogisticRegression(**params)
            
            if model_type != 'svm':
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_clean, y, cv=5, scoring='accuracy')
            
            # Feature importance (if available)
            top_features = []
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
                top_features = sorted(zip(X_clean.columns, feature_importance), 
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
            print(f"❌ {model_type} training error: {str(e)}")
            return None
    
    def run_advanced_ensemble_optimization(self):
        """Run advanced ensemble optimization"""
        print("🚀 ADVANCED ENSEMBLE OPTIMIZATION FOR 95% ACCURACY")
        print("=" * 70)
        
        self.load_data()
        print(f"📊 Starting with: {len(self.df)} patients")
        
        # Advanced feature engineering
        df_engineered = self.advanced_medical_feature_engineering(self.df)
        
        # Handle missing values
        df_imputed = self.handle_missing_values_advanced(df_engineered)
        
        # Encode features
        df_encoded = self.encode_features_advanced(df_imputed)
        
        print(f"✅ Processing completed: {len(df_encoded)} patients, {len(df_encoded.columns)} features")
        
        if 'heart_attack' not in df_encoded.columns:
            print("❌ heart_attack column not found")
            return
        
        # Prepare data
        X = df_encoded.drop(columns=['heart_attack'])
        y = df_encoded['heart_attack']
        
        print(f"🎯 Final dataset: {X.shape[1]} features, {len(y)} patients")
        
        # Different feature selection methods
        fs_methods = [
            {'name': 'ensemble', 'n_features': 35},
            {'name': 'lgb_rfe', 'n_features': 30},
            {'name': 'correlation_only', 'n_features': 40}
        ]
        
        # Different models to try
        models_config = [
            {
                'name': 'lightgbm_advanced',
                'type': 'lightgbm',
                'params': {
                    'n_estimators': 2000,
                    'max_depth': 16,
                    'learning_rate': 0.01,
                    'num_leaves': 128,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.1,
                    'reg_lambda': 0.1,
                    'min_child_samples': 10,
                    'random_state': 42,
                    'verbose': -1
                }
            },
            {
                'name': 'random_forest_deep',
                'type': 'random_forest', 
                'params': {
                    'n_estimators': 500,
                    'max_depth': 20,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42
                }
            },
            {
                'name': 'gradient_boosting_advanced',
                'type': 'gradient_boosting',
                'params': {
                    'n_estimators': 500,
                    'learning_rate': 0.05,
                    'max_depth': 8,
                    'min_samples_split': 10,
                    'random_state': 42
                }
            }
        ]
        
        best_accuracy = 0
        best_combo = None
        
        for fs_method in fs_methods:
            print(f"\n🔍 Feature Selection: {fs_method['name']}")
            
            X_selected = self.advanced_feature_selection(
                X, y, method=fs_method['name'], n_features=fs_method['n_features']
            )
            
            for model_config in models_config:
                run_name = f"{fs_method['name']}_{model_config['name']}"
                
                with mlflow.start_run(run_name=run_name):
                    try:
                        print(f"\n🔬 Testing: {run_name}")
                        
                        # Log configuration
                        mlflow.log_params({
                            "feature_selection": fs_method['name'],
                            "model_type": model_config['type'],
                            "features_used": X_selected.shape[1],
                            "total_patients": len(df_encoded),
                            "feature_engineering": "advanced_medical"
                        })
                        
                        # Train and evaluate
                        results = self.train_advanced_models(
                            X_selected, y, 
                            model_type=model_config['type'],
                            params=model_config['params']
                        )
                        
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
                        
                        # Log top features if available
                        if results['top_features']:
                            for i, (feature, importance) in enumerate(results['top_features'][:5]):
                                mlflow.log_param(f"top_feature_{i+1}_name", feature)
                                mlflow.log_metric(f"top_feature_{i+1}_importance", float(importance))
                        
                        print(f"✅ {run_name}")
                        print(f"   📊 Accuracy: {results['accuracy']:.4f}")
                        print(f"   🎯 Precision: {results['precision']:.4f}")
                        print(f"   🔄 Recall: {results['recall']:.4f}")
                        print(f"   ⚖️ F1 Score: {results['f1_score']:.4f}")
                        print(f"   📈 AUC: {results['auc_score']:.4f}")
                        
                        if results['top_features']:
                            print(f"   🏆 Top Features:")
                            for feature, importance in results['top_features'][:3]:
                                print(f"     - {feature}: {importance:.4f}")
                        
                        # Track best result
                        if results['accuracy'] > best_accuracy:
                            best_accuracy = results['accuracy']
                            best_combo = {
                                'fs_method': fs_method,
                                'model_config': model_config,
                                'results': results
                            }
                        
                        if results['accuracy'] >= 0.95:
                            print("🎉🎉🎉 95% ACCURACY ACHIEVED! 🎉🎉🎉")
                            
                    except Exception as e:
                        print(f"❌ Error: {str(e)}")
        
        # Final results
        print(f"\n{'='*70}")
        print("🎯 ADVANCED ENSEMBLE OPTIMIZATION COMPLETED")
        print(f"{'='*70}")
        
        if best_combo:
            print(f"🏆 BEST CONFIGURATION:")
            print(f"   Feature Selection: {best_combo['fs_method']['name']}")
            print(f"   Model: {best_combo['model_config']['name']}")
            print(f"   📊 Accuracy: {best_accuracy:.4f}")
            print(f"   ⚖️ F1 Score: {best_combo['results']['f1_score']:.4f}")
            print(f"   Features Used: {best_combo['fs_method']['n_features']}")
            
            if best_combo['results']['top_features']:
                print(f"\n🏆 Top 5 Features:")
                for feature, importance in best_combo['results']['top_features'][:5]:
                    print(f"   - {feature}: {importance:.4f}")
            
            if best_accuracy >= 0.95:
                print(f"\n🎉 SUCCESS: 95%+ accuracy achieved!")
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
        optimizer = AdvancedEnsembleOptimizer(data_path)
        optimizer.run_advanced_ensemble_optimization()