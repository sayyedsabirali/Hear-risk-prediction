import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
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
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

class OptimizedModelTrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.setup_mlflow()
    
    def setup_mlflow(self):
        """Setup MLflow tracking"""
        mlflow.set_tracking_uri("mlruns")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        experiment_name = f"11. Final_Model_log_{timestamp}"

        mlflow.set_experiment(experiment_name)
        print(f"MLflow setup completed - Experiment: {experiment_name}")
    
    def load_data(self):
        """Load the dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded: {self.df.shape}")
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
            if 'heart_rate' in temp_df.columns:
                temp_df['age_adjusted_hr_max'] = 220 - temp_df['anchor_age']
                temp_df['hr_percentage_max'] = temp_df['heart_rate'] / temp_df['age_adjusted_hr_max']
        
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
            
        else:
            final_features = X.columns.tolist()
        
        print(f"✅ Selected {len(final_features)} features using {method}")
        return X[final_features]
    
    def train_optimized_model(self):
        """Train model with exact parameters from your metrics"""
        print("🚀 TRAINING OPTIMIZED MODEL WITH EXACT PARAMETERS")
        print("=" * 70)
        
        # Load and prepare data
        self.load_data()
        print(f"📊 Starting with: {len(self.df)} patients")
        
        # Apply the exact same preprocessing pipeline
        df_engineered = self.advanced_medical_feature_engineering(self.df)
        df_imputed = self.handle_missing_values_advanced(df_engineered)
        df_encoded = self.encode_features_advanced(df_imputed)
        
        print(f"✅ Processing completed: {len(df_encoded)} patients, {len(df_encoded.columns)} features")
        
        if 'heart_attack' not in df_encoded.columns:
            print("❌ heart_attack column not found")
            return
        
        # Prepare data
        X = df_encoded.drop(columns=['heart_attack'])
        y = df_encoded['heart_attack']
        
        print(f"🎯 Final dataset: {X.shape[1]} features, {len(y)} patients")
        
        # Apply exact feature selection method
        X_selected = self.advanced_feature_selection(
            X, y, method='ensemble', n_features=30
        )
        
        # Train with exact parameters
        with mlflow.start_run(run_name="OPTIMIZED_MODEL_EXACT_PARAMS"):
            try:
                # Log exact parameters from your metrics
                mlflow.log_params({
                    "features_used": 30,
                    "feature_engineering": "advanced_medical",
                    "feature_selection": "ensemble",
                    "model_type": "lightgbm",
                    "total_patients": len(df_encoded)
                })
                
                # Train LightGBM with optimized parameters
                print("🤖 Training LightGBM with optimized parameters...")
                
                # Ensure data is clean
                X_clean = X_selected.copy()
                for col in X_clean.columns:
                    if X_clean[col].dtype == 'object':
                        X_clean[col] = pd.factorize(X_clean[col])[0]
                X_clean = X_clean.fillna(X_clean.median())
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_clean, y, test_size=0.2, random_state=42, stratify=y
                )
                
                print(f"   Training on {X_train.shape[0]} samples, {X_train.shape[1]} features")
                
                # Train LightGBM model (adjust parameters to match your metrics)
                model = lgb.LGBMClassifier(
                    n_estimators=2000,
                    max_depth=12,
                    learning_rate=0.02,
                    num_leaves=64,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    min_child_samples=20,
                    random_state=42,
                    verbose=-1
                )
                
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
                # cv_scores = cross_val_score(model, X_clean, y, cv=5, scoring='accuracy')
                
                # Feature importance
                feature_importance = model.feature_importances_
                top_features = sorted(zip(X_clean.columns, feature_importance), 
                                    key=lambda x: x[1], reverse=True)[:10]
                
                # Log metrics
                mlflow.log_metrics({
                    "accuracy": accuracy,
                    "auc_score": auc_score,
                    # "cv_mean_accuracy": cv_scores.mean(),
                    "f1_score": f1,
                    "precision": precision,
                    "recall": recall
                })
                
                # Log top features
                if top_features:
                    for i, (feature, importance) in enumerate(top_features[:5]):
                        mlflow.log_param(f"top_feature_{i+1}_name", feature)
                        mlflow.log_metric(f"top_feature_{i+1}_importance", float(importance))
                
                # ✅ PROPER MODEL LOGGING WITH SIGNATURE AND INPUT EXAMPLE
                print("📦 Logging model to MLflow with signature and input example...")
                
                # Infer the model signature
                signature = infer_signature(X_train, model.predict(X_train))
                
                # Create input example (first 2 rows)
                input_example = X_train.iloc[:2]
                
                # Log the model with proper signature and input example
                model_info = mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="heart_attack_model",
                    signature=signature,
                    input_example=input_example,
                    registered_model_name="HeartAttackPredictor"
                )
                
                # ✅ FIXED: Use regular tags instead of set_logged_model_tags
                mlflow.set_tags({
                    "model_type": "LightGBM Classifier",
                    "purpose": "Heart Attack Risk Prediction", 
                    "domain": "Healthcare",
                    "training_data": "MIMIC-IV Clinical Data",
                    "feature_engineering": "Advanced Medical Features",
                    "performance": f"Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}",
                    "dataset_info": f"{len(df_encoded)} patients, {X_clean.shape[1]} features"
                })
                
                # ✅ LOG FEATURE IMPORTANCE PLOT
                print("📊 Creating feature importance plot...")
                plt.figure(figsize=(12, 8))
                top_10_features = top_features[:10]
                features, importances = zip(*top_10_features)
                
                y_pos = np.arange(len(features))
                plt.barh(y_pos, importances, align='center', alpha=0.7)
                plt.yticks(y_pos, features)
                plt.xlabel('Feature Importance')
                plt.title('Top 10 Most Important Features for Heart Attack Prediction')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                
                # Save plot
                plt.savefig("feature_importance.png", dpi=300, bbox_inches='tight')
                mlflow.log_artifact("feature_importance.png")
                plt.close()
                
                # Save feature list and schema information
                feature_schema = {
                    "input_features": X_clean.columns.tolist(),
                    "target_variable": "heart_attack",
                    "feature_count": X_clean.shape[1],
                    "data_types": {col: str(dtype) for col, dtype in X_clean.dtypes.items()},
                    "model_signature": {
                        "inputs": str(signature.inputs),
                        "outputs": str(signature.outputs)
                    }
                }
                
                with open("model_schema.json", "w") as f:
                    json.dump(feature_schema, f, indent=2)
                mlflow.log_artifact("model_schema.json")
                
                # Log feature statistics
                feature_stats = X_clean.describe().to_dict()
                with open("feature_statistics.json", "w") as f:
                    json.dump(feature_stats, f, indent=2)
                mlflow.log_artifact("feature_statistics.json")
                
                # Log model configuration
                model_config = {
                    "model_parameters": model.get_params(),
                    "training_data_shape": {
                        "X_train": X_train.shape,
                        "X_test": X_test.shape,
                        "y_train": y_train.shape,
                        "y_test": y_test.shape
                    },
                    "performance_metrics": {
                        "accuracy": float(accuracy),
                        "precision": float(precision),
                        "recall": float(recall),
                        "f1_score": float(f1),
                        "auc_score": float(auc_score),
                        # "cv_mean_accuracy": float(cv_scores.mean())
                    },
                    "training_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                with open("model_config.json", "w") as f:
                    json.dump(model_config, f, indent=2)
                mlflow.log_artifact("model_config.json")
                
                print(f"\n✅ MODEL TRAINING AND LOGGING COMPLETED SUCCESSFULLY!")
                print(f"📊 Accuracy: {accuracy:.4f}")
                print(f"🎯 Precision: {precision:.4f}")
                print(f"🔄 Recall: {recall:.4f}")
                print(f"⚖️ F1 Score: {f1:.4f}")
                print(f"📈 AUC Score: {auc_score:.4f}")
                # print(f"📋 Cross-Validation Mean Accuracy: {cv_scores.mean():.4f}")
                
                if top_features:
                    print(f"\n🏆 Top 5 Features:")
                    for feature, importance in top_features[:5]:
                        print(f"   - {feature}: {importance:.4f}")
                
                print(f"\n📦 Model Logging Details:")
                print(f"   - Model Registered: HeartAttackPredictor")
                print(f"   - Signature: {signature}")
                print(f"   - Input Example Shape: {input_example.shape}")
                print(f"   - Run ID: {mlflow.active_run().info.run_id}")
                
                # Clean up temporary files
                for file in ["feature_importance.png", "model_schema.json", "feature_statistics.json", "model_config.json"]:
                    if os.path.exists(file):
                        os.remove(file)
                
                return {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'auc_score': auc_score,
                    # 'cv_mean_accuracy': cv_scores.mean(),
                    'top_features': top_features,
                    'model': model,
                    'model_info': model_info,
                    'signature': signature,
                    'run_id': mlflow.active_run().info.run_id
                }
                
            except Exception as e:
                print(f"❌ Training error: {str(e)}")
                traceback.print_exc()
                return None

from mlflow_experiments.config.data_config import DATA_PATH

if __name__ == "__main__":
    data_path = DATA_PATH
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("Please update the data_path variable with your actual file path")
    else:
        trainer = OptimizedModelTrainer(data_path)
        results = trainer.train_optimized_model()
        
        if results:
            print(f"\n🎉 Model trained and logged successfully!")
            print(f"📊 Final Accuracy: {results['accuracy']:.4f}")
            print(f"🔍 Check MLflow UI: mlflow ui --port 5000")
            print(f"📦 Model registered as: HeartAttackPredictor")
            print(f"🏷️ Run ID: {results['run_id']}")