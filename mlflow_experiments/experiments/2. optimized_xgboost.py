"""Complete Hyperparameter Tuning with Separate Preprocessing for Each Model - FIXED VERSION"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb

class HyperparameterTuner:
    """Complete hyperparameter tuning with separate preprocessing for each model - FIXED"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X_train_xgb, self.X_test_xgb, self.y_train_xgb, self.y_test_xgb = None, None, None, None
        
        # Create artifacts directory
        self.artifacts_dir = Path("hyperparameter_artifacts")
        self.artifacts_dir.mkdir(exist_ok=True)
        
        # Setup MLflow
        mlflow.set_tracking_uri("mlruns")
        
    def setup_mlflow_experiment(self, experiment_name):
        """Setup MLflow experiment"""
        try:
            # Check if experiment exists
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                print(f"Creating new experiment: {experiment_name}")
                experiment_id = mlflow.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
            
            mlflow.set_experiment(experiment_name)
            print(f"✅ MLflow Experiment Set: {experiment_name}")
            return True
        except Exception as e:
            print(f"❌ MLflow setup failed: {e}")
            return False


    def load_and_preprocess_data_separate(self):
        """Load and preprocess data with SEPARATE preprocessing for XGBoost only"""
        print("📂 Loading and preprocessing data with SEPARATE strategies...")
        self.df = pd.read_csv(self.data_path)
        original_rows = len(self.df)
        print(f"📊 Original dataset: {original_rows} rows, {self.df.shape[1]} columns")
        
        # Remove ID columns
        columns_to_remove = ['subject_id', 'hadm_id']
        existing_columns_to_remove = [col for col in columns_to_remove if col in self.df.columns]
        
        if existing_columns_to_remove:
            print(f"🗑️ Removing columns: {existing_columns_to_remove}")
            self.df = self.df.drop(columns=existing_columns_to_remove)
        
        # Apply XGBoost preprocessing only (CatBoost removed due to issues)
        print("\n🔧 Applying XGBoost preprocessing strategy:")
        
        # XGBoost: BEST preprocessing from Combo 12
        df_xgb = self.apply_xgboost_preprocessing(self.df.copy())
        
        print(f"✅ XGBoost data: {len(df_xgb)} rows")
        
        # Prepare features and target for XGBoost
        X_xgb = df_xgb.drop(columns=['heart_attack'])
        y_xgb = df_xgb['heart_attack']
        
        print(f"🎯 XGBoost - Features: {X_xgb.shape[1]}, Target: {y_xgb.value_counts().to_dict()}")
        
        # Split data for XGBoost
        self.X_train_xgb, self.X_test_xgb, self.y_train_xgb, self.y_test_xgb = train_test_split(
            X_xgb, y_xgb, test_size=0.2, random_state=42, stratify=y_xgb
        )
        
        print(f"📈 XGBoost - Train: {self.X_train_xgb.shape[0]}, Test: {self.X_test_xgb.shape[0]}")
        
        return df_xgb

    def apply_xgboost_preprocessing(self, df):
        """Apply XGBoost BEST preprocessing from Combo 12"""
        df_processed = df.copy()
        
        # Identify columns
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target from numerical if present
        if 'heart_attack' in numerical_cols:
            numerical_cols.remove('heart_attack')
        
        print(f"\n🎯 XGBoost Preprocessing (Combo 12 Strategy):")
        print("   - Missing Values: Median (Numerical), Unknown (Categorical)")
        print("   - Outliers: Capping at 3.0 IQR")
        print("   - Skewness: Log Transform")
        print("   - Scaling: None")
        
        # 1. Handle missing values - Combo 12: Median for numerical, Unknown for categorical
        for col in numerical_cols:
            if col in df_processed.columns and df_processed[col].isnull().any():
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        
        for col in categorical_cols:
            if col in df_processed.columns and df_processed[col].isnull().any():
                df_processed[col] = df_processed[col].fillna('Unknown')
        
        # 2. Handle outliers - Combo 12: Capping at 3.0 IQR
        for col in numerical_cols:
            if col in df_processed.columns:
                Q1 = df_processed[col].quantile(0.25)
                Q3 = df_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3.0 * IQR
                upper_bound = Q3 + 3.0 * IQR
                df_processed[col] = np.clip(df_processed[col], lower_bound, upper_bound)
        
        # 3. Reduce skewness - Combo 12: Log transform
        for col in numerical_cols:
            if col in df_processed.columns and df_processed[col].min() > 0:
                df_processed[col] = np.log1p(df_processed[col])
        
        # 4. Encode categorical variables
        for col in categorical_cols:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].astype('category').cat.codes
        
        return df_processed

    def create_confusion_matrix_plot(self, y_true, y_pred, model_name):
        """Create and save confusion matrix plot"""
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                       xticklabels=['No Heart Attack', 'Heart Attack'],
                       yticklabels=['No Heart Attack', 'Heart Attack'])
            plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
            plt.ylabel('True Label', fontsize=12)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.tight_layout()
            
            safe_name = model_name.replace('/', '_').replace(' ', '_')
            cm_path = self.artifacts_dir / f'confusion_matrix_{safe_name}.png'
            plt.savefig(str(cm_path), dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Confusion matrix saved: {cm_path}")
            return str(cm_path)
        except Exception as e:
            print(f"❌ Confusion matrix creation failed: {e}")
            return None

    def create_feature_importance_plot(self, feature_importance, feature_names, model_name):
        """Create and save feature importance plot"""
        try:
            # Create DataFrame for plotting
            importance_df = (
                pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
                .sort_values('importance', ascending=False)
                .head(10)
                .reset_index(drop=True)
            )
            
            plt.figure(figsize=(12, 8))
            bars = plt.barh(range(len(importance_df)), importance_df['importance'].values, 
                           color='steelblue', alpha=0.7)
            plt.yticks(range(len(importance_df)), importance_df['feature'].values)
            plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
            plt.ylabel('Features', fontsize=12, fontweight='bold')
            plt.title(f'Top 10 Feature Importances - {model_name}', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            
            # Add value labels on bars
            for i, v in enumerate(importance_df['importance'].values):
                plt.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            
            safe_name = model_name.replace('/', '_').replace(' ', '_')
            fi_path = self.artifacts_dir / f'feature_importance_{safe_name}.png'
            plt.savefig(str(fi_path), dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Feature importance saved: {fi_path}")
            return str(fi_path), importance_df
            
        except Exception as e:
            print(f"❌ Feature importance plot failed: {e}")
            return None, None

    def calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate all evaluation metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1_score': f1_score(y_true, y_pred, average='binary'),
            'auc': roc_auc_score(y_true, y_pred_proba)
        }

    def log_trial_metrics(self, metrics, params, model_name, trial_num, preprocessing_strategy):
        """Log metrics and parameters for each trial"""
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_params({
            'model': model_name,
            'trial_number': trial_num,
            'preprocessing_strategy': preprocessing_strategy,
            'features_count': metrics.get('features_count', 'N/A'),
            'train_size': metrics.get('train_size', 'N/A'),
            'test_size': metrics.get('test_size', 'N/A')
        })

    def tune_xgboost(self, n_trials=100):
        """XGBoost hyperparameter tuning with its own preprocessing"""
        print("\n" + "="*70)
        print("🚀 XGBOOST HYPERPARAMETER TUNING")
        print(f"   Testing {n_trials} combinations with Combo 12 preprocessing")
        print("="*70)
        
        if not self.setup_mlflow_experiment("2.OPTIMIZED_XGBoost_"):
            return None, None, None
        
        # XGBoost parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
            'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.15, 0.2],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5, 1, 2],
            'reg_lambda': [0, 0.1, 0.5, 1, 2, 5],
            'min_child_weight': [1, 3, 5, 7],
            'gamma': [0, 0.1, 0.2, 0.5, 1]
        }
        
        best_f1 = 0
        best_params = None
        best_metrics = None
        
        print(f"🔍 Testing {n_trials} XGBoost combinations...\n")
        
        for i in range(n_trials):
            # Randomly sample parameters
            params = {
                'n_estimators': np.random.choice(param_grid['n_estimators']),
                'max_depth': np.random.choice(param_grid['max_depth']),
                'learning_rate': np.random.choice(param_grid['learning_rate']),
                'subsample': np.random.choice(param_grid['subsample']),
                'colsample_bytree': np.random.choice(param_grid['colsample_bytree']),
                'reg_alpha': np.random.choice(param_grid['reg_alpha']),
                'reg_lambda': np.random.choice(param_grid['reg_lambda']),
                'min_child_weight': np.random.choice(param_grid['min_child_weight']),
                'gamma': np.random.choice(param_grid['gamma'])
            }
            
            run_name = f"XGB_Trial_{i+1:03d}"
            
            with mlflow.start_run(run_name=run_name, nested=True):
                try:
                    # Train model with XGBoost data
                    model = xgb.XGBClassifier(**params, random_state=42, eval_metric='logloss')
                    model.fit(self.X_train_xgb, self.y_train_xgb)
                    
                    # Predictions
                    y_pred = model.predict(self.X_test_xgb)
                    y_pred_proba = model.predict_proba(self.X_test_xgb)[:, 1]
                    
                    # Calculate metrics
                    metrics = self.calculate_metrics(self.y_test_xgb, y_pred, y_pred_proba)
                    metrics['features_count'] = self.X_train_xgb.shape[1]
                    metrics['train_size'] = len(self.y_train_xgb)
                    metrics['test_size'] = len(self.y_test_xgb)
                    
                    # Log trial metrics
                    self.log_trial_metrics(metrics, params, 'xgboost', i+1, 'Combo_12_Preprocessing')
                    
                    # Log top 5 features
                    feature_importance = model.feature_importances_
                    top_features = pd.DataFrame({
                        'feature': self.X_train_xgb.columns,
                        'importance': feature_importance
                    }).nlargest(10, 'importance')
                    
                    for j, (_, row) in enumerate(top_features.iterrows()):
                        mlflow.log_metric(f"top_feature_{j+1}", row['importance'])
                        mlflow.log_param(f"top_feature_{j+1}_name", row['feature'])
                    
                    # Update best model
                    if metrics['f1_score'] > best_f1:
                        best_f1 = metrics['f1_score']
                        best_params = params
                        best_metrics = metrics
                    
                    print(f"  ✅ Trial {i+1:03d}: F1={metrics['f1_score']:.4f} | Acc={metrics['accuracy']:.4f} | AUC={metrics['auc']:.4f}")
                    
                except Exception as e:
                    print(f"  ❌ Trial {i+1:03d} failed: {str(e)}")
                    continue
        
        # Log BEST XGBoost model with artifacts
        if best_params is not None:
            print(f"\n🏆 Logging BEST XGBoost Model (F1: {best_f1:.4f})...")
            
            with mlflow.start_run(run_name="XGB_BEST_MODEL", nested=True):
                # Train best model
                best_model = xgb.XGBClassifier(**best_params, random_state=42, eval_metric='logloss')
                best_model.fit(self.X_train_xgb, self.y_train_xgb)
                
                # Predictions
                y_pred = best_model.predict(self.X_test_xgb)
                y_pred_proba = best_model.predict_proba(self.X_test_xgb)[:, 1]
                
                # Calculate final metrics
                final_metrics = self.calculate_metrics(self.y_test_xgb, y_pred, y_pred_proba)
                final_metrics['features_count'] = self.X_train_xgb.shape[1]
                final_metrics['train_size'] = len(self.y_train_xgb)
                final_metrics['test_size'] = len(self.y_test_xgb)
                
                # Log everything
                self.log_trial_metrics(final_metrics, best_params, 'xgboost_best', 'BEST', 'Combo_12_Preprocessing')
                
                # Create and log artifacts
                cm_path = self.create_confusion_matrix_plot(self.y_test_xgb, y_pred, "XGBoost_BEST_Combo12")
                if cm_path:
                    mlflow.log_artifact(cm_path)
                
                feature_importance = best_model.feature_importances_
                fi_path, importance_df = self.create_feature_importance_plot(
                    feature_importance, self.X_train_xgb.columns, "XGBoost_BEST_Combo12"
                )
                if fi_path:
                    mlflow.log_artifact(fi_path)
                
                # Log top 10 features in detail
                if importance_df is not None:
                    for idx, row in importance_df.iterrows():
                        mlflow.log_param(f"feature_{idx+1:02d}_name", row['feature'])
                        mlflow.log_metric(f"feature_{idx+1:02d}_importance", row['importance'])
                
                # Log model
                mlflow.xgboost.log_model(best_model, "model")
                
                print(f"\n✅ XGBoost Tuning Complete!")
                print(f"   Best F1-Score: {final_metrics['f1_score']:.4f}")
                print(f"   Best Accuracy: {final_metrics['accuracy']:.4f}")
                print(f"   Best AUC: {final_metrics['auc']:.4f}")
                print(f"   Preprocessing: Combo 12 Strategy")
                print(f"   Best Parameters: {best_params}")
            
            return best_model, best_params, final_metrics
        else:
            print("❌ No successful XGBoost runs")
            return None, None, None

    def run_complete_tuning(self):
        """Run complete hyperparameter tuning pipeline - XGBoost only"""
        print("\n" + "="*80)
        print("🚀 COMPLETE HYPERPARAMETER TUNING PIPELINE")
        print("="*80)
        print("   Model: XGBoost Only (CatBoost removed due to technical issues)")
        print("   Trials: 100 XGBoost combinations")
        print("   Preprocessing: Combo 12 strategy")
        print("="*80)
        
        # Load and preprocess data
        df_xgb = self.load_and_preprocess_data_separate()
        
        # Tune XGBoost with its own preprocessing - 100 TRIALS
        xgb_model, xgb_params, xgb_metrics = self.tune_xgboost(n_trials=100)
        
        # Final results
        print("\n" + "="*80)
        print("🏆 FINAL XGBOOST RESULTS")
        print("="*80)
        
        if xgb_metrics:
            print(f"\n🎯 XGBoost with Combo 12 Preprocessing:")
            print(f"   Accuracy:  {xgb_metrics['accuracy']:.4f}")
            print(f"   F1-Score:  {xgb_metrics['f1_score']:.4f}")
            print(f"   Precision: {xgb_metrics['precision']:.4f}")
            print(f"   Recall:    {xgb_metrics['recall']:.4f}")
            print(f"   AUC:       {xgb_metrics['auc']:.4f}")
            print(f"   Features:  {xgb_metrics['features_count']}")
            print(f"   Train Size: {xgb_metrics['train_size']:,}")
            print(f"   Test Size:  {xgb_metrics['test_size']:,}")
            
            # Save final results
            results_df = pd.DataFrame([{
                'Model': 'XGBoost (Combo 12)',
                'Accuracy': xgb_metrics['accuracy'],
                'F1-Score': xgb_metrics['f1_score'],
                'Precision': xgb_metrics['precision'],
                'Recall': xgb_metrics['recall'],
                'AUC': xgb_metrics['auc'],
                'Preprocessing': 'Combo 12'
            }])
            
            results_path = self.artifacts_dir / 'final_xgboost_results.csv'
            results_df.to_csv(results_path, index=False)
            print(f"\n💾 Results saved: {results_path}")
        else:
            print("❌ No results available")
        
        print("\n📊 View detailed results:")
        print("   MLflow UI: mlflow ui")
        print("   Artifacts: hyperparameter_artifacts/")
        print("="*80)
        
        return {
            'xgb_model': xgb_model,
            'xgb_metrics': xgb_metrics
        }

# Run the complete tuning pipeline
if __name__ == "__main__":
    # Update this path to your dataset
    DATA_PATH = "F:\\18. MAJOR PROJECT\\Heart-related-content\\heart_risk_complete_dataset.csv"
    
    print("Starting Hyperparameter Tuning with XGBoost Only...")
    tuner = HyperparameterTuner(DATA_PATH)
    results = tuner.run_complete_tuning()