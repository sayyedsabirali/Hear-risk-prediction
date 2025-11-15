import mlflow
import pandas as pd
import numpy as np
import os
import sys
import traceback
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

class OptimizeCatBoost:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.artifacts_dir = "catboost_artifacts"
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.setup_mlflow()
    
    def setup_mlflow(self):
        """Setup MLflow tracking"""
        mlflow.set_tracking_uri("mlruns")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        experiment_name = f"3. OPTIMIZED_CATBOOST_{timestamp}"

        mlflow.set_experiment(experiment_name)
        print(f"MLflow setup completed - Experiment: {experiment_name}")
    
    def load_data(self):
        """Load and prepare dataset"""
        self.df = pd.read_csv(self.data_path)
        id_cols = ['subject_id', 'hadm_id']
        self.df = self.df.drop(columns=[col for col in id_cols if col in self.df.columns], errors='ignore')
        
        print(f" Shape: {self.df.shape}")
        print(f" Target: {self.df['heart_attack'].value_counts().to_dict()}")
        return self.df
    
    def apply_specific_preprocessing(self, df, preprocessing_config):
        """Apply specific preprocessing based on given parameters"""
        try:
            print("🔄 Applying specific preprocessing...")
            print(f"   Config: {preprocessing_config}")
            
            df_clean = df.copy()
            
            # Identify columns
            numerical_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
            
            if 'heart_attack' in numerical_cols:
                numerical_cols.remove('heart_attack')
            
            print(f"   Numerical cols: {len(numerical_cols)}, Categorical cols: {len(categorical_cols)}")
            
            # 1. Handle missing values - Numerical
            print("   Handling missing values...")
            for col in numerical_cols:
                if col in df_clean.columns and df_clean[col].isnull().any():
                    if preprocessing_config['missing_numerical'] == 'median':
                        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                    elif preprocessing_config['missing_numerical'] == 'mean':
                        df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
            
            # Handle missing values - Categorical
            for col in categorical_cols:
                if col in df_clean.columns and df_clean[col].isnull().any():
                    if preprocessing_config['missing_categorical'] == 'mode':
                        mode_val = df_clean[col].mode()
                        if len(mode_val) > 0:
                            df_clean[col] = df_clean[col].fillna(mode_val[0])
                        else:
                            df_clean[col] = df_clean[col].fillna('Unknown')
                    elif preprocessing_config['missing_categorical'] == 'unknown':
                        df_clean[col] = df_clean[col].fillna('Unknown')
            
            print(f"   After missing value handling: {df_clean.shape}")
            
            # 2. Handle outliers
            if preprocessing_config['outlier_method'] != "none":
                print("   Handling outliers...")
                for col in numerical_cols:
                    if col in df_clean.columns:
                        Q1 = df_clean[col].quantile(0.25)
                        Q3 = df_clean[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - preprocessing_config['outlier_threshold'] * IQR
                        upper_bound = Q3 + preprocessing_config['outlier_threshold'] * IQR
                        
                        if preprocessing_config['outlier_method'] == 'clipper':
                            df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
                
                print(f"   After outlier handling: {df_clean.shape}")
            
            # 3. Reduce skewness
            if preprocessing_config['skewness_method'] != "none":
                print("   Reducing skewness...")
                for col in numerical_cols:
                    if col in df_clean.columns and df_clean[col].min() > 0:
                        if preprocessing_config['skewness_method'] == 'log':
                            df_clean[col] = np.log1p(df_clean[col])
                
                print(f"   After skewness reduction: {df_clean.shape}")
            
            # 4. Scale features
            if preprocessing_config['scaling_method'] != "none":
                print("   Scaling features...")
                # For simplicity, we'll skip scaling as CatBoost doesn't need it
                print("   Skipping scaling (CatBoost is scale-invariant)")
            
            # 5. Encode categorical
            print("   Encoding categorical...")
            for col in categorical_cols:
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].astype('category').cat.codes
            
            print(f"   After encoding: {df_clean.shape}")
            
            print("✅ Preprocessing completed successfully!")
            return df_clean
            
        except Exception as e:
            print(f"❌ Error in specific preprocessing: {str(e)}")
            traceback.print_exc()
            return None

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
            
            safe_name = model_name.replace(' ', '_').replace('/', '_')
            cm_path = os.path.join(self.artifacts_dir, f'confusion_matrix_{safe_name}.png')
            plt.savefig(cm_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Confusion matrix saved: {cm_path}")
            return cm_path
        except Exception as e:
            print(f"❌ Confusion matrix creation failed: {e}")
            return None

    def create_feature_importance_plot(self, feature_importance, feature_names, model_name, top_n=10):
        """Create and save feature importance plot"""
        try:
            # Create DataFrame for plotting
            importance_df = (
                pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
                .sort_values('importance', ascending=False)
                .head(top_n)
                .reset_index(drop=True)
            )
            
            plt.figure(figsize=(12, 8))
            bars = plt.barh(range(len(importance_df)), importance_df['importance'].values, 
                           color='steelblue', alpha=0.7)
            plt.yticks(range(len(importance_df)), importance_df['feature'].values)
            plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
            plt.ylabel('Features', fontsize=12, fontweight='bold')
            plt.title(f'Top {top_n} Feature Importances - {model_name}', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            
            # Add value labels on bars
            for i, v in enumerate(importance_df['importance'].values):
                plt.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            
            safe_name = model_name.replace(' ', '_').replace('/', '_')
            fi_path = os.path.join(self.artifacts_dir, f'feature_importance_{safe_name}.png')
            plt.savefig(fi_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Feature importance saved: {fi_path}")
            return fi_path, importance_df
            
        except Exception as e:
            print(f"❌ Feature importance plot failed: {e}")
            return None, None
    
    def train_catboost(self, X, y, params, preprocessing_name):
        """Train CatBoost with given parameters"""
        try:
            print(f"🤖 Training CatBoost model...")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"   Train size: {X_train.shape}, Test size: {X_test.shape}")
            
            # Train model
            model = CatBoostClassifier(**params)
            model.fit(X_train, y_train, verbose=False)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='binary')
            recall = recall_score(y_test, y_pred, average='binary')
            f1 = f1_score(y_test, y_pred, average='binary')
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # Feature importance
            feature_importance = model.get_feature_importance()
            top_features = sorted(zip(X.columns, feature_importance), 
                                key=lambda x: x[1], reverse=True)[:10]
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_score': auc_score,
                'top_features': top_features,
                'model': model,
                'y_test': y_test,
                'y_pred': y_pred,
                'feature_names': X.columns,
                'feature_importance': feature_importance
            }
            
        except Exception as e:
            print(f"❌ Error in model training: {str(e)}")
            traceback.print_exc()
            return None
    
    def run_catboost_optimization(self):
        """Run CatBoost optimization experiments"""
        print("🚀 OPTIMIZING CATBOOST FOR 95% ACCURACY")
        print("=" * 60)
        
        self.load_data()
        
        # Define the specific preprocessing configuration
        preprocessing_config = {
            'missing_numerical': 'median',
            'missing_categorical': 'mode', 
            'outlier_method': 'clipper',
            'outlier_threshold': 2.0,
            'skewness_method': 'log',
            'scaling_method': 'minmax'
        }
        
        # Apply specific preprocessing
        print("🔧 Applying specific preprocessing...")
        df_processed = self.apply_specific_preprocessing(self.df, preprocessing_config)
        
        if df_processed is None:
            print("❌ Preprocessing failed, exiting...")
            return
        
        print(f"✅ Processed data: {df_processed.shape}")
        
        # Check if heart_attack column exists
        if 'heart_attack' not in df_processed.columns:
            print("❌ heart_attack column not found in processed data")
            return
        
        # Prepare features and target
        X = df_processed.drop(columns=['heart_attack'])
        y = df_processed['heart_attack']
        
        print(f"🎯 Features: {X.shape[1]}, Target: {y.shape}")
        
        # CatBoost parameter combinations
        param_combinations = [
            # Basic optimized
            {
                "name": "basic_optimized",
                "params": {
                    "iterations": 1000,
                    "depth": 8,
                    "learning_rate": 0.05,
                    "l2_leaf_reg": 3,
                    "random_seed": 42,
                    "verbose": False
                }
            },
            # High capacity
            {
                "name": "high_capacity", 
                "params": {
                    "iterations": 2000,
                    "depth": 10,
                    "learning_rate": 0.03,
                    "l2_leaf_reg": 5,
                    "random_seed": 42,
                    "verbose": False
                }
            },
            # Deep learning
            {
                "name": "deep_learning",
                "params": {
                    "iterations": 1500,
                    "depth": 12,
                    "learning_rate": 0.02,
                    "l2_leaf_reg": 7,
                    "random_seed": 42,
                    "verbose": False
                }
            }
        ]
        
        best_accuracy = 0
        best_params = None
        best_results = None

        for param_combo in param_combinations:
            run_name = f"catboost_{param_combo['name']}"
            
            with mlflow.start_run(run_name=run_name):
                try:
                    print(f"\n🔬 Testing: {run_name}")
                    
                    # Log preprocessing parameters
                    mlflow.log_params({
                        "missing_numerical": preprocessing_config['missing_numerical'],
                        "missing_categorical": preprocessing_config['missing_categorical'],
                        "outlier_method": preprocessing_config['outlier_method'],
                        "outlier_threshold": preprocessing_config['outlier_threshold'],
                        "skewness_method": preprocessing_config['skewness_method'],
                        "scaling_method": preprocessing_config['scaling_method'],
                        "data_rows": len(df_processed)
                    })
                    
                    # Log CatBoost parameters
                    for param_name, param_value in param_combo['params'].items():
                        mlflow.log_param(f"catboost_{param_name}", param_value)
                    
                    # Train and evaluate
                    results = self.train_catboost(X, y, param_combo['params'], param_combo['name'])
                    
                    if results is None:
                        print(f"❌ {run_name} - Training failed")
                        continue
                    
                    # Log metrics
                    mlflow.log_metrics({
                        "accuracy": results['accuracy'],
                        "precision": results['precision'],
                        "recall": results['recall'],
                        "f1_score": results['f1_score'],
                        "auc_score": results['auc_score']
                    })
                    
                    # 🆕 CREATE AND LOG CONFUSION MATRIX
                    cm_path = self.create_confusion_matrix_plot(
                        results['y_test'], results['y_pred'], run_name
                    )
                    if cm_path:
                        mlflow.log_artifact(cm_path)
                    
                    # 🆕 CREATE AND LOG FEATURE IMPORTANCE PLOT
                    fi_path, importance_df = self.create_feature_importance_plot(
                        results['feature_importance'], 
                        results['feature_names'], 
                        run_name
                    )
                    if fi_path:
                        mlflow.log_artifact(fi_path)
                    
                    # 🆕 LOG TOP 10 FEATURES TO MLFLOW
                    top_features = results['top_features']
                    for i, (feature, importance) in enumerate(top_features[:10]):
                        mlflow.log_param(f"top_feature_{i+1}_name", feature)
                        mlflow.log_metric(f"top_feature_{i+1}_importance", importance)
                    
                    print(f"✅ {run_name} - Accuracy: {results['accuracy']:.4f}, AUC: {results['auc_score']:.4f}")
                    
                    # Show top features
                    print(f"   🏆 Top 5 Features:")
                    for feature, importance in results['top_features'][:5]:
                        print(f"     - {feature}: {importance:.4f}")
                    
                    # Track best result
                    if results['accuracy'] > best_accuracy:
                        best_accuracy = results['accuracy']
                        best_params = param_combo
                        best_results = results
                        
                        if results['accuracy'] >= 0.95:
                            print("🎉 CONGRATULATIONS! 95% ACCURACY ACHIEVED! 🎉")
                            
                except Exception as e:
                    print(f"❌ {run_name} - Error: {str(e)}")
                    traceback.print_exc()
        
        # Final summary
        print(f"\n{'='*60}")
        print("🎯 CATBOOST OPTIMIZATION COMPLETED!")
        print(f"{'='*60}")
        
        if best_params and best_results:
            print(f"🏆 BEST CATBOOST CONFIG: {best_params['name']}")
            print(f"🎯 Accuracy: {best_accuracy:.4f}")
            print(f"🎯 F1-Score: {best_results['f1_score']:.4f}")
            print(f"🎯 AUC: {best_results['auc_score']:.4f}")
            print(f"⚙️ Parameters: {best_params['params']}")
            
            print(f"\n🏆 TOP 5 FEATURES:")
            for feature, importance in best_results['top_features'][:5]:
                print(f"   - {feature}: {importance:.4f}")
            
        else:
            print("No successful runs completed")
        
        print(f"\nView results with: mlflow ui")
        print(f"Artifacts saved in: {self.artifacts_dir}")

# Main execution
if __name__ == "__main__":
    DATA_PATH = "F:\\18. MAJOR PROJECT\\Heart-related-content\\heart_risk_complete_dataset.csv"
    
    if not os.path.exists(DATA_PATH):
        print(f"Data file not found: {DATA_PATH}")
    else:
        optimizer = OptimizeCatBoost(DATA_PATH)
        optimizer.run_catboost_optimization()