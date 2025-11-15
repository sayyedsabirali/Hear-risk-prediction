import mlflow
import pandas as pd
import numpy as np
import os
import sys
import traceback
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessor import DataPreprocessor

class OptimizeLightGBM:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.preprocessor = DataPreprocessor()
        self.setup_mlflow()
    
    def setup_mlflow(self):
        """Setup MLflow tracking"""
        mlflow.set_tracking_uri("mlruns")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        experiment_name = f"4.OPTIMIZED_LIGHTGBM_{timestamp}"

        mlflow.set_experiment(experiment_name)
        print(f"MLflow setup completed - Experiment: {experiment_name}")
    
    def load_data(self):
        """Load the dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        
        # Remove subject_id and admission_iid columns if they exist
        columns_to_remove = ['subject_id', 'admission_iid']
        existing_columns_to_remove = [col for col in columns_to_remove if col in self.df.columns]
        
        if existing_columns_to_remove:
            self.df = self.df.drop(columns=existing_columns_to_remove)
            print(f" Removed columns: {existing_columns_to_remove}")
        
        print(f" Dataset loaded: {self.df.shape}")
        print(f" Columns: {list(self.df.columns)}")
        return self.df
    
    def apply_specific_preprocessing(self, df, preprocessing_config):
        """Apply specific preprocessing based on given parameters"""
        try:
            print("🔄 Applying specific preprocessing...")
            print(f"   Config: {preprocessing_config}")
            
            # Step 1: Identify columns first
            numerical_cols, categorical_cols = self.preprocessor.identify_columns(df)
            print(f"   Numerical cols: {len(numerical_cols)}, Categorical cols: {len(categorical_cols)}")
            
            # Step 2: Handle missing values
            print("   Handling missing values...")
            df_clean = self.preprocessor.handle_missing_values(
                df, 
                preprocessing_config['missing_numerical'],
                preprocessing_config['missing_categorical']
            )
            print(f"   After missing value handling: {df_clean.shape}")
            
            # Step 3: SKIP OUTLIER HANDLING - Don't remove outliers
            print("   ⚠️ Skipping outlier handling (keeping all data points)")
            # No outlier removal - keep all data as is
            
            # Step 4: Reduce skewness
            if preprocessing_config['skewness_method'] != "none":
                print("   Reducing skewness...")
                df_clean = self.preprocessor.reduce_skewness(
                    df_clean, preprocessing_config['skewness_method']
                )
                print(f"   After skewness reduction: {df_clean.shape}")
            
            # Step 5: Scale features
            if preprocessing_config['scaling_method'] != "none":
                print("   Scaling features...")
                df_clean = self.preprocessor.scale_features(
                    df_clean, preprocessing_config['scaling_method']
                )
                print(f"   After scaling: {df_clean.shape}")
            
            # Step 6: Encode categorical
            print("   Encoding categorical...")
            df_clean = self.preprocessor.encode_categorical(df_clean)
            print(f"   After encoding: {df_clean.shape}")
            
            print("✅ Preprocessing completed successfully!")
            return df_clean
            
        except Exception as e:
            print(f"❌ Error in specific preprocessing: {str(e)}")
            traceback.print_exc()
            return None
    
    def plot_feature_importance(self, feature_importance, top_n=10, run_name=""):
        """Plot feature importance"""
        try:
            # Create plot
            plt.figure(figsize=(10, 6))
            features = [x[0] for x in feature_importance[:top_n]]
            importance_values = [x[1] for x in feature_importance[:top_n]]
            
            plt.barh(range(len(features)), importance_values, align='center')
            plt.yticks(range(len(features)), features)
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_n} Feature Importance - {run_name}')
            plt.gca().invert_yaxis()
            
            # Save plot
            os.makedirs('feature_plots', exist_ok=True)
            plot_path = f'feature_plots/feature_importance_{run_name}.png'
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Log to MLflow
            mlflow.log_artifact(plot_path)
            print(f"   📊 Feature importance plot saved: {plot_path}")
            
            return plot_path
        except Exception as e:
            print(f"   ⚠️ Could not create feature importance plot: {e}")
            return None
    
    def plot_confusion_matrix(self, y_true, y_pred, run_name=""):
        """Plot confusion matrix"""
        try:
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Create plot
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['No Heart Attack', 'Heart Attack'],
                       yticklabels=['No Heart Attack', 'Heart Attack'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Confusion Matrix - {run_name}')
            
            # Save plot
            os.makedirs('confusion_matrices', exist_ok=True)
            plot_path = f'confusion_matrices/confusion_matrix_{run_name}.png'
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Log to MLflow
            mlflow.log_artifact(plot_path)
            print(f"   📈 Confusion matrix plot saved: {plot_path}")
            
            return cm, plot_path
        except Exception as e:
            print(f"   ⚠️ Could not create confusion matrix plot: {e}")
            return None, None
    
    def train_lightgbm(self, X, y, params, preprocessing_name):
        """Train LightGBM with given parameters"""
        try:
            print(f"🤖 Training LightGBM model...")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"   Train size: {X_train.shape}, Test size: {X_test.shape}")
            
            # Train model
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='binary')
            recall = recall_score(y_test, y_pred, average='binary')
            f1 = f1_score(y_test, y_pred, average='binary')
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Feature importance
            feature_importance = model.feature_importances_
            top_features = sorted(zip(X.columns, feature_importance), 
                                key=lambda x: x[1], reverse=True)[:10]  # Top 10 features
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_score': auc_score,
                'confusion_matrix': cm,
                'y_true': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'top_features': top_features,
                'model': model,
                'feature_names': X.columns.tolist()
            }
            
        except Exception as e:
            print(f"❌ Error in model training: {str(e)}")
            traceback.print_exc()
            return None
    
    def run_lightgbm_optimization(self):
        """Run LightGBM optimization experiments"""
        print("🚀 OPTIMIZING LIGHTGBM FOR 95% ACCURACY")
        print("=" * 60)
        
        self.load_data()
        
        # Define the specific preprocessing configuration (from combo_015) - WITHOUT OUTLIER REMOVAL
        preprocessing_config = {
            'missing_numerical': 'median',
            'missing_categorical': 'mode', 
            'outlier_method': 'none',  # Changed to 'none' to skip outlier removal
            'outlier_threshold': 1.5,
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
        
        # LightGBM parameter combinations - Optimized for this preprocessing
        param_combinations = [
            # Current Best + Improvements
            {
                "name": "current_optimized",
                "params": {
                    "n_estimators": 500,
                    "max_depth": 12,
                    "learning_rate": 0.05,
                    "num_leaves": 64,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_alpha": 0.1,
                    "reg_lambda": 0.1,
                    "min_child_samples": 20,
                    "random_state": 42,
                    "verbose": -1
                }
            },
            # High Precision (for your high precision score)
            {
                "name": "high_precision", 
                "params": {
                    "n_estimators": 1000,
                    "max_depth": 15,
                    "learning_rate": 0.02,
                    "num_leaves": 128,
                    "subsample": 0.7,
                    "colsample_bytree": 0.7,
                    "reg_alpha": 0.3,
                    "reg_lambda": 0.3,
                    "min_child_samples": 30,
                    "scale_pos_weight": 1,
                    "random_state": 42,
                    "verbose": -1
                }
            },
            # Balanced F1 Score
            {
                "name": "balanced_f1",
                "params": {
                    "n_estimators": 800,
                    "max_depth": 10,
                    "learning_rate": 0.03,
                    "num_leaves": 96,
                    "subsample": 0.85,
                    "colsample_bytree": 0.85,
                    "reg_alpha": 0.2,
                    "reg_lambda": 0.2,
                    "min_child_samples": 15,
                    "random_state": 42,
                    "verbose": -1
                }
            },
            # Ultra High Capacity
            {
                "name": "ultra_capacity",
                "params": {
                    "n_estimators": 1500,
                    "max_depth": 20,
                    "learning_rate": 0.01,
                    "num_leaves": 256,
                    "subsample": 0.6,
                    "colsample_bytree": 0.6,
                    "reg_alpha": 0.5,
                    "reg_lambda": 0.5,
                    "min_child_samples": 10,
                    "random_state": 42,
                    "verbose": -1
                }
            },
            # Fast Learning
            {
                "name": "fast_learning",
                "params": {
                    "n_estimators": 300,
                    "max_depth": 8,
                    "learning_rate": 0.1,
                    "num_leaves": 32,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                    "reg_alpha": 0.05,
                    "reg_lambda": 0.05,
                    "min_child_samples": 5,
                    "random_state": 42,
                    "verbose": -1
                }
            },
            # Medical Data Optimized
            {
                "name": "medical_optimized",
                "params": {
                    "n_estimators": 1200,
                    "max_depth": 18,
                    "learning_rate": 0.015,
                    "num_leaves": 150,
                    "subsample": 0.75,
                    "colsample_bytree": 0.75,
                    "reg_alpha": 0.4,
                    "reg_lambda": 0.4,
                    "min_child_samples": 25,
                    "min_child_weight": 0.001,
                    "random_state": 42,
                    "verbose": -1
                }
            },
            # Conservative Regularization
            {
                "name": "conservative",
                "params": {
                    "n_estimators": 2000,
                    "max_depth": 25,
                    "learning_rate": 0.005,
                    "num_leaves": 200,
                    "subsample": 0.5,
                    "colsample_bytree": 0.5,
                    "reg_alpha": 1.0,
                    "reg_lambda": 1.0,
                    "min_child_samples": 50,
                    "random_state": 42,
                    "verbose": -1
                }
            }
        ]
        
        best_accuracy = 0
        best_f1 = 0
        best_params = None

        for param_combo in param_combinations:
            run_name = f"lightgbm_{param_combo['name']}"
            
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
                    
                    # Log LightGBM parameters
                    for param_name, param_value in param_combo['params'].items():
                        mlflow.log_param(f"lgb_{param_name}", param_value)
                    
                    # Train and evaluate
                    results = self.train_lightgbm(X, y, param_combo['params'], param_combo['name'])
                    
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
                    
                    # 🆕 LOG CONFUSION MATRIX VALUES
                    cm = results['confusion_matrix']
                    tn, fp, fn, tp = cm.ravel()
                    mlflow.log_metrics({
                        "true_negative": tn,
                        "false_positive": fp,
                        "false_negative": fn,
                        "true_positive": tp
                    })
                    
                    # 🆕 CREATE AND LOG CONFUSION MATRIX PLOT
                    self.plot_confusion_matrix(results['y_true'], results['y_pred'], run_name)
                    
                    # 🆕 CREATE AND LOG FEATURE IMPORTANCE PLOT
                    self.plot_feature_importance(results['top_features'], top_n=10, run_name=run_name)
                    
                    # 🆕 LOG TOP 3 FEATURES TO MLFLOW
                    top_features = results['top_features']
                    for i, (feature, importance) in enumerate(top_features[:3]):
                        mlflow.log_param(f"top_feature_{i+1}_name", feature)
                        mlflow.log_metric(f"top_feature_{i+1}_importance", float(importance))
                    
                    # 🆕 LOG ALL TOP FEATURES AS STRING FOR ANALYSIS
                    all_feature_names = [f[0] for f in top_features]
                    all_feature_importances = [float(f[1]) for f in top_features]
                    mlflow.log_param("all_top_features", str(all_feature_names))
                    mlflow.log_param("all_feature_importances", str(all_feature_importances))
                    
                    print(f"✅ {run_name}")
                    print(f"   📊 Accuracy: {results['accuracy']:.4f}")
                    print(f"   🎯 Precision: {results['precision']:.4f}")
                    print(f"   🔄 Recall: {results['recall']:.4f}")
                    print(f"   ⚖️ F1 Score: {results['f1_score']:.4f}")
                    print(f"   📈 AUC: {results['auc_score']:.4f}")
                    
                    # Show confusion matrix details
                    print(f"   🎯 Confusion Matrix:")
                    print(f"     - True Negative: {tn}")
                    print(f"     - False Positive: {fp}")
                    print(f"     - False Negative: {fn}")
                    print(f"     - True Positive: {tp}")
                    
                    # Show top features
                    print(f"   🏆 Top 5 Features:")
                    for feature, importance in results['top_features'][:5]:
                        print(f"     - {feature}: {importance:.4f}")
                    
                    # Track best result
                    if results['accuracy'] > best_accuracy:
                        best_accuracy = results['accuracy']
                        best_f1 = results['f1_score']
                        best_params = param_combo
                        
                        if results['accuracy'] >= 0.95:
                            print("🎉 CONGRATULATIONS! 95% ACCURACY ACHIEVED! 🎉")
                            
                except Exception as e:
                    print(f"❌ {run_name} - Error: {str(e)}")
                    traceback.print_exc()
        
        # Final summary
        print(f"\n{'='*60}")
        print("🎯 LIGHTGBM OPTIMIZATION COMPLETED!")
        print(f"{'='*60}")
        
        if best_params:
            print(f"🏆 BEST LIGHTGBM CONFIG: {best_params['name']}")
            print(f"🎯 Accuracy: {best_accuracy:.4f}")
            print(f"⚖️ F1 Score: {best_f1:.4f}")
            print(f"⚙️ Key Parameters:")
            for param, value in list(best_params['params'].items())[:5]:
                print(f"   - {param}: {value}")
            
            if best_accuracy >= 0.95:
                print("🎉🎉🎉 95% ACCURACY TARGET ACHIEVED! 🎉🎉🎉")
            else:
                improvement = (0.95 - best_accuracy) * 100
                print(f"📈 Need {improvement:.2f}% improvement to reach 95%")
        else:
            print("❌ No successful runs completed")
        
        print(f"\n📊 View results with: mlflow ui")

if __name__ == "__main__":
    data_path = "F:\\18. MAJOR PROJECT\\Heart-related-content\\heart_risk_complete_dataset.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
    else:
        optimizer = OptimizeLightGBM(data_path)
        optimizer.run_lightgbm_optimization()