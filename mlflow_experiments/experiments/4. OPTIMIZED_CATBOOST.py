import mlflow
import pandas as pd
import numpy as np
import os
import sys
import traceback
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.preprocessor import BoxPlotPreprocessor

class OptimizeCatBoost:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.preprocessor = BoxPlotPreprocessor()
        self.setup_mlflow()
    
    def setup_mlflow(self):
        """Setup MLflow tracking"""
        mlflow.set_tracking_uri("mlruns")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        experiment_name = f"4. OPTIMIZED_CATBOOST_{timestamp}"

        mlflow.set_experiment(experiment_name)
        print(f"MLflow setup completed - Experiment: {experiment_name}")
    
    def load_data(self):
        """Load the dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
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
            df_clean, missing_metrics = self.preprocessor.handle_missing_values(
                df, 
                preprocessing_config['missing_numerical'],
                preprocessing_config['missing_categorical']
            )
            print(f"   After missing value handling: {df_clean.shape}")
            
            # Step 3: Handle outliers
            if preprocessing_config['outlier_method'] != "none":
                print("   Handling outliers...")
                df_clean, outlier_metrics = self.preprocessor.handle_outliers_boxplot(
                    df_clean, 
                    preprocessing_config['outlier_method'],
                    preprocessing_config['outlier_threshold']
                )
                print(f"   After outlier handling: {df_clean.shape}")
            
            # Step 4: Reduce skewness
            if preprocessing_config['skewness_method'] != "none":
                print("   Reducing skewness...")
                df_clean, skew_metrics = self.preprocessor.reduce_skewness(
                    df_clean, preprocessing_config['skewness_method']
                )
                print(f"   After skewness reduction: {df_clean.shape}")
            
            # Step 5: Scale features
            if preprocessing_config['scaling_method'] != "none":
                print("   Scaling features...")
                df_clean, scale_metrics = self.preprocessor.scale_features(
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
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            
            # Feature importance
            feature_importance = model.get_feature_importance()
            top_features = sorted(zip(X.columns, feature_importance), 
                                key=lambda x: x[1], reverse=True)[:5]
            
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
                        "auc_score": results['auc_score'],
                        "cv_mean_accuracy": results['cv_mean_accuracy']
                    })
                    
                    # 🆕 LOG TOP 3 FEATURES TO MLFLOW
                    top_features = results['top_features']
                    for i, (feature, importance) in enumerate(top_features[:3]):
                        mlflow.log_param(f"top_feature_{i+1}_name", feature)
                        mlflow.log_metric(f"top_feature_{i+1}_importance", importance)
                    
                    print(f"✅ {run_name} - Accuracy: {results['accuracy']:.4f}, AUC: {results['auc_score']:.4f}")
                    
                    # Show top features
                    print(f"   🏆 Top 3 Features:")
                    for feature, importance in results['top_features'][:3]:
                        print(f"     - {feature}: {importance:.4f}")
                    
                    # Track best result
                    if results['accuracy'] > best_accuracy:
                        best_accuracy = results['accuracy']
                        best_params = param_combo
                        
                        if results['accuracy'] >= 0.95:
                            print("🎉 CONGRATULATIONS! 95% ACCURACY ACHIEVED! 🎉")
                            
                except Exception as e:
                    print(f"❌ {run_name} - Error: {str(e)}")
                    traceback.print_exc()
        
        # Final summary
        print(f"\n{'='*60}")
        print("🎯 CATBOOST OPTIMIZATION COMPLETED!")
        print(f"{'='*60}")
        
        if best_params:
            print(f"🏆 BEST CATBOOST CONFIG: {best_params['name']}")
            print(f"🎯 Accuracy: {best_accuracy:.4f}")
            print(f"⚙️ Parameters: {best_params['params']}")
            
            if best_accuracy >= 0.95:
                print("🎉🎉🎉 95% ACCURACY TARGET ACHIEVED! 🎉🎉🎉")
            else:
                improvement = (0.95 - best_accuracy) * 100
                print(f"📈 Need {improvement:.2f}% improvement to reach 95%")
        else:
            print("❌ No successful runs completed")
        
        print(f"\n📊 View results with: mlflow ui")

from mlflow_experiments.config.data_config import DATA_PATH
if __name__ == "__main__":
    data_path = DATA_PATH
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
    else:
        optimizer = OptimizeCatBoost(data_path)
        optimizer.run_catboost_optimization()