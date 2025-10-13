import mlflow
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from mlflow_experiments.config.data_config import *
from pipeline.preprocessor import BoxPlotPreprocessor



# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



class OptimizeBestCombo:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.preprocessor = BoxPlotPreprocessor()
        self.setup_mlflow()
    
    def setup_mlflow(self):
        """Setup MLflow tracking"""
        mlflow.set_tracking_uri("mlruns")
        
        # Dynamic name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        experiment_name = f"2. OPTIMIZE_XGBOOST_{timestamp}"
        
        mlflow.set_experiment(experiment_name)
        print(f"MLflow setup completed - Experiment: {experiment_name}")
    
    def load_data(self):
        """Load the dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded: {self.df.shape}")
        print(f"Target distribution: {self.df['heart_attack'].value_counts().to_dict()}")
        return self.df
    
    def apply_best_preprocessing(self, df):
        """Apply the best preprocessing combination found"""
        # Step 1: Identify columns
        numerical_cols, categorical_cols = self.preprocessor.identify_columns(df)
        
        # Step 2: Handle missing values (mean for numerical, mode for categorical)
        df_clean, _ = self.preprocessor.handle_missing_values(
            df, "mean", "mode"
        )
        
        # Step 3: Remove outliers with 1.5 IQR
        df_clean, _ = self.preprocessor.handle_outliers_boxplot(
            df_clean, "remove", 1.5
        )
        
        # Step 4: No skewness reduction
        df_clean, _ = self.preprocessor.reduce_skewness(df_clean, "none")
        
        # Step 5: No scaling
        df_clean, _ = self.preprocessor.scale_features(df_clean, "none")
        
        # Step 6: Encode categorical
        df_clean = self.preprocessor.encode_categorical(df_clean)
        
        return df_clean
    
    def train_and_evaluate(self, X, y, model_params):
        """Train and evaluate XGBoost with given parameters"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize and train model
        model = xgb.XGBClassifier(**model_params)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_mean_accuracy': cv_scores.mean(),
            'cv_std_accuracy': cv_scores.std(),
            'model': model
        }
    
    def run_optimization_experiments(self):
        """Run optimization experiments to reach 95% accuracy"""
        print("🚀 STARTING OPTIMIZATION FOR 95% ACCURACY")
        print("=" * 60)
        
        self.load_data()
        
        # Apply best preprocessing
        print("🔧 Applying best preprocessing combination...")
        df_processed = self.apply_best_preprocessing(self.df)
        print(f"✅ Processed data: {df_processed.shape}")
        
        # Prepare features and target
        X = df_processed.drop(columns=['heart_attack'])
        y = df_processed['heart_attack']
        
        # XGBoost parameter combinations to try
        param_combinations = [
            # Basic tuning
            {
                "name": "basic_tuned",
                "params": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "random_state": 42
                }
            },
            # Increased complexity
            {
                "name": "increased_trees",
                "params": {
                    "n_estimators": 200,
                    "max_depth": 8,
                    "learning_rate": 0.1,
                    "random_state": 42
                }
            },
            # Deeper trees
            {
                "name": "deeper_trees", 
                "params": {
                    "n_estimators": 150,
                    "max_depth": 10,
                    "learning_rate": 0.1,
                    "random_state": 42
                }
            },
            # Lower learning rate
            {
                "name": "slow_learning",
                "params": {
                    "n_estimators": 300,
                    "max_depth": 6,
                    "learning_rate": 0.05,
                    "random_state": 42
                }
            },
            # With regularization
            {
                "name": "with_regularization",
                "params": {
                    "n_estimators": 200,
                    "max_depth": 8,
                    "learning_rate": 0.1,
                    "reg_alpha": 0.1,
                    "reg_lambda": 0.1,
                    "random_state": 42
                }
            },
            # Subsample and colsample
            {
                "name": "feature_sampling",
                "params": {
                    "n_estimators": 200,
                    "max_depth": 8,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": 42
                }
            },
            # Comprehensive tuning
            {
                "name": "comprehensive",
                "params": {
                    "n_estimators": 500,
                    "max_depth": 12,
                    "learning_rate": 0.01,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_alpha": 0.1,
                    "reg_lambda": 0.1,
                    "random_state": 42
                }
            },
            # Medical data specific
            {
                "name": "medical_optimized",
                "params": {
                    "n_estimators": 1000,
                    "max_depth": 15,
                    "learning_rate": 0.005,
                    "subsample": 0.7,
                    "colsample_bytree": 0.7,
                    "reg_alpha": 0.2,
                    "reg_lambda": 0.2,
                    "scale_pos_weight": 1,
                    "random_state": 42
                }
            },
            # High capacity
            {
                "name": "high_capacity",
                "params": {
                    "n_estimators": 800,
                    "max_depth": 20,
                    "learning_rate": 0.01,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                    "reg_alpha": 0.05,
                    "reg_lambda": 0.05,
                    "random_state": 42
                }
            },
            # Balanced
            {
                "name": "balanced",
                "params": {
                    "n_estimators": 400,
                    "max_depth": 10,
                    "learning_rate": 0.1,
                    "subsample": 0.85,
                    "colsample_bytree": 0.85,
                    "reg_alpha": 0.15,
                    "reg_lambda": 0.15,
                    "random_state": 42
                }
            }
        ]
        
        best_accuracy = 0
        best_params = None
        best_run_name = None
        
        for param_combo in param_combinations:
            run_name = f"xgb_optimize_{param_combo['name']}"
            
            with mlflow.start_run(run_name=run_name):
                try:
                    print(f"🔬 Testing: {run_name}")
                    
                    # Log parameters
                    mlflow.log_params({
                        "preprocessing": "best_combo_152",
                        "missing_numerical": "mean",
                        "missing_categorical": "mode", 
                        "outlier_method": "remove",
                        "outlier_threshold": 1.5,
                        "skewness_method": "none",
                        "scaling_method": "none"
                    })
                    
                    # Log XGBoost parameters
                    for param_name, param_value in param_combo['params'].items():
                        mlflow.log_param(f"xgb_{param_name}", param_value)
                    
                    # Train and evaluate
                    results = self.train_and_evaluate(X, y, param_combo['params'])
                    
                    if results:
                        # Log metrics
                        mlflow.log_metrics({
                            "accuracy": results['accuracy'],
                            "precision": results['precision'],
                            "recall": results['recall'],
                            "f1_score": results['f1_score'],
                            "cv_mean_accuracy": results['cv_mean_accuracy']
                        })
                        
                        print(f"{run_name} - Accuracy: {results['accuracy']:.4f}, CV: {results['cv_mean_accuracy']:.4f}")
                        
                        # Track best result
                        if results['accuracy'] > best_accuracy:
                            best_accuracy = results['accuracy']
                            best_params = param_combo
                            best_run_name = run_name
                            
                    else:
                        print(f" {run_name} - Failed")
                        
                except Exception as e:
                    print(f" {run_name} - Error: {str(e)}")
        

        print(" OPTIMIZATION COMPLETED!")
        
        if best_params:
            print(f"BEST RESULT: {best_run_name}")
            print(f"Accuracy: {best_accuracy:.4f}")
            print(f"Parameters: {best_params['params']}")
            
            if best_accuracy >= 0.95:
                print("CONGRATULATIONS! 95% ACCURACY ACHIEVED! ")
    
if __name__ == "__main__":
    data_path = DATA_PATH
    
    # Validate data path
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Please check the file path and try again.")
    else:
        optimizer = OptimizeBestCombo(data_path)
        optimizer.run_optimization_experiments()