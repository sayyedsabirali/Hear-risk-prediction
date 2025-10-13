import mlflow
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.preprocessor import BoxPlotPreprocessor

class FinalOptimization:
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
        experiment_name = f"3. AGGRESIVE_OPTIMIZED_XGBOOST_{timestamp}"
        
        mlflow.set_experiment(experiment_name)
        print(f"MLflow setup completed - Experiment: {experiment_name}")
    
    def load_data(self):
        """Load the dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded: {self.df.shape}")
        print(f"Target distribution: {self.df['heart_attack'].value_counts().to_dict()}")
        return self.df
    
    def apply_preprocessing_variations(self, variation="best"):
        """Apply different preprocessing variations"""
        df_temp = self.df.copy()
        
        # Identify columns
        numerical_cols, categorical_cols = self.preprocessor.identify_columns(df_temp)
        
        if variation == "best":
            # Original best preprocessing
            df_clean, _ = self.preprocessor.handle_missing_values(df_temp, "mean", "mode")
            df_clean, _ = self.preprocessor.handle_outliers_boxplot(df_clean, "remove", 1.5)
            df_clean, _ = self.preprocessor.reduce_skewness(df_clean, "none")
            df_clean, _ = self.preprocessor.scale_features(df_clean, "none")
            
        elif variation == "no_outlier_remove":
            # Don't remove outliers, just cap them
            df_clean, _ = self.preprocessor.handle_missing_values(df_temp, "mean", "mode")
            df_clean, _ = self.preprocessor.handle_outliers_boxplot(df_clean, "clipper", 2.0)
            df_clean, _ = self.preprocessor.reduce_skewness(df_clean, "none")
            df_clean, _ = self.preprocessor.scale_features(df_clean, "none")
            
        elif variation == "with_scaling":
            # Add scaling
            df_clean, _ = self.preprocessor.handle_missing_values(df_temp, "mean", "mode")
            df_clean, _ = self.preprocessor.handle_outliers_boxplot(df_clean, "remove", 1.5)
            df_clean, _ = self.preprocessor.reduce_skewness(df_clean, "none")
            df_clean, _ = self.preprocessor.scale_features(df_clean, "standard")
            
        elif variation == "knn_impute":
            # Use KNN for missing values
            df_clean, _ = self.preprocessor.handle_missing_values(df_temp, "knn", "mode")
            df_clean, _ = self.preprocessor.handle_outliers_boxplot(df_clean, "remove", 1.5)
            df_clean, _ = self.preprocessor.reduce_skewness(df_clean, "none")
            df_clean, _ = self.preprocessor.scale_features(df_clean, "none")
            
        elif variation == "full_data":
            # Keep all data (no outlier removal)
            df_clean, _ = self.preprocessor.handle_missing_values(df_temp, "mean", "mode")
            # No outlier removal to keep maximum data
            df_clean, _ = self.preprocessor.reduce_skewness(df_clean, "none")
            df_clean, _ = self.preprocessor.scale_features(df_clean, "none")
        
        # Encode categorical
        df_clean = self.preprocessor.encode_categorical(df_clean)
        
        return df_clean
    
    def train_and_evaluate_advanced(self, X, y, model_params, preprocessing_variation):
        """Train and evaluate with advanced techniques"""
        # Use stratified K-fold for better validation
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize model
        model = xgb.XGBClassifier(**model_params)
        
        # Cross-validation on training data
        cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
        
        # Train on full training data
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        
        # Additional metrics for threshold tuning
        from sklearn.metrics import roc_auc_score
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc_score,
            'cv_mean_accuracy': cv_scores.mean(),
            'cv_std_accuracy': cv_scores.std(),
            'model': model,
            'test_size': len(X_test),
            'train_size': len(X_train)
        }
    
    def run_final_optimization(self):
        """Run final optimization experiments to reach 95%"""
        print("🚀 FINAL OPTIMIZATION FOR 95% ACCURACY")
        print("=" * 60)
        
        self.load_data()
        
        # Advanced parameter combinations
        advanced_param_combinations = [
            # Ultra High Capacity
            {
                "name": "ultra_high_capacity",
                "params": {
                    "n_estimators": 1000,
                    "max_depth": 15,
                    "learning_rate": 0.005,
                    "subsample": 0.7,
                    "colsample_bytree": 0.7,
                    "colsample_bylevel": 0.7,
                    "reg_alpha": 0.3,
                    "reg_lambda": 0.3,
                    "gamma": 0.1,
                    "min_child_weight": 3,
                    "random_state": 42
                }
            },
            # Aggressive Learning
            {
                "name": "aggressive_learning",
                "params": {
                    "n_estimators": 800,
                    "max_depth": 20,
                    "learning_rate": 0.01,
                    "subsample": 0.6,
                    "colsample_bytree": 0.6,
                    "reg_alpha": 0.5,
                    "reg_lambda": 0.5,
                    "gamma": 0.2,
                    "min_child_weight": 5,
                    "random_state": 42
                }
            },
            # Conservative but Deep
            {
                "name": "conservative_deep",
                "params": {
                    "n_estimators": 1500,
                    "max_depth": 25,
                    "learning_rate": 0.001,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_alpha": 0.1,
                    "reg_lambda": 0.1,
                    "gamma": 0.05,
                    "min_child_weight": 1,
                    "random_state": 42
                }
            },
            # Best from previous + enhancements
            {
                "name": "enhanced_comprehensive",
                "params": {
                    "n_estimators": 800,
                    "max_depth": 15,
                    "learning_rate": 0.008,
                    "subsample": 0.75,
                    "colsample_bytree": 0.75,
                    "colsample_bylevel": 0.75,
                    "reg_alpha": 0.2,
                    "reg_lambda": 0.2,
                    "gamma": 0.1,
                    "min_child_weight": 2,
                    "random_state": 42
                }
            },
            # Medical Data Specialized
            {
                "name": "medical_specialized",
                "params": {
                    "n_estimators": 1200,
                    "max_depth": 18,
                    "learning_rate": 0.003,
                    "subsample": 0.65,
                    "colsample_bytree": 0.65,
                    "colsample_bylevel": 0.65,
                    "reg_alpha": 0.4,
                    "reg_lambda": 0.4,
                    "gamma": 0.15,
                    "min_child_weight": 4,
                    "scale_pos_weight": 1,
                    "random_state": 42
                }
            }
        ]
        
        # Preprocessing variations to test
        preprocessing_variations = [
            "best",
            "no_outlier_remove", 
            "with_scaling",
            "knn_impute",
            "full_data"
        ]
        
        best_accuracy = 0
        best_combo = None
        results = []
        
        for preprocess_var in preprocessing_variations:
            print(f"\n🔧 Testing preprocessing: {preprocess_var}")
            
            # Apply preprocessing
            df_processed = self.apply_preprocessing_variations(preprocess_var)
            print(f"📊 Processed data shape: {df_processed.shape}")
            
            # Skip if too little data
            if len(df_processed) < 1000:
                print(f"❌ Skipping {preprocess_var} - insufficient data")
                continue
            
            # Prepare features and target
            X = df_processed.drop(columns=['heart_attack'])
            y = df_processed['heart_attack']
            
            for param_combo in advanced_param_combinations:
                run_name = f"{preprocess_var}_{param_combo['name']}"
                
                with mlflow.start_run(run_name=run_name):
                    try:
                        print(f"🔬 Testing: {run_name}")
                        
                        # Log preprocessing parameters
                        mlflow.log_params({
                            "preprocessing_variation": preprocess_var,
                            "missing_numerical": "mean",
                            "missing_categorical": "mode",
                            "outlier_method": "remove",
                            "outlier_threshold": 1.5,
                            "skewness_method": "none",
                            "scaling_method": "none",
                            "final_data_rows": len(df_processed)
                        })
                        
                        # Log XGBoost parameters
                        for param_name, param_value in param_combo['params'].items():
                            mlflow.log_param(f"xgb_{param_name}", param_value)
                        
                        # Train and evaluate
                        results_dict = self.train_and_evaluate_advanced(
                            X, y, param_combo['params'], preprocess_var
                        )
                        
                        if results_dict:
                            # Log metrics
                            mlflow.log_metrics({
                                "accuracy": results_dict['accuracy'],
                                "precision": results_dict['precision'],
                                "recall": results_dict['recall'],
                                "f1_score": results_dict['f1_score'],
                                "auc_score": results_dict['auc_score'],
                                "cv_mean_accuracy": results_dict['cv_mean_accuracy'],
                                "test_size": results_dict['test_size'],
                                "train_size": results_dict['train_size']
                            })
                            
                            accuracy = results_dict['accuracy']
                            print(f"✅ {run_name} - Accuracy: {accuracy:.4f}, AUC: {results_dict['auc_score']:.4f}")
                            
                            # Track results
                            results.append({
                                'run_name': run_name,
                                'preprocessing': preprocess_var,
                                'accuracy': accuracy,
                                'auc_score': results_dict['auc_score'],
                                'cv_mean': results_dict['cv_mean_accuracy'],
                                'data_rows': len(df_processed),
                                'params': param_combo['params']
                            })
                            
                            # Track best result
                            if accuracy > best_accuracy:
                                best_accuracy = accuracy
                                best_combo = {
                                    'run_name': run_name,
                                    'preprocessing': preprocess_var,
                                    'accuracy': accuracy,
                                    'params': param_combo['params'],
                                    'data_rows': len(df_processed)
                                }
                                
                                if accuracy >= 0.95:
                                    print("🎉 CONGRATULATIONS! 95% ACCURACY ACHIEVED! 🎉")
                                    print(f"🏆 Winning combination: {run_name}")
                                    
                        else:
                            print(f"❌ {run_name} - Training failed")
                            
                    except Exception as e:
                        print(f"❌ {run_name} - Error: {str(e)}")
        
        # Final analysis
        print(f"\n{'='*60}")
        print("🎯 FINAL OPTIMIZATION COMPLETED!")
        print(f"{'='*60}")
        
        if best_combo:
            print(f"🏆 BEST OVERALL RESULT: {best_combo['run_name']}")
            print(f"🎯 Accuracy: {best_accuracy:.4f}")
            print(f"🔧 Preprocessing: {best_combo['preprocessing']}")
            print(f"📊 Data Rows: {best_combo['data_rows']:,}")
            print(f"⚙️ Key Parameters:")
            for param, value in list(best_combo['params'].items())[:5]:  # Show first 5 params
                print(f"   - {param}: {value}")
            
            if best_accuracy >= 0.95:
                print("\n🎉🎉🎉 95% ACCURACY TARGET ACHIEVED! 🎉🎉🎉")
            else:
                improvement_needed = (0.95 - best_accuracy) * 100
                print(f"\n📈 Need {improvement_needed:.2f}% improvement to reach 95%")
        
        # Show top 5 results
        if results:
            results_df = pd.DataFrame(results)
            top_5 = results_df.nlargest(5, 'accuracy')
            
            print(f"\n📈 TOP 5 COMBINATIONS:")
            for i, row in top_5.iterrows():
                print(f"{i+1}. {row['run_name']} - Acc: {row['accuracy']:.4f}, Data: {row['data_rows']:,}")
        
        print(f"\n📊 View detailed results with: mlflow ui")
from mlflow_experiments.config.data_config import *

if __name__ == "__main__":
    data_path = DATA_PATH
    
    # Validate data path
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("Please check the file path and try again.")
    else:
        optimizer = FinalOptimization(data_path)
        optimizer.run_final_optimization()