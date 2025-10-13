import mlflow
import pandas as pd
import numpy as np
import os
import sys
import traceback
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.preprocessor import BoxPlotPreprocessor

class OptimizeLightGBM:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.preprocessor = BoxPlotPreprocessor()
        self.setup_mlflow()
    
    def setup_mlflow(self):
        """Setup MLflow tracking"""
        mlflow.set_tracking_uri("mlruns")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        experiment_name = f"6. ADVANCED_LIGHTGBM_{timestamp}"

        mlflow.set_experiment(experiment_name)
        print(f"MLflow setup completed - Experiment: {experiment_name}")
    
    def load_data(self):
        """Load the dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f" Dataset loaded: {self.df.shape}")
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
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            
            # Feature importance
            feature_importance = model.feature_importances_
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
    
    def run_lightgbm_optimization(self):
        """Run LightGBM optimization experiments"""
        print("🚀 OPTIMIZING LIGHTGBM FOR 95% ACCURACY")
        print("=" * 60)
        
        self.load_data()
        
        # Define the specific preprocessing configuration (from combo_015)
        preprocessing_config = {
            'missing_numerical': 'median',
            'missing_categorical': 'mode', 
            'outlier_method': 'remove',
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
            {
                "name": "current_optimized",
                "params": {
                "n_estimators": 2000,  # Increase from 1500
                "max_depth": 25,       # Increase from 20  
                "learning_rate": 0.005, # More conservative
                "num_leaves": 128,     # Optimized
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "reg_alpha": 0.3,
                "reg_lambda": 0.3,
                "min_child_samples": 5, # Reduced for more flexibility
                "random_state": 42
            }
            },
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
                        "auc_score": results['auc_score'],
                        "cv_mean_accuracy": results['cv_mean_accuracy']
                    })
                    
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
                    print(f"   🎪 CV Mean: {results['cv_mean_accuracy']:.4f}")
                    
                    # Show top features
                    print(f"   🏆 Top 3 Features:")
                    for feature, importance in results['top_features'][:3]:
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

from mlflow_experiments.config.data_config import DATA_PATH
if __name__ == "__main__":
    data_path = DATA_PATH
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
    else:
        optimizer = OptimizeLightGBM(data_path)
        optimizer.run_lightgbm_optimization()