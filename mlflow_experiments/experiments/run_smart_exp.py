import mlflow
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.data_config import DataConfig
from config.experiment_config import SMART_COMBINATIONS
from pipeline.graph_nodes import DataProcessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC

class SmartExperimentRunner:
    def __init__(self, data_config: DataConfig):
        self.data_config = data_config
        self.processor = DataProcessor()
        self.setup_mlflow()
        
    def setup_mlflow(self):
        """Setup MLflow tracking"""
        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment("smart_preprocessing_experiments")
    
    def load_and_sample_data(self):
        """Load data and sample 2000 rows"""
        # Fix path issues
        data_path = self.data_config.DATA_PATH
        
        if not os.path.exists(data_path):
            print(f"Data file {data_path} not found. Creating sample data...")
            self.create_sample_data()
            
        df = pd.read_csv(data_path)
        sampled_df = df.sample(n=min(self.data_config.SAMPLE_SIZE, len(df)), random_state=42)
        print(f"Loaded data: {sampled_df.shape}")
        return sampled_df
    
    def create_sample_data(self):
        """Create sample data for testing"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.data_config.DATA_PATH), exist_ok=True)
        
        np.random.seed(42)
        n_samples = 5000
        
        # Create realistic heart disease data
        data = {
            'age': np.random.normal(55, 10, n_samples),  # Heart disease common in older age
            'blood_pressure': np.random.normal(130, 20, n_samples),
            'cholesterol': np.random.normal(200, 40, n_samples),
            'max_heart_rate': np.random.normal(150, 20, n_samples),
            'chest_pain_type': np.random.choice([0, 1, 2, 3], n_samples),  # 0: typical, 1: atypical, etc.
            'gender': np.random.choice([0, 1], n_samples),  # 0: female, 1: male
            'target': np.random.choice([0, 1], n_samples, p=[0.55, 0.45])  # 45% have heart disease
        }
        
        df = pd.DataFrame(data)
        
        # Add realistic missing values
        mask = np.random.random(n_samples) < 0.05
        df.loc[mask, 'cholesterol'] = np.nan
        mask = np.random.random(n_samples) < 0.03
        df.loc[mask, 'blood_pressure'] = np.nan
        
        # Add some duplicates
        df = pd.concat([df, df.head(50)], ignore_index=True)
        
        df.to_csv(self.data_config.DATA_PATH, index=False)
        print(f"Sample heart data created at {self.data_config.DATA_PATH}")
    
    def apply_preprocessing_pipeline(self, df, config):
        """Apply all preprocessing steps"""
        self.processor.set_target_column(self.data_config.TARGET_COLUMN)
        
        # Auto-detect numerical and categorical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if self.data_config.TARGET_COLUMN in numerical_cols:
            numerical_cols.remove(self.data_config.TARGET_COLUMN)
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Step 1: Handle duplicates
        df_processed, dup_metrics = self.processor.handle_duplicates(df, config['duplicate_handling'])
        
        # Step 2: Handle missing values
        df_processed, missing_metrics = self.processor.handle_missing_values(
            df_processed, 
            config['numerical_missing'], 
            config['categorical_missing'], 
            numerical_cols, 
            categorical_cols
        )
        
        # Step 3: Handle outliers - FIXED CALL
        outlier_config = config['outlier_handling']
        df_processed, outlier_metrics = self.processor.handle_outliers(
            df_processed, 
            outlier_config['method'], 
            numerical_cols, 
            **{k: v for k, v in outlier_config.items() if k != 'method'}
        )
                
        # Step 4: Reduce skewness
        df_processed, skew_metrics = self.processor.reduce_skewness(
            df_processed, config['skewness_reduction'], numerical_cols
        )
        
        # Step 5: Encode categorical
        df_processed, encode_metrics = self.processor.encode_categorical(
            df_processed, config['encoding'], categorical_cols
        )
        
        # Step 6: Scale features
        df_processed, scale_metrics = self.processor.scale_features(
            df_processed, config['scaling'], numerical_cols
        )
        
        # Combine all metrics
        all_metrics = {**dup_metrics, **missing_metrics, **outlier_metrics, 
                      **skew_metrics, **encode_metrics, **scale_metrics}
        
        return df_processed, all_metrics
    
    def train_multiple_models(self, X, y):
        """Train multiple models and return their performance"""
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=2000),  # ✅ Increased iterations
            # 'svm': SVC(random_state=42, probability=True)
        }
        
        results = {}
        
        for model_name, model in models.items():
            try:
                print(f"🤖 Training {model_name}...")
                
                # Cross-validation scores
                cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
                
                # Train-test split evaluation
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                results[model_name] = {
                    'cv_mean_accuracy': cv_scores.mean(),
                    'cv_std_accuracy': cv_scores.std(),
                    'test_accuracy': accuracy_score(y_test, y_pred),
                    'test_f1': f1_score(y_test, y_pred, average='weighted'),
                    'feature_count': X.shape[1]
                }
                
                print(f"✅ {model_name} - CV Accuracy: {cv_scores.mean():.4f}")
                
            except Exception as e:
                print(f"❌ {model_name} failed: {str(e)}")
                results[model_name] = None
        
        return results
    
    def run_smart_combinations(self):
        """Run only smart predefined combinations"""
        df_original = self.load_and_sample_data()
        
        if df_original is None:
            print("❌ Cannot proceed without data!")
            return
        
        print(f"🚀 Running {len(SMART_COMBINATIONS)} smart combinations...")
        
        best_score = 0
        best_combination = None
        
        for i, config in enumerate(SMART_COMBINATIONS):
            try:
                print(f"\n" + "="*50)
                print(f"🧪 Running combination {i+1}: {config['name']}")
                print("="*50)
                
                with mlflow.start_run(run_name=config['name']):
                    # Apply preprocessing
                    df_processed, preprocessing_metrics = self.apply_preprocessing_pipeline(
                        df_original.copy(), config
                    )
                    
                    if df_processed is None:
                        continue
                        
                    # Prepare for modeling
                    X = df_processed.drop(columns=[self.data_config.TARGET_COLUMN])
                    y = df_processed[self.data_config.TARGET_COLUMN]
                    
                    # Handle any remaining missing values
                    X = X.fillna(X.mean(numeric_only=True))
                    
                    # Remove any infinite values
                    X = X.replace([np.inf, -np.inf], np.nan)
                    X = X.fillna(X.mean(numeric_only=True))
                    
                    # Train and evaluate models
                    model_results = self.train_multiple_models(X, y)
                    
                    # Log parameters
                    mlflow.log_params(config)
                    
                    # ✅ FIX: Log only numeric metrics
                    numeric_metrics = {}
                    for key, value in preprocessing_metrics.items():
                        if isinstance(value, (int, float, np.number)):
                            numeric_metrics[key] = float(value)
                        else:
                            print(f"⚠️  Skipping non-numeric metric: {key} = {value}")
                    
                    mlflow.log_metrics(numeric_metrics)
                    
                    # Log model results
                    if model_results:
                        for model_name, metrics in model_results.items():
                            if metrics:
                                for metric_name, value in metrics.items():
                                    mlflow.log_metric(f"{model_name}_{metric_name}", value)
                                
                                # Track best combination
                                if model_name == 'random_forest' and metrics['cv_mean_accuracy'] > best_score:
                                    best_score = metrics['cv_mean_accuracy']
                                    best_combination = config['name']
                    
                    # Log dataset info
                    mlflow.log_metrics({
                        "final_rows": df_processed.shape[0],
                        "final_columns": df_processed.shape[1],
                        "final_features": X.shape[1]
                    })
                    
                    print(f"✅ Completed: {config['name']}")
                    
            except Exception as e:
                print(f"❌ Failed {config['name']}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        if best_combination:
            print(f"\n🏆 BEST COMBINATION: {best_combination} with score: {best_score:.4f}")
        else:
            print(f"\n❌ No successful experiments!")



if __name__ == "__main__":
    # Initialize with YOUR data configuration
    config = DataConfig()
    config.DATA_PATH = r"F:\18. MAJOR PROJECT\Data\DATA_FOR_HEART_RISK_PRED.csv"
    config.TARGET_COLUMN = "heart_flag"  # ✅ YOUR TARGET COLUMN
    config.SAMPLE_SIZE = 10000
    
    runner = SmartExperimentRunner(config)
    runner.run_smart_combinations()
    
    print("\n🎉 Smart experiments completed!")
    print("Run 'mlflow ui' to view results")