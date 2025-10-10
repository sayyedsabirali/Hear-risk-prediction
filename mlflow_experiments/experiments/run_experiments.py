import mlflow
import pandas as pd
import numpy as np
from itertools import product
import os
import sys

# Add parent directory to path to import configs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.data_config import DataConfig
from config.experiment_config import EXPERIMENT_CONFIG
from pipeline.graph_nodes import DataProcessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score

class MLflowExperimentRunnerWithModel:
    def __init__(self, data_config: DataConfig):
        self.data_config = data_config
        self.processor = DataProcessor()
        self.setup_mlflow()
        
    def setup_mlflow(self):
        """Setup MLflow tracking"""
        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment("data_preprocessing_experiments")
    
    def load_and_sample_data(self):
        """Load data and sample 2000 rows"""
        # Create sample data if doesn't exist
        if not os.path.exists(self.data_config.DATA_PATH):
            print(f"Data file {self.data_config.DATA_PATH} not found. Creating sample data...")
            self.create_sample_data()
            
        df = pd.read_csv(self.data_config.DATA_PATH)
        sampled_df = df.sample(n=min(self.data_config.SAMPLE_SIZE, len(df)), random_state=42)
        print(f"Loaded data: {sampled_df.shape}")
        return sampled_df
    
    def create_sample_data(self):
        """Create sample data for testing"""
        os.makedirs("data", exist_ok=True)
        np.random.seed(42)
        
        n_samples = 5000
        data = {
            'age': np.random.normal(35, 10, n_samples),
            'income': np.random.lognormal(10, 1, n_samples),
            'credit_score': np.random.normal(650, 100, n_samples),
            'city': np.random.choice(['New York', 'London', 'Tokyo', 'Paris', 'Unknown'], n_samples),
            'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
            'target': np.random.randint(0, 2, n_samples)
        }
        
        # Add some missing values
        df = pd.DataFrame(data)
        mask = np.random.random(n_samples) < 0.1
        df.loc[mask, 'age'] = np.nan
        mask = np.random.random(n_samples) < 0.05
        df.loc[mask, 'income'] = np.nan
        mask = np.random.random(n_samples) < 0.08  
        df.loc[mask, 'city'] = np.nan
        
        # Add some duplicates
        df = pd.concat([df, df.head(100)], ignore_index=True)
        
        df.to_csv(self.data_config.DATA_PATH, index=False)
        print(f"Sample data created at {self.data_config.DATA_PATH}")
    
    def apply_preprocessing_pipeline(self, df, preprocessing_config):
        """Apply all preprocessing steps"""
        # Set target column for processor
        self.processor.set_target_column(self.data_config.TARGET_COLUMN)
        
        # Auto-detect numerical and categorical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if self.data_config.TARGET_COLUMN in numerical_cols:
            numerical_cols.remove(self.data_config.TARGET_COLUMN)
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Step 1: Handle duplicates
        df_processed, _ = self.processor.handle_duplicates(df, preprocessing_config['duplicate_method'])
        
        # Step 2: Handle missing values
        df_processed, _ = self.processor.handle_missing_values(
            df_processed, 
            preprocessing_config['numerical_missing'], 
            preprocessing_config['categorical_missing'], 
            numerical_cols, 
            categorical_cols
        )
        
        # Step 3: Handle outliers
        outlier_config = preprocessing_config['outlier_config']
        df_processed, _ = self.processor.handle_outliers(
            df_processed, outlier_config['method'], numerical_cols, **outlier_config
        )
        
        # Step 4: Reduce skewness
        df_processed, _ = self.processor.reduce_skewness(
            df_processed, preprocessing_config['skewness_method'], numerical_cols
        )
        
        # Step 5: Encode categorical
        df_processed, _ = self.processor.encode_categorical(
            df_processed, preprocessing_config['encoding_method'], categorical_cols
        )
        
        # Step 6: Scale features
        df_processed, _ = self.processor.scale_features(
            df_processed, preprocessing_config['scaling_method'], numerical_cols
        )
        
        return df_processed
    
    def train_and_evaluate_models(self, X, y):
        """Train multiple models and return their performance"""
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        results = {}
        
        for model_name, model in models.items():
            try:
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
                    'test_f1': f1_score(y_test, y_pred, average='weighted')
                }
            except Exception as e:
                print(f"Model {model_name} failed: {str(e)}")
                results[model_name] = None
        
        return results
    
    def run_experiments(self):
        """Run all combinations of preprocessing steps with model training"""
        df_original = self.load_and_sample_data()
        
        # Generate all combinations
        combinations = list(product(
            EXPERIMENT_CONFIG["duplicate_handling"],
            EXPERIMENT_CONFIG["missing_value_handling"]["numerical"],
            EXPERIMENT_CONFIG["missing_value_handling"]["categorical"],
            EXPERIMENT_CONFIG["outlier_handling"],
            EXPERIMENT_CONFIG["skewness_reduction"],
            EXPERIMENT_CONFIG["encoding"],
            EXPERIMENT_CONFIG["scaling"]
        ))
        
        print(f"Total combinations to run: {len(combinations)}")
        
        for i, (dup_config, num_missing, cat_missing, outlier_config, 
                skew_config, encode_config, scale_config) in enumerate(combinations[:10]):  # Limit to 10 for testing
            
            try:
                preprocessing_config = {
                    'duplicate_method': dup_config['method'],
                    'numerical_missing': num_missing,
                    'categorical_missing': cat_missing,
                    'outlier_config': outlier_config,
                    'skewness_method': skew_config['method'],
                    'encoding_method': encode_config['method'],
                    'scaling_method': scale_config['method']
                }
                
                self.run_single_experiment(df_original.copy(), preprocessing_config, i)
                print(f"✅ Completed experiment {i+1}/10")
                
            except Exception as e:
                print(f"❌ Failed experiment {i+1}: {str(e)}")
                continue
    
    def run_single_experiment(self, df, preprocessing_config, run_id):
        """Run single experiment with MLflow tracking"""
        
        with mlflow.start_run(run_name=f"exp_{run_id}"):
            # Log preprocessing parameters
            mlflow.log_params({
                "duplicate_handling": preprocessing_config['duplicate_method'],
                "numerical_missing": preprocessing_config['numerical_missing'],
                "categorical_missing": preprocessing_config['categorical_missing'],
                "outlier_handling": preprocessing_config['outlier_config']['method'],
                "skewness_reduction": preprocessing_config['skewness_method'],
                "encoding": preprocessing_config['encoding_method'],
                "scaling": preprocessing_config['scaling_method']
            })
            
            # Apply preprocessing pipeline
            df_processed = self.apply_preprocessing_pipeline(df, preprocessing_config)
            
            # Prepare data for modeling
            if self.data_config.TARGET_COLUMN not in df_processed.columns:
                print(f"Target column {self.data_config.TARGET_COLUMN} not found after preprocessing")
                return
                
            X = df_processed.drop(columns=[self.data_config.TARGET_COLUMN])
            y = df_processed[self.data_config.TARGET_COLUMN]
            
            # Handle any remaining missing values
            X = X.fillna(X.mean())
            
            # Train and evaluate models
            model_results = self.train_and_evaluate_models(X, y)
            
            # Log model performance metrics
            if model_results:
                for model_name, metrics in model_results.items():
                    if metrics:
                        for metric_name, value in metrics.items():
                            mlflow.log_metric(f"{model_name}_{metric_name}", value)
            
            # Log final dataset info
            mlflow.log_metrics({
                "final_rows": df_processed.shape[0],
                "final_columns": df_processed.shape[1],
                "final_features": X.shape[1]
            })
            
            # Save processed data
            output_dir = "processed_data"
            os.makedirs(output_dir, exist_ok=True)
            output_path = f"{output_dir}/experiment_{run_id}.csv"
            df_processed.to_csv(output_path, index=False)
            mlflow.log_artifact(output_path)

if __name__ == "__main__":
    # Initialize configuration
    config = DataConfig()
    
    # Create and run experiments
    runner = MLflowExperimentRunnerWithModel(config)
    runner.run_experiments()
    
    print("🎉 All experiments completed!")
    print("Run 'mlflow ui' to view results in MLflow UI")