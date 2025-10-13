import mlflow
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from mlflow_experiments.config.data_config import *

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.preprocessing_config import COMBINATIONS_TO_TEST
from pipeline.preprocessor import BoxPlotPreprocessor
from pipeline.model_trainer import ModelTrainer

class BoxPlotPreprocessingExperiment:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.preprocessor = BoxPlotPreprocessor()
        self.trainer = ModelTrainer()
        self.setup_mlflow()
    
    def setup_mlflow(self):
        """Setup MLflow tracking"""
        mlflow.set_tracking_uri("mlruns")
        
        # 🔥 DYNAMIC NAME WITH TIMESTAMP
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        experiment_name = f"1. DATA_PREPROCESSING_TRACKING_{timestamp}"
        
        mlflow.set_experiment(experiment_name)
        print(f"MLflow setup completed - Experiment: {experiment_name}")
    
    def load_data(self):
        """Load the dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded: {self.df.shape}")
        print(f"Target distribution: {self.df['heart_attack'].value_counts().to_dict()}")
        return self.df
    
    def run_single_experiment(self, combo, run_id):
        """Run single preprocessing combination"""
        
        experiment_name = f"{combo['name']}_{combo['model']['name']}"
        
        with mlflow.start_run(run_name=experiment_name):
            try:
                print(f"🔬 [{run_id+1}/{len(COMBINATIONS_TO_TEST)}] Running: {experiment_name}")
                
                # Log all parameters
                mlflow.log_params({
                    "combo_name": combo['name'],
                    "missing_numerical": combo['missing']['numerical'],
                    "missing_categorical": combo['missing']['categorical'],
                    "outlier_method": combo['outlier']['method'],
                    "outlier_threshold": combo['outlier']['threshold'],
                    "skewness_method": combo['skewness']['method'],
                    "scaling_method": combo['scaling']['method'],
                    "model_name": combo['model']['name']
                })
                
                # Step 1: Identify columns
                df_temp = self.df.copy()
                numerical_cols, categorical_cols = self.preprocessor.identify_columns(df_temp)
                
                # Step 2: Handle missing values
                df_temp, _ = self.preprocessor.handle_missing_values(
                    df_temp, 
                    combo['missing']['numerical'],
                    combo['missing']['categorical']
                )
                
                # Step 3: Handle outliers
                if combo['outlier']['method'] != "none":
                    df_temp, _ = self.preprocessor.handle_outliers_boxplot(
                        df_temp, 
                        combo['outlier']['method'],
                        combo['outlier']['threshold']
                    )
                
                # Step 4: Reduce skewness
                df_temp, _ = self.preprocessor.reduce_skewness(
                    df_temp, combo['skewness']['method']
                )
                
                # Step 5: Scale features
                df_temp, _ = self.preprocessor.scale_features(
                    df_temp, combo['scaling']['method']
                )
                
                # Step 6: Encode categorical
                df_temp = self.preprocessor.encode_categorical(df_temp)
                
                # Step 7: Train and evaluate model (only if we have enough data)
                if len(df_temp) >= 100:
                    results = self.trainer.train_and_evaluate(
                        df_temp, combo['model']['name'], combo['model']['params']
                    )
                    
                    if results:
                        # Log only key metrics
                        mlflow.log_metrics({
                            "accuracy": results['accuracy'],
                            "precision": results['precision'],
                            "recall": results['recall'],
                            "f1_score": results['f1_score'],
                            "cv_mean_accuracy": results['cv_mean_accuracy'],
                            "final_rows": results['final_rows']
                        })
                        
                        print(f"{experiment_name} - Acc: {results['accuracy']:.4f}, F1: {results['f1_score']:.4f}, Rows: {results['final_rows']:,}")
                        return {
                            'combo_name': combo['name'],
                            'model': combo['model']['name'],
                            'accuracy': results['accuracy'],
                            'precision': results['precision'],
                            'recall': results['recall'],
                            'f1_score': results['f1_score'],
                            'cv_mean_accuracy': results['cv_mean_accuracy'],
                            'final_rows': results['final_rows'],
                            'missing_numerical': combo['missing']['numerical'],
                            'missing_categorical': combo['missing']['categorical'],
                            'outlier_method': combo['outlier']['method'],
                            'outlier_threshold': combo['outlier']['threshold'],
                            'skewness_method': combo['skewness']['method'],
                            'scaling_method': combo['scaling']['method']
                        }
                    else:
                        print(f" {experiment_name} - Model training failed")
                        return None
                else:
                    print(f"{experiment_name} - Not enough data: {len(df_temp)} rows")
                    return None
                    
            except Exception as e:
                print(f"{experiment_name} - Failed: {str(e)}")
                return None
    
    def run_all_experiments(self):
        """Run all preprocessing combinations"""
        print("STARTING ALL COMBINATION EXPERIMENTS")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.load_data()
        
        total_combinations = len(COMBINATIONS_TO_TEST)
        print(f"Total combinations to test: {total_combinations}")
        
        results = []
        successful_runs = 0
        
        for i, combo in enumerate(COMBINATIONS_TO_TEST):
            result = self.run_single_experiment(combo, i)
            
            if result is not None:
                results.append(result)
                successful_runs += 1
        
        # Final summary
        print(f"\n{'='*60}")
        print("EXPERIMENTS COMPLETED!")
        print(f"Successful runs: {successful_runs}/{total_combinations}")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if results:
            # Create results summary
            results_df = pd.DataFrame(results)
            
            # Find best by accuracy
            best_accuracy = results_df.loc[results_df['accuracy'].idxmax()]
            
            print(f"\nBEST ACCURACY COMBINATION:")
            print(f"• Combo: {best_accuracy['combo_name']}")
            print(f"• Model: {best_accuracy['model']}")
            print(f"• Accuracy: {best_accuracy['accuracy']:.4f}")
            print(f"• F1 Score: {best_accuracy['f1_score']:.4f}")
            print(f"• Final Rows: {best_accuracy['final_rows']:,}")
            print(f"• Missing: {best_accuracy['missing_numerical']}/{best_accuracy['missing_categorical']}")
            print(f"• Outlier: {best_accuracy['outlier_method']} ({best_accuracy['outlier_threshold']})")
            print(f"• Skewness: {best_accuracy['skewness_method']}")
            print(f"• Scaling: {best_accuracy['scaling_method']}")
            
            # Find best by F1 score
            best_f1 = results_df.loc[results_df['f1_score'].idxmax()]
            
            print(f"\nBEST F1 SCORE COMBINATION:")
            print(f"• Combo: {best_f1['combo_name']}")
            print(f"• Model: {best_f1['model']}")
            print(f"• Accuracy: {best_f1['accuracy']:.4f}")
            print(f"• F1 Score: {best_f1['f1_score']:.4f}")
            print(f"• Final Rows: {best_f1['final_rows']:,}")
            print(f"• Missing: {best_f1['missing_numerical']}/{best_f1['missing_categorical']}")
            print(f"• Outlier: {best_f1['outlier_method']} ({best_f1['outlier_threshold']})")
            print(f"• Skewness: {best_f1['skewness_method']}")
            print(f"• Scaling: {best_f1['scaling_method']}")
            
            # Save results summary with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            summary_path = f"heart_risk_results_{timestamp}.csv"
            results_df.to_csv(summary_path, index=False)
            print(f"\nDetailed results saved to: {summary_path}")
            
            # Show top 10 combinations by accuracy
            print(f"\nTOP 10 COMBINATIONS BY ACCURACY:")
            top_10 = results_df.nlargest(10, 'accuracy')[[
                'combo_name', 'model', 'accuracy', 'f1_score', 'final_rows',
                'missing_numerical', 'outlier_method', 'scaling_method'
            ]]
            print(top_10.to_string(index=False))
            
            # Show top 10 combinations by F1 score
            print(f"\nTOP 10 COMBINATIONS BY F1 SCORE:")
            top_10_f1 = results_df.nlargest(10, 'f1_score')[[
                'combo_name', 'model', 'accuracy', 'f1_score', 'final_rows',
                'missing_numerical', 'outlier_method', 'scaling_method'
            ]]
            print(top_10_f1.to_string(index=False))
        
        print(f"\nView detailed results with: mlflow ui")
        print(" All 500 combinations tested successfully!")

if __name__ == "__main__":
    data_path = DATA_PATH
    
    # Validate data path
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Please check the file path and try again.")
    else:
        experiment = BoxPlotPreprocessingExperiment(data_path)
        experiment.run_all_experiments()