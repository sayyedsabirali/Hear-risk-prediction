"""Complete MLflow Experiment Pipeline"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from preprocessor import DataPreprocessor
from model_trainer import ModelTrainer
from config import (generate_combinations, 
                    generate_stratified_combinations, 
                    get_full_grid_combinations)

class MLflowExperiment:
    """Complete MLflow experiment with all features"""
    
    def __init__(self, data_path, experiment_name=None):
        self.data_path = data_path
        self.df = None
        self.artifacts_dir = Path("mlflow_artifacts")
        self.artifacts_dir.mkdir(exist_ok=True)
        
        # Setup MLflow
        mlflow.set_tracking_uri("mlruns")
        if experiment_name is None:
            experiment_name = f"1. DATA_PREPROCESSING_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mlflow.set_experiment(experiment_name)
        print(f"✅ MLflow Experiment: {experiment_name}")
        
    def load_data(self):
        """Load and prepare dataset"""
        print("\n📂 Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        
        # Drop ID columns
        id_cols = ['subject_id', 'hadm_id']
        self.df = self.df.drop(columns=[col for col in id_cols if col in self.df.columns], errors='ignore')
        
        print(f"✅ Shape: {self.df.shape}")
        print(f"🎯 Target: {self.df['heart_attack'].value_counts().to_dict()}")
        return self.df
    
    def create_confusion_matrix_plot(self, cm, combo_name):
        """Create and save confusion matrix plot"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=['No Attack', 'Attack'],
                   yticklabels=['No Attack', 'Attack'])
        plt.title(f'Confusion Matrix\n{combo_name}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        safe_name = combo_name.replace('/', '_').replace(' ', '_')
        cm_path = self.artifacts_dir / f'confusion_matrix_{safe_name}.png'
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(cm_path)
    
    def create_feature_importance_plot(self, feature_df, combo_name):
        """Create and save feature importance plot"""
        if feature_df is None or len(feature_df) == 0:
            return None
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(feature_df)), feature_df['importance'].values)
        plt.yticks(range(len(feature_df)), feature_df['feature'].values)
        plt.xlabel('Importance', fontsize=12)
        plt.title(f'Top 10 Feature Importances\n{combo_name}', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        safe_name = combo_name.replace('/', '_').replace(' ', '_')
        fi_path = self.artifacts_dir / f'feature_importance_{safe_name}.png'
        plt.savefig(fi_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(fi_path)
    
    def run_single_experiment(self, combo):
        """Run single experiment with MLflow logging"""
        
        exp_name = f"{combo['name']}_{combo['model']['name']}"
        
        with mlflow.start_run(run_name=exp_name):
            try:
                print(f"\n{'='*70}")
                print(f"🔬 Running: {exp_name}")
                print(f"{'='*70}")
                
                # Initialize preprocessor
                preprocessor = DataPreprocessor()
                trainer = ModelTrainer()
                
                # Copy original data
                df_processed = self.df.copy()
                original_rows = len(df_processed)
                
                # Identify columns
                preprocessor.identify_columns(df_processed)
                
                # Step 1: Handle missing values
                print("  ➤ Handling missing values...")
                df_processed = preprocessor.handle_missing_values(
                    df_processed,
                    combo['missing']['numerical'],
                    combo['missing']['categorical']
                )
                
                # Step 2: Handle outliers
                print("  ➤ Handling outliers...")
                df_processed = preprocessor.handle_outliers(
                    df_processed,
                    combo['outlier']['method'],
                    combo['outlier']['threshold']
                )
                
                # Step 3: Reduce skewness
                print("  ➤ Reducing skewness...")
                df_processed = preprocessor.reduce_skewness(
                    df_processed,
                    combo['skewness']['method']
                )
                
                # Step 4: Scale features
                print("  ➤ Scaling features...")
                df_processed = preprocessor.scale_features(
                    df_processed,
                    combo['scaling']['method']
                )
                
                # Step 5: Encode categorical
                print("  ➤ Encoding categorical...")
                df_processed = preprocessor.encode_categorical(df_processed)
                
                final_rows = len(df_processed)
                
                # Check if enough data remains
                if final_rows < 100:
                    print(f"  ⚠️  Insufficient data: {final_rows} rows")
                    mlflow.set_tag("status", "insufficient_data")
                    return None
                
                # Log preprocessing parameters
                mlflow.log_params({
                    "combo_id": combo['id'],
                    "combo_name": combo['name'],
                    "model_name": combo['model']['name'],
                    "missing_numerical": combo['missing']['numerical'],
                    "missing_categorical": combo['missing']['categorical'],
                    "outlier_method": combo['outlier']['method'],
                    "outlier_threshold": combo['outlier']['threshold'],
                    "skewness_method": combo['skewness']['method'],
                    "scaling_method": combo['scaling']['method'],
                })
                
                # Log data metrics
                mlflow.log_metrics({
                    "original_rows": original_rows,
                    "final_rows": final_rows,
                    "rows_removed": original_rows - final_rows,
                    "data_retention_pct": (final_rows / original_rows) * 100
                })
                
                # Log preprocessing steps
                for i, step in enumerate(preprocessor.get_preprocessing_summary()):
                    mlflow.log_param(f"preprocessing_step_{i+1}", f"{step['step']}: {step.get('method', 'N/A')}")
                
                # Train model
                print(f"  ➤ Training {combo['model']['name']}...")
                results = trainer.train_and_evaluate(
                    df_processed,
                    combo['model']['name'],
                    combo['model']['params']
                )
                
                if results is None:
                    print("  ❌ Training failed")
                    mlflow.set_tag("status", "training_failed")
                    return None
                
                # Log model metrics
                mlflow.log_metrics({
                    "accuracy": results['accuracy'],
                    "precision": results['precision'],
                    "recall": results['recall'],
                    "auc": results['auc'],
                    "f1_score": results['f1_score'],
                    "train_size": results['train_size'],
                    "test_size": results['test_size']
                })
                
                # Create and log confusion matrix
                print("  ➤ Creating confusion matrix...")
                cm_path = self.create_confusion_matrix_plot(
                    results['confusion_matrix'],
                    exp_name
                )
                mlflow.log_artifact(cm_path)
                
                # Create and log feature importance
                if results['feature_importance'] is not None:
                    print("  ➤ Creating feature importance plot...")
                    fi_path = self.create_feature_importance_plot(
                        results['feature_importance'],
                        exp_name
                    )
                    if fi_path:
                        mlflow.log_artifact(fi_path)
                    
                    # Log top 10 features
                    feature_df = results['feature_importance'].head(10).reset_index(drop=True)

                    for idx, row in feature_df.iterrows():
                        mlflow.log_param(
                            f"top_feature_{idx+1}",
                            f"{row['feature']}: {row['importance']:.4f}"
                        )
                
                # Log model
                mlflow.sklearn.log_model(results['model'], "model")
                
                # Set success status
                mlflow.set_tag("status", "success")
                
                print(f"\n  SUCCESS!")
                print(f"     Accuracy: {results['accuracy']:.4f}")
                print(f"     Precision: {results['precision']:.4f}")
                print(f"     Recall: {results['recall']:.4f}")
                print(f"     F1-Score: {results['f1_score']:.4f}")
                print(f"     AUC: {results['auc']:.4f}")
                
                return {
                    'combo_id': combo['id'],
                    'combo_name': combo['name'],
                    'model': combo['model']['name'],
                    'accuracy': results['accuracy'],
                    'precision': results['precision'],
                    'recall': results['recall'],
                    'f1_score': results['f1_score'],
                    'auc': results['auc'],
                    'data_retention_pct': (final_rows / original_rows) * 100
                }
                
            except Exception as e:
                print(f" Error: {str(e)}")
                mlflow.set_tag("status", "error")
                mlflow.set_tag("error_message", str(e))
                return None
    
    def run_all_experiments(self, max_combinations=20):
        """Run multiple experiments"""
        print("\n" + "="*70)
        print("🚀 STARTING MLFLOW EXPERIMENTS")
        print("="*70)
        
        # Load data
        self.load_data()
        
        # Generate combinations
        combinations = generate_stratified_combinations(n_per_model=20)
        print(f"\n📊 Total combinations to test: {len(combinations)}")
        
        # Run experiments
        results = []
        successful = 0
        
        for i, combo in enumerate(combinations):
            print(f"\n[{i+1}/{len(combinations)}]")
            result = self.run_single_experiment(combo)
            
            if result is not None:
                results.append(result)
                successful += 1
        
        # Save results
        if results:
            results_df = pd.DataFrame(results)
            results_path = self.artifacts_dir / 'experiment_results.csv'
            results_df.to_csv(results_path, index=False)
            
            print("\n" + "="*70)
            print("🎉 EXPERIMENTS COMPLETED!")
            print("="*70)
            print(f"✅ Successful: {successful}/{len(combinations)}")
            print(f"❌ Failed: {len(combinations) - successful}/{len(combinations)}")
            print(f"\n📄 Results saved to: {results_path}")
            
            # Show top performers
            print("\n🏆 TOP 5 PERFORMERS BY ACCURACY:")
            top5 = results_df.nlargest(5, 'accuracy')
            for idx, row in top5.iterrows():
                print(f"  {row['combo_name']} ({row['model']})")
                print(f"    Accuracy: {row['accuracy']:.4f} | F1: {row['f1_score']:.4f} | AUC: {row['auc']:.4f}")
            
            return results_df
        else:
            print("\n❌ No successful experiments!")
            return None


if __name__ == "__main__":
    # Update this path to your data
    DATA_PATH = "F:\\18. MAJOR PROJECT\\Heart-related-content\\heart_risk_complete_dataset.csv" 
    
    # Run experiments
    experiment = MLflowExperiment(DATA_PATH)
    results = experiment.run_all_experiments(max_combinations=200)
    
    print("\n Done! Check MLflow UI with: mlflow ui")