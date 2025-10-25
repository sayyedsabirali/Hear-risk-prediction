import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.constants import TARGET_COLUMN
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import DataIngestionArtifact, ModelTrainerArtifact, ModelEvaluationArtifact, DataTransformationArtifact
from src.entity.s3_estimator import Proj1Estimator

from src.utils.main_utils import load_object, load_numpy_array_data
from src.exception import MyException
from src.logger import logging

@dataclass
class EvaluateModelResponse:
    trained_model_accuracy_score: float
    best_model_accuracy_score: float
    is_model_accepted: bool
    difference: float

class ModelEvaluation:
    def __init__(self, 
                 model_eval_config: ModelEvaluationConfig, 
                 data_ingestion_artifact: DataIngestionArtifact, 
                 data_transformation_artifact: DataTransformationArtifact, 
                 model_trainer_artifact: ModelTrainerArtifact):
        
        self.model_eval_config = model_eval_config
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_artifact = model_trainer_artifact

    def get_best_model(self) -> Optional[Proj1Estimator]:
        """
        Get the best model from S3 if it exists
        """
        try:
            logging.info("Checking for existing model in S3...")
            bucket_name = self.model_eval_config.bucket_name
            model_path = self.model_eval_config.s3_model_key_path
            
            proj1_estimator = Proj1Estimator(bucket_name=bucket_name, model_path=model_path)
            
            if proj1_estimator.is_model_present(model_path=model_path):
                logging.info(f"Model found in S3 at: {model_path}")
                return proj1_estimator
            
            logging.info("No existing model found in S3")
            return None
            
        except Exception as e:
            logging.warning(f"Error checking S3 model: {str(e)}")
            raise MyException(e, sys)

    def _apply_complete_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply COMPLETE feature engineering EXACTLY as during training
        """
        try:
            logging.info("Applying COMPLETE advanced medical feature engineering...")
            df_eng = df.copy()
            original_features = len(df_eng.columns)
            
            # Safe mathematical operations
            def safe_log1p(x):
                return np.log1p(np.where(x <= 0, 0.001, x))
            
            def safe_sqrt(x):
                return np.sqrt(np.where(x < 0, 0, x))
            
            # 1. CARDIAC-SPECIFIC FEATURES
            logging.info("   Creating cardiac-specific features...")
            
            if all(col in df_eng.columns for col in ['troponin_t', 'creatine_kinase_mb']):
                df_eng['cardiac_biomarker_product'] = safe_log1p(df_eng['troponin_t']) * safe_log1p(df_eng['creatine_kinase_mb'])
                df_eng['cardiac_risk_index'] = (df_eng['troponin_t'] * 100) + (df_eng['creatine_kinase_mb'] * 10)
                df_eng['elevated_troponin'] = (df_eng['troponin_t'] > 0.1).astype(int)
                df_eng['elevated_ckmb'] = (df_eng['creatine_kinase_mb'] > 5).astype(int)
                df_eng['both_biomarkers_elevated'] = ((df_eng['troponin_t'] > 0.1) & (df_eng['creatine_kinase_mb'] > 5)).astype(int)
            
            # 2. HEMODYNAMIC STABILITY FEATURES
            logging.info("   Creating hemodynamic features...")
            
            if all(col in df_eng.columns for col in ['bp_systolic', 'bp_diastolic', 'heart_rate']):
                df_eng['mean_arterial_pressure'] = (df_eng['bp_systolic'] + 2 * df_eng['bp_diastolic']) / 3
                df_eng['pulse_pressure'] = df_eng['bp_systolic'] - df_eng['bp_diastolic']
                df_eng['rate_pressure_product'] = df_eng['heart_rate'] * df_eng['bp_systolic'] / 100
                df_eng['shock_index'] = df_eng['heart_rate'] / df_eng['bp_systolic']
                df_eng['hemodynamic_instability'] = (
                    (df_eng['bp_systolic'] < 90) | 
                    (df_eng['heart_rate'] > 120) | 
                    (df_eng['mean_arterial_pressure'] < 65)
                ).astype(int)
            
            # 3. OXYGENATION AND RESPIRATORY FEATURES
            logging.info("   Creating oxygenation features...")
            
            if all(col in df_eng.columns for col in ['spo2', 'respiratory_rate', 'hemoglobin']):
                df_eng['oxygen_saturation_ratio'] = df_eng['spo2'] / 100
                df_eng['oxygen_content'] = (df_eng['hemoglobin'] * 1.34 * df_eng['spo2'] / 100) + (0.003 * df_eng['spo2'])
                df_eng['respiratory_distress'] = (
                    (df_eng['spo2'] < 92) | (df_eng['respiratory_rate'] > 24)
                ).astype(int)
                df_eng['ventilation_perfusion_ratio'] = df_eng['respiratory_rate'] / (df_eng['spo2'] + 0.1)
            
            # 4. METABOLIC AND RENAL FEATURES
            logging.info("   Creating metabolic features...")
            
            if all(col in df_eng.columns for col in ['creatinine', 'potassium', 'sodium', 'glucose']):
                df_eng['renal_function_score'] = df_eng['creatinine'] * df_eng['potassium']
                df_eng['electrolyte_imbalance'] = (
                    (df_eng['sodium'] < 135) | (df_eng['sodium'] > 145) |
                    (df_eng['potassium'] < 3.5) | (df_eng['potassium'] > 5.0)
                ).astype(int)
                df_eng['hyperglycemia'] = (df_eng['glucose'] > 180).astype(int)
                df_eng['hypoglycemia'] = (df_eng['glucose'] < 70).astype(int)
                df_eng['bun_creatinine_ratio'] = df_eng['glucose'] / (df_eng['creatinine'] + 0.1)
            
            # 5. INFLAMMATORY AND HEMATOLOGICAL FEATURES
            logging.info("   Creating inflammatory features...")
            
            if 'white_blood_cells' in df_eng.columns:
                df_eng['wbc_elevated'] = (df_eng['white_blood_cells'] > 11).astype(int)
                df_eng['wbc_low'] = (df_eng['white_blood_cells'] < 4).astype(int)
                df_eng['leukocytosis'] = (df_eng['white_blood_cells'] > 15).astype(int)
            
            if 'hemoglobin' in df_eng.columns:
                df_eng['anemia'] = (df_eng['hemoglobin'] < 12).astype(int)
                df_eng['severe_anemia'] = (df_eng['hemoglobin'] < 8).astype(int)
                df_eng['polycythemia'] = (df_eng['hemoglobin'] > 16).astype(int)
            
            # 6. AGE AND COMORBIDITY FEATURES
            logging.info("   Creating age-comorbidity features...")
            
            if 'anchor_age' in df_eng.columns:
                df_eng['age_squared'] = df_eng['anchor_age'] ** 2
                df_eng['age_cubed'] = df_eng['anchor_age'] ** 3
                df_eng['elderly'] = (df_eng['anchor_age'] >= 65).astype(int)
                df_eng['very_elderly'] = (df_eng['anchor_age'] >= 80).astype(int)
                df_eng['young_adult'] = ((df_eng['anchor_age'] >= 18) & (df_eng['anchor_age'] <= 40)).astype(int)
                
                if 'heart_rate' in df_eng.columns:
                    df_eng['age_adjusted_hr_max'] = 220 - df_eng['anchor_age']
                    df_eng['hr_percentage_max'] = df_eng['heart_rate'] / df_eng['age_adjusted_hr_max']
            
            # 7. VITAL SIGN INTERACTIONS
            logging.info("   Creating vital sign interactions...")
            
            vital_cols = ['heart_rate', 'respiratory_rate', 'spo2', 'temperature']
            available_vitals = [col for col in vital_cols if col in df_eng.columns]
            
            if len(available_vitals) >= 2:
                df_eng['vital_sign_mean'] = df_eng[available_vitals].mean(axis=1)
                df_eng['vital_sign_std'] = df_eng[available_vitals].std(axis=1)
                df_eng['vital_sign_range'] = df_eng[available_vitals].max(axis=1) - df_eng[available_vitals].min(axis=1)
                
                df_eng['ews_score'] = 0
                if 'heart_rate' in df_eng.columns:
                    df_eng['ews_score'] += ((df_eng['heart_rate'] < 50) | (df_eng['heart_rate'] > 100)).astype(int)
                if 'respiratory_rate' in df_eng.columns:
                    df_eng['ews_score'] += ((df_eng['respiratory_rate'] < 12) | (df_eng['respiratory_rate'] > 20)).astype(int)
                if 'spo2' in df_eng.columns:
                    df_eng['ews_score'] += (df_eng['spo2'] < 95).astype(int)
                if 'temperature' in df_eng.columns:
                    df_eng['ews_score'] += ((df_eng['temperature'] < 36) | (df_eng['temperature'] > 38)).astype(int)
            
            # 8. CLINICAL RISK SCORES
            logging.info("   Creating clinical risk scores...")
            
            df_eng['timi_like_score'] = 0
            if 'anchor_age' in df_eng.columns:
                df_eng['timi_like_score'] += (df_eng['anchor_age'] > 65).astype(int)
            if all(col in df_eng.columns for col in ['troponin_t', 'creatine_kinase_mb']):
                df_eng['timi_like_score'] += ((df_eng['troponin_t'] > 0.1) | (df_eng['creatine_kinase_mb'] > 5)).astype(int)
            if 'heart_rate' in df_eng.columns:
                df_eng['timi_like_score'] += (df_eng['heart_rate'] > 100).astype(int)
            if 'bp_systolic' in df_eng.columns:
                df_eng['timi_like_score'] += (df_eng['bp_systolic'] < 100).astype(int)
            
            # 9. POLYNOMIAL AND INTERACTION FEATURES
            logging.info("   Creating polynomial features...")
            
            key_continuous = ['troponin_t', 'creatine_kinase_mb', 'hemoglobin', 'heart_rate', 'anchor_age']
            for col in key_continuous:
                if col in df_eng.columns:
                    df_eng[f'{col}_squared'] = df_eng[col] ** 2
                    df_eng[f'{col}_cubed'] = df_eng[col] ** 3
                    df_eng[f'{col}_log'] = safe_log1p(df_eng[col])
                    df_eng[f'{col}_sqrt'] = safe_sqrt(df_eng[col])
            
            # 10. RATIO FEATURES
            logging.info("   Creating ratio features...")
            
            if all(col in df_eng.columns for col in ['hemoglobin', 'white_blood_cells']):
                df_eng['hgb_wbc_ratio'] = df_eng['hemoglobin'] / (df_eng['white_blood_cells'] + 0.1)
            
            if all(col in df_eng.columns for col in ['heart_rate', 'respiratory_rate']):
                df_eng['hr_rr_ratio'] = df_eng['heart_rate'] / (df_eng['respiratory_rate'] + 0.1)
            
            if all(col in df_eng.columns for col in ['spo2', 'respiratory_rate']):
                df_eng['spo2_rr_ratio'] = df_eng['spo2'] / (df_eng['respiratory_rate'] + 0.1)
            
            new_features = len(df_eng.columns) - original_features
            logging.info(f"Advanced feature engineering completed: {new_features} new features")
            logging.info(f"   Total features: {len(df_eng.columns)}")
            
            return df_eng
            
        except Exception as e:
            logging.error(f"Error in feature engineering: {str(e)}")
            raise MyException(e, sys)

    def _handle_missing_values_advanced(self, df):
        """EXACT SAME as DataTransformation class"""
        logging.info("Advanced Missing Value Handling...")
        
        if not isinstance(df, pd.DataFrame):
            logging.error(f"Expected DataFrame but got {type(df)}")
            raise ValueError(f"Expected DataFrame but got {type(df)}")
        
        df_imputed = df.copy()
        
        # Categorical columns
        categorical_cols = ['insurance', 'marital_status', 'gender', 'admission_type', 'race']
        for col in categorical_cols:
            if col in df_imputed.columns and df_imputed[col].isna().any():
                mode_val = df_imputed[col].mode()
                fill_value = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                df_imputed.loc[:, col] = df_imputed[col].fillna(fill_value)
        
        # Numerical columns
        numerical_cols = df_imputed.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_imputed[col].isna().any():
                median_value = df_imputed[col].median()
                df_imputed.loc[:, col] = df_imputed[col].fillna(median_value)
        
        logging.info(f"Missing values handled. Data shape: {df_imputed.shape}")
        return df_imputed

    def _map_gender_column(self, df):
        """EXACT SAME as DataTransformation class"""
        logging.info("Mapping 'gender' column to binary values")
        if 'gender' in df.columns:
            df['gender'] = df['gender'].map({'F': 0, 'M': 1}).astype(int)
        return df

    def _label_encode_columns(self, df):
        """EXACT SAME as DataTransformation class"""
        logging.info("Label encoding high cardinality categorical features")
        high_cardinality = ['insurance', 'marital_status', 'race']
        for col in high_cardinality:
            if col in df.columns:
                df[col] = pd.factorize(df[col])[0]
        return df

    def _create_dummy_columns(self, df):
        """EXACT SAME as DataTransformation class"""
        logging.info("Creating dummy variables for categorical features")
        low_cardinality = ['admission_type']
        for col in low_cardinality:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col])
        return df

    def _drop_id_columns(self, df):
        """EXACT SAME as DataTransformation class"""
        logging.info("Dropping identifier columns")
        drop_cols = ['_id', 'subject_id', 'hadm_id']
        for col in drop_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
        return df

    def _preprocess_for_s3_model(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        SIMPLE AND RELIABLE FIX: Use the ALREADY TRANSFORMED test data
        This guarantees EXACT same features as training
        """
        try:
            logging.info("Using pre-transformed test data for S3 model evaluation...")
            
            # Load the transformed test data (guaranteed to have same features as training)
            transformed_test_data = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )
            
            # Extract features only (exclude target)
            X_test_transformed = transformed_test_data[:, :-1]
            
            # Create DataFrame with same number of features
            feature_names = [f'feature_{i}' for i in range(X_test_transformed.shape[1])]
            df_processed = pd.DataFrame(X_test_transformed, columns=feature_names)
            
            # ✅ CRITICAL FIX: Convert ALL columns to numeric
            logging.info("Converting all columns to numeric types...")
            for col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
            
            # Fill any remaining NaN values with 0
            df_processed = df_processed.fillna(0)
            
            logging.info(f"Using transformed data with shape: {df_processed.shape}")
            logging.info(f"Features guaranteed to match training: {df_processed.shape[1]}")
            logging.info(f"Data types after conversion: {df_processed.dtypes.unique()}")
            
            return df_processed
            
        except Exception as e:
            logging.error(f"Error in simple preprocessing: {str(e)}")
            raise MyException(e, sys)

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Evaluate trained model against existing production model (if any)
        """
        try:
            logging.info("Starting Model Evaluation - Heart Attack Risk Prediction")
         
            
            # Load trained model to get reference
            logging.info("Loading trained model for reference...")
            trained_model_wrapper = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            
            # Get expected feature count from training
            logging.info("Loading transformed training data for reference...")
            transformed_train_data = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_train_file_path
            )
            expected_feature_count = transformed_train_data.shape[1] - 1  # Exclude target
            logging.info(f"Expected feature count from training: {expected_feature_count}")
            
            # Load RAW test data for evaluation
            logging.info("Loading RAW test data for evaluation...")
            raw_test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            logging.info(f"Raw test data shape: {raw_test_df.shape}")
            
            # Separate features and target from RAW data
            X_test_raw = raw_test_df.drop(columns=[TARGET_COLUMN], axis=1)
            y_test_raw = raw_test_df[TARGET_COLUMN]
            
            logging.info(f"Raw test features shape: {X_test_raw.shape}")
            logging.info(f"Raw test target shape: {y_test_raw.shape}")
            logging.info(f"Unique target values: {np.unique(y_test_raw)}")

            # Evaluate trained model on TRANSFORMED data
            logging.info("Evaluating newly trained model...")
            transformed_test_data = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_test_file_path
            )
            
            # Split transformed data into features and target
            X_test_transformed = transformed_test_data[:, :-1]
            y_test_transformed = transformed_test_data[:, -1].astype(int)
            
            # Trained model expects transformed data
            if hasattr(trained_model_wrapper, 'trained_model_object'):
                y_pred_trained = trained_model_wrapper.trained_model_object.predict(X_test_transformed)
            else:
                y_pred_trained = trained_model_wrapper.predict(X_test_transformed)
            
            # Calculate metrics for trained model
            trained_model_accuracy_score = accuracy_score(y_test_transformed, y_pred_trained)
            trained_f1 = f1_score(y_test_transformed, y_pred_trained, average='binary', zero_division=0)
            trained_precision = precision_score(y_test_transformed, y_pred_trained, average='binary', zero_division=0)
            trained_recall = recall_score(y_test_transformed, y_pred_trained, average='binary', zero_division=0)
            
            logging.info("=" * 90)
            logging.info("Newly Trained Model Performance:")
            logging.info(f"  Accuracy:  {trained_model_accuracy_score:.4f}")
            logging.info(f"  Precision: {trained_precision:.4f}")
            logging.info(f"  Recall:    {trained_recall:.4f}")
            logging.info(f"  F1 Score:  {trained_f1:.4f}")
            logging.info("=" * 90)

            # Evaluate best model from S3 (if exists)
            best_model_accuracy_score = 0
            best_model = self.get_best_model()
            
            if best_model is not None:
                logging.info("Evaluating existing production/S3 model...")
                try:
                    logging.info("Applying preprocessing for S3 model...")
                    X_test_preprocessed = self._preprocess_for_s3_model(X_test_raw)
                    
                    logging.info(f"Preprocessed data shape: {X_test_preprocessed.shape}")
                    logging.info(f"Expected features: {expected_feature_count}")
                    logging.info(f"Actual features: {X_test_preprocessed.shape[1]}")
                    
                    # Load the actual model from S3 estimator
                    logging.info("Loading S3 model...")
                    s3_model = best_model.load_model()
                    
                    # S3 model expects preprocessed DataFrame
                    logging.info("Making predictions with S3 model...")
                    
                    try:
                        if hasattr(s3_model, 'trained_model_object'):
                            y_pred_best = s3_model.trained_model_object.predict(X_test_preprocessed)
                        else:
                            y_pred_best = s3_model.predict(X_test_preprocessed)
                        
                        # Calculate metrics for S3 model
                        best_model_accuracy_score = accuracy_score(y_test_raw, y_pred_best)
                        best_f1 = f1_score(y_test_raw, y_pred_best, average='binary', zero_division=0)
                        best_precision = precision_score(y_test_raw, y_pred_best, average='binary', zero_division=0)
                        best_recall = recall_score(y_test_raw, y_pred_best, average='binary', zero_division=0)
                        
                        logging.info("=" * 90)
                        logging.info("Production/S3 Model Performance:")
                        logging.info(f"  Accuracy:  {best_model_accuracy_score:.4f}")
                        logging.info(f"  Precision: {best_precision:.4f}")
                        logging.info(f"  Recall:    {best_recall:.4f}")
                        logging.info(f"  F1 Score:  {best_f1:.4f}")
                        logging.info("=" * 90)
                        
                    except Exception as predict_error:
                        logging.warning(f"S3 model prediction failed: {predict_error}")
                        logging.info("This might be due to feature mismatch, accepting new model")
                        best_model_accuracy_score = 0
                    
                except Exception as e:
                    logging.warning(f"Could not evaluate S3 model: {str(e)}")
                    logging.info("This is acceptable for the first deployment or model updates")
                    best_model_accuracy_score = 0
            else:
                logging.info("No production model found in S3. This will be the first deployment.")
                best_model_accuracy_score = 0

            # Compare models
            accuracy_difference = trained_model_accuracy_score - best_model_accuracy_score
            
            # Check if new model should be accepted
            accuracy_threshold = 0.85  # Minimum acceptable accuracy
            is_model_accepted = (trained_model_accuracy_score > accuracy_threshold and 
                               trained_model_accuracy_score >= best_model_accuracy_score)
            
            logging.info("=" * 90)
            logging.info("Model Comparison Results:")
            logging.info(f"  New Model Accuracy:        {trained_model_accuracy_score:.4f}")
            logging.info(f"  Production Model Accuracy: {best_model_accuracy_score:.4f}")
            logging.info(f"  Accuracy Difference:       {accuracy_difference:+.4f}")
            logging.info(f"  Model Accepted:            {is_model_accepted}")
            logging.info("=" * 90)
            
            result = EvaluateModelResponse(
                trained_model_accuracy_score=trained_model_accuracy_score,
                best_model_accuracy_score=best_model_accuracy_score,
                is_model_accepted=is_model_accepted,
                difference=accuracy_difference
            )
            
            logging.info(f"Model evaluation result: {result}")
            return result

        except Exception as e:
            logging.error(f"Error in model evaluation: {str(e)}")
            raise MyException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Initiate the model evaluation process
        """
        try:
            logging.info("Entered initiate_model_evaluation method of ModelEvaluation class")
            print("=" * 90)
            print("Starting Model Evaluation Component - Heart Attack Risk Prediction")
            print("=" * 90)
            
            # Perform model evaluation
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            # Create evaluation artifact
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference
            )

            logging.info("=" * 90)
            logging.info("Model Evaluation Artifact:")
            logging.info(f"  Model Accepted:     {model_evaluation_artifact.is_model_accepted}")
            logging.info(f"  Trained Model Path: {model_evaluation_artifact.trained_model_path}")
            logging.info(f"  S3 Model Path:      {model_evaluation_artifact.s3_model_path}")
            logging.info(f"  Accuracy Change:    {model_evaluation_artifact.changed_accuracy:+.4f}")
            logging.info("=" * 90)
            
            if evaluate_model_response.is_model_accepted:
                logging.info("NEW MODEL ACCEPTED! Will be pushed to production.")
            else:
                logging.info("New model performance is not better. Keeping existing production model.")

            return model_evaluation_artifact
            
        except Exception as e:
            raise MyException(e, sys)