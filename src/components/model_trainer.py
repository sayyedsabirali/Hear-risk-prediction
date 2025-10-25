import sys
from typing import Tuple

import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, load_object, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from src.entity.estimator import MyModel
from src.constants import *
class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, object]:
        try:
            logging.info("Training LightGBM Classifier with optimized parameters for Heart Attack Prediction")

            # Splitting the train and test data into features and target variables
            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]
            
            # Convert target to integer (fix for "Unknown label type: unknown" error)
            y_train = y_train.astype(int)
            y_test = y_test.astype(int)
            
            logging.info("train-test split done.")
            logging.info(f"Target variable - Unique values in y_train: {np.unique(y_train)}")
            logging.info(f"Target variable - Unique values in y_test: {np.unique(y_test)}")

            # Initialize LightGBM Classifier with optimized parameters
            model = lgb.LGBMClassifier(
                n_estimators=MODEL_TRAINER_N_ESTIMATORS,
                max_depth=MODEL_TRAINER_MAX_DEPTH,
                learning_rate=MODEL_TRAINER_LEARNING_RATE,
                num_leaves=MODEL_TRAINER_NUM_LEAVES,
                subsample=MODEL_TRAINER_SUBSAMPLE,
                colsample_bytree=MODEL_TRAINER_COLSAMPLE_BYTREE,
                reg_alpha=MODEL_TRAINER_REG_ALPHA,
                reg_lambda=MODEL_TRAINER_REG_LAMBDA,
                min_child_samples=MODEL_TRAINER_MIN_CHILD_SAMPLES,
                random_state=MODEL_TRAINER_RANDOM_STATE,
                verbose=MODEL_TRAINER_VERBOSE
            )

            # Fit the model
            logging.info("Model training started...")
            model.fit(x_train, y_train)
            logging.info("Model training completed successfully.")

            # Predictions and evaluation metrics
            logging.info("Calculating evaluation metrics...")
            y_pred = model.predict(x_test)
            y_pred_proba = model.predict_proba(x_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
            precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
            auc_score = roc_auc_score(y_test, y_pred_proba)

            logging.info(f"Model Performance Metrics:")
            logging.info(f"  Accuracy: {accuracy:.4f}")
            logging.info(f"  Precision: {precision:.4f}")
            logging.info(f"  Recall: {recall:.4f}")
            logging.info(f"  F1 Score: {f1:.4f}")
            logging.info(f"  AUC Score: {auc_score:.4f}")

            # Creating metric artifact with all 5 metrics
            metric_artifact = ClassificationMetricArtifact(
                f1_score=f1, 
                precision_score=precision, 
                recall_score=recall,
                accuracy_score=accuracy,
                auc_score=auc_score
            )
            
            return model, metric_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        try:
            print("-------- Starting Model Trainer Component ----------")
            
            # Load transformed train and test data
            logging.info("Loading transformed train and test data...")
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            logging.info(f"Train data shape: {train_arr.shape}, Test data shape: {test_arr.shape}")
            
            # Train model and get metrics
            logging.info("Training model with advanced medical features...")
            trained_model, metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)
            logging.info("Model training and evaluation completed.")
            
            # Load preprocessing object
            logging.info("Loading preprocessing object...")
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            logging.info("Preprocessing object loaded successfully.")
            
            # Use test accuracy directly from metric_artifact
            test_accuracy = metric_artifact.accuracy_score
            logging.info(f"Test accuracy (from metric artifact): {test_accuracy:.4f}")
            logging.info(f"Expected accuracy threshold: {self.model_trainer_config.expected_accuracy}")
            
            if test_accuracy < self.model_trainer_config.expected_accuracy:
                logging.error(f"Model accuracy {test_accuracy:.4f} is below expected threshold {self.model_trainer_config.expected_accuracy}")
                raise Exception(f"No model found with score above the expected accuracy: {self.model_trainer_config.expected_accuracy}")
            
            # Save the final model object (preprocessing + trained model)
            logging.info("Model performance exceeds threshold. Saving model...")
            my_model = MyModel(preprocessing_object=preprocessing_obj, trained_model_object=trained_model)
            save_object(self.model_trainer_config.trained_model_file_path, my_model)
            logging.info(f"Model saved successfully at: {self.model_trainer_config.trained_model_file_path}")

            # Create and return the ModelTrainerArtifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
            logging.info("Model Trainer Artifact Created:")
            logging.info(f"  Model Path: {model_trainer_artifact.trained_model_file_path}")
            logging.info(f"  Accuracy: {metric_artifact.accuracy_score:.4f}")
            logging.info(f"  F1 Score: {metric_artifact.f1_score:.4f}")
            logging.info(f"  Precision: {metric_artifact.precision_score:.4f}")
            logging.info(f"  Recall: {metric_artifact.recall_score:.4f}")
            logging.info(f"  AUC Score: {metric_artifact.auc_score:.4f}")
            
            return model_trainer_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e
