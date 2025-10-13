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


class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, object]:
        try:
            logging.info("Training LightGBM with EXACT MLflow Parameters")

            # Splitting the train and test data
            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]
            logging.info(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")

            # LightGBM with EXACT MLflow Parameters from config
            model = lgb.LGBMClassifier(
                n_estimators=self.model_trainer_config.n_estimators,
                max_depth=self.model_trainer_config.max_depth,
                learning_rate=self.model_trainer_config.learning_rate,
                num_leaves=self.model_trainer_config.num_leaves,
                subsample=self.model_trainer_config.subsample,
                colsample_bytree=self.model_trainer_config.colsample_bytree,
                reg_alpha=self.model_trainer_config.reg_alpha,
                reg_lambda=self.model_trainer_config.reg_lambda,
                min_child_samples=self.model_trainer_config.min_child_samples,
                random_state=self.model_trainer_config.random_state,
                verbose=self.model_trainer_config.verbose
            )

            # Fit the model - EXACT same as MLflow
            logging.info("Model training started...")
            model.fit(x_train, y_train)
            logging.info("Model training completed")

            # Predictions and evaluation - EXACT same metrics as MLflow
            y_pred = model.predict(x_test)
            y_pred_proba = model.predict_proba(x_test)[:, 1]

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='binary')
            precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
            auc_score = roc_auc_score(y_test, y_pred_proba)

            logging.info("Model Performance (MLflow Metrics):")
            logging.info(f"   Accuracy: {accuracy:.4f}")
            logging.info(f"   Precision: {precision:.4f}")
            logging.info(f"   Recall: {recall:.4f}")
            logging.info(f"   F1 Score: {f1:.4f}")
            logging.info(f"   AUC Score: {auc_score:.4f}")

            # Creating metric artifact with MLflow metrics
            metric_artifact = ClassificationMetricArtifact(
                f1_score=f1, 
                precision_score=precision, 
                recall_score=recall,
                accuracy_score=accuracy,
                auc_score=auc_score
            )
            
            return model, metric_artifact
        
        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise MyException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Starting Model Trainer - EXACT MLflow Approach")
        try:
            
            # Load transformed data
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            logging.info("Train-test data loaded")

            # Train model
            trained_model, metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)
            logging.info("Model training completed")

            # Load preprocessing object
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            logging.info("Preprocessing object loaded")

            # Save final model
            logging.info("Saving final model...")
            my_model = MyModel(preprocessing_object=preprocessing_obj, trained_model_object=trained_model)
            save_object(self.model_trainer_config.trained_model_file_path, my_model)
            logging.info(f"Final model saved")

            # Create artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
            return model_trainer_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e