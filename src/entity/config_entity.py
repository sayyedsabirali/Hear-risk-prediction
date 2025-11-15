from dataclasses import dataclass
import os
from src.constants import *


@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.database_name: str = DATABASE_NAME
        self.collection_name: str = COLLECTION_NAME
        self.data_ingestion_dir: str = os.path.join(ARTIFACT_DIR, DATA_INGESTION_DIR_NAME)
        self.feature_store_file_path: str = os.path.join(self.data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME)
        self.training_file_path: str = os.path.join(self.data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
        self.testing_file_path: str = os.path.join(self.data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
        self.train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO


@dataclass
class DataValidationConfig:
    def __init__(self):
        self.data_validation_dir: str = os.path.join(ARTIFACT_DIR, DATA_VALIDATION_DIR_NAME)
        self.report_file_path: str = os.path.join(self.data_validation_dir, DATA_VALIDATION_REPORT_FILE_NAME)
        # ADDED: validation_report_file_path attribute
        self.validation_report_file_path: str = os.path.join(self.data_validation_dir, DATA_VALIDATION_REPORT_FILE_NAME)


@dataclass
class DataTransformationConfig:
    def __init__(self):
        self.data_transformation_dir: str = os.path.join(ARTIFACT_DIR, DATA_TRANSFORMATION_DIR_NAME)
        self.transformed_train_file_path: str = os.path.join(self.data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, TRAIN_FILE_NAME.replace("csv", "npy"))
        self.transformed_test_file_path: str = os.path.join(self.data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, TEST_FILE_NAME.replace("csv", "npy"))
        self.transformed_object_file_path: str = os.path.join(self.data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR, PREPROCSSING_OBJECT_FILE_NAME)


@dataclass
class ModelTrainerConfig:
    def __init__(self):
        self.model_trainer_dir: str = os.path.join(ARTIFACT_DIR, MODEL_TRAINER_DIR_NAME)
        self.trained_model_file_path: str = os.path.join(self.model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR, MODEL_TRAINER_TRAINED_MODEL_NAME)
        self.expected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE
        self.model_config_file_path: str = MODEL_TRAINER_MODEL_CONFIG_FILE_PATH
        self.model_type: str = MODEL_TRAINER_MODEL_TYPE
        self.n_estimators: int = MODEL_TRAINER_N_ESTIMATORS
        self.max_depth: int = MODEL_TRAINER_MAX_DEPTH
        self.learning_rate: float = MODEL_TRAINER_LEARNING_RATE
        self.subsample: float = MODEL_TRAINER_SUBSAMPLE
        self.colsample_bytree: float = MODEL_TRAINER_COLSAMPLE_BYTREE
        self.reg_alpha: float = MODEL_TRAINER_REG_ALPHA
        self.reg_lambda: float = MODEL_TRAINER_REG_LAMBDA
        self.gamma: float = MODEL_TRAINER_GAMMA
        self.min_child_weight: int = MODEL_TRAINER_MIN_CHILD_WEIGHT
        self.random_state: int = MODEL_TRAINER_RANDOM_STATE
        self.verbose: int = MODEL_TRAINER_VERBOSE
        # REMOVED: num_leaves and min_child_samples (LightGBM specific)


@dataclass
class ModelEvaluationConfig:
    def __init__(self):
        self.model_evaluation_dir: str = os.path.join(ARTIFACT_DIR, "model_evaluation")
        self.changed_threshold_score: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
        self.bucket_name: str = MODEL_BUCKET_NAME
        self.s3_model_key_path: str = MODEL_PUSHER_S3_KEY


@dataclass
class ModelPusherConfig:
    def __init__(self):
        self.model_pusher_dir: str = os.path.join(ARTIFACT_DIR, "model_pusher")
        self.bucket_name: str = MODEL_BUCKET_NAME
        self.s3_model_key_path: str = MODEL_PUSHER_S3_KEY