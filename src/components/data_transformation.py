import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file
from src.utils.preprocessing_utils import PreprocessingUtils

# FIX: Define identity function outside the class
def identity_function(x):
    """Identity function that returns the same data - can be pickled"""
    return x

class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        logging.info("Entered get_data_transformer_object method of DataTransformation class")
        try:
            # FIX: Use predefined identity function instead of lambda
            identity_transformer = FunctionTransformer(identity_function)
            
            logging.info("Using Identity Transformer (No scaling as per best model)")

            # Load schema configurations
            num_features = self._schema_config['num_features']
            logging.info("Columns loaded from schema.")

            # Creating preprocessor pipeline with identity transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("Identity", identity_transformer, num_features)
                ],
                remainder='passthrough'
            )

            # Wrapping everything in a single pipeline
            final_pipeline = Pipeline(steps=[("Preprocessor", preprocessor)])
            logging.info("Identity Pipeline Ready (No Scaling)!!")
            logging.info("Exited get_data_transformer_object method of DataTransformation class")
            return final_pipeline

        except Exception as e:
            logging.exception("Exception occurred in get_data_transformer_object method of DataTransformation class")
            raise MyException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Data Transformation Started !!!")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            # Load train and test data
            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info("Train-Test data loaded")

            # ✅ REUSE: Apply feature engineering from shared utils
            logging.info("Applying feature engineering to train data")
            train_df = PreprocessingUtils.apply_complete_feature_engineering(train_df)
            
            logging.info("Applying feature engineering to test data")
            test_df = PreprocessingUtils.apply_complete_feature_engineering(test_df)

            # Separate features and target
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            logging.info("Input and Target cols defined for both train and test df.")

            # ✅ REUSE: Apply preprocessing transformations from shared utils
            logging.info("Applying preprocessing transformations to train data")
            input_feature_train_df = PreprocessingUtils.apply_preprocessing_transformations(input_feature_train_df)
            
            logging.info("Applying preprocessing transformations to test data")
            input_feature_test_df = PreprocessingUtils.apply_preprocessing_transformations(input_feature_test_df)

            # CHANGED: Get identity transformer (no scaling)
            logging.info("Getting identity transformer (no scaling as per best model)")
            preprocessor = self.get_data_transformer_object()
            logging.info("Got the identity transformer object")

            # CHANGED: Just transform without any scaling
            logging.info("Transforming Training-data (no scaling)")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
        
            logging.info("Transforming Testing-data (no scaling)")
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            logging.info("Transformation done (no scaling applied)")

            # Create final arrays
            logging.info("Concatenating features and target")
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info("Feature-target concatenation done for train-test df.")

            # Save objects
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            logging.info("Saving transformation object and transformed files.")

            logging.info("Data transformation completed successfully (No scaling applied)")
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            logging.error(f"Error in data transformation: {str(e)}")
            raise MyException(e, sys) from e