import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file


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
            # Initialize transformers
            numeric_transformer = StandardScaler()
            min_max_scaler = MinMaxScaler()
            logging.info("Transformers Initialized: StandardScaler-MinMaxScaler")

            # Load schema configurations
            num_features = self._schema_config['num_features']
            mm_columns = self._schema_config['mm_columns']
            logging.info("Cols loaded from schema.")

            # Creating preprocessor pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ("StandardScaler", numeric_transformer, num_features),
                    ("MinMaxScaler", min_max_scaler, mm_columns)
                ],
                remainder='passthrough'  # Leaves other columns as they are
            )

            # Wrapping everything in a single pipeline
            final_pipeline = Pipeline(steps=[("Preprocessor", preprocessor)])
            logging.info("Final Pipeline Ready!!")
            logging.info("Exited get_data_transformer_object method of DataTransformation class")
            return final_pipeline

        except Exception as e:
            logging.exception("Exception occurred in get_data_transformer_object method of DataTransformation class")
            raise MyException(e, sys) from e

    def _map_gender_column(self, df):
        logging.info("Mapping 'gender' column to binary values")
        df['gender'] = df['gender'].map({
            'F': 0, 
            'M': 1
            }).astype(int)
        return df

    def _create_dummy_columns(self, df):
        logging.info("Creating dummy variables for categorical features")
        categorical_columns = self._schema_config['categorical_columns']
        
        # Remove gender since it's already mapped to binary
        categorical_columns = [col for col in categorical_columns if col != 'gender']
        
        # Create dummies only for remaining categorical columns
        if categorical_columns:
            df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
        
        return df

    def _align_columns(self, train_df, test_df):
        """Align columns between train and test data to avoid missing columns"""
        logging.info("Aligning columns between train and test data")
        
        # Get all columns from both datasets
        train_cols = set(train_df.columns)
        test_cols = set(test_df.columns)
        
        # Find missing columns in test data
        missing_in_test = train_cols - test_cols
        # Find missing columns in train data  
        missing_in_train = test_cols - train_cols
        
        # Add missing columns to test data with 0 values
        for col in missing_in_test:
            test_df[col] = 0
        
        # Add missing columns to train data with 0 values
        for col in missing_in_train:
            train_df[col] = 0
        
        # Ensure both have same column order
        common_cols = sorted(list(train_cols.union(test_cols)))
        train_df = train_df.reindex(columns=common_cols, fill_value=0)
        test_df = test_df.reindex(columns=common_cols, fill_value=0)
        
        logging.info(f"Added {len(missing_in_test)} missing columns to test data")
        logging.info(f"Added {len(missing_in_train)} missing columns to train data")
        
        return train_df, test_df


    def _rename_columns(self, df):
        logging.info("Renaming specific columns with special characters")
        # Replace spaces and special characters in column names
        df.columns = df.columns.str.replace(' ', '_').str.replace('-', '_').str.replace('/', '_')
        
        # Convert boolean columns to int
        for col in df.columns:
            if df[col].dtype == 'bool':
                df[col] = df[col].astype('int')
        return df

    def _drop_id_column(self, df):
        logging.info("Dropping specified columns")
        drop_cols = self._schema_config['drop_columns']  # assume list
        drop_cols = [col for col in drop_cols if col in df.columns]  # filter only valid columns
        if drop_cols:
            df = df.drop(columns=drop_cols)
        return df

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Data Transformation Started !!!")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            # Load train and test data
            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info("Train-Test data loaded")

            # FIX: Use 'heart_attack' as target column
            input_feature_train_df = train_df.drop(columns=['heart_attack'], axis=1)
            target_feature_train_df = train_df['heart_attack']

            input_feature_test_df = test_df.drop(columns=['heart_attack'], axis=1)
            target_feature_test_df = test_df['heart_attack']
            logging.info("Input and Target cols defined for both train and test df.")

            # Apply custom transformations in specified sequence
            input_feature_train_df = self._map_gender_column(input_feature_train_df)
            input_feature_train_df = self._drop_id_column(input_feature_train_df)
            input_feature_train_df = self._create_dummy_columns(input_feature_train_df)
            input_feature_train_df = self._rename_columns(input_feature_train_df)

            input_feature_test_df = self._map_gender_column(input_feature_test_df)
            input_feature_test_df = self._drop_id_column(input_feature_test_df)
            input_feature_test_df = self._create_dummy_columns(input_feature_test_df)
            input_feature_test_df = self._rename_columns(input_feature_test_df)
            
            # NEW: Align columns between train and test
            input_feature_train_df, input_feature_test_df = self._align_columns(
                input_feature_train_df, input_feature_test_df
            )
            logging.info("Custom transformations applied to train and test data")

            logging.info("Starting data transformation")
            preprocessor = self.get_data_transformer_object()
            logging.info("Got the preprocessor object")

            logging.info("Initializing transformation for Training-data")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            logging.info("Initializing transformation for Testing-data")
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            logging.info("Transformation done end to end to train-test df.")

            # REMOVED SMOTEENN PART - Directly use the transformed arrays
            logging.info("Using transformed data directly (SMOTEENN removed)")
            input_feature_train_final = input_feature_train_arr
            target_feature_train_final = target_feature_train_df
            
            input_feature_test_final = input_feature_test_arr  
            target_feature_test_final = target_feature_test_df

            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]
            logging.info("feature-target concatenation done for train-test df.")

            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            logging.info("Saving transformation object and transformed files.")

            logging.info("Data transformation completed successfully")
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise MyException(e, sys) from e