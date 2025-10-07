import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
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
        try:
            min_max_scaler = MinMaxScaler()
            logging.info("Transformer Initialized: MinMaxScaler")
            mm_columns = self._schema_config.get('mm_columns', [])
            
            if not mm_columns:
                raise ValueError("mm_columns not found in schema.yaml file!")
            
            logging.info(f"MinMaxScaler will be applied to {len(mm_columns)} columns: {mm_columns}")
            preprocessor = ColumnTransformer(
                transformers=[
                    ("MinMaxScaler", min_max_scaler, mm_columns)
                ],
                remainder='passthrough'  # Keep other columns as they are
            )

            # Wrapping everything in a single pipeline
            final_pipeline = Pipeline(steps=[("Preprocessor", preprocessor)])
            logging.info("Final Pipeline Ready!!")
            return final_pipeline

        except Exception as e:
            logging.exception("Exception occurred in get_data_transformer_object method")
            raise MyException(e, sys) from e

    def _handle_missing_values(self, df):
        """Fill missing values with appropriate strategies"""
        logging.info("Handling missing values")
        
        # Numeric columns - fill with median
        numeric_cols = self._schema_config.get('numerical_columns', [])
        
        for col in numeric_cols:
            if col in df.columns and df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logging.info(f"Filled {col} missing values with median: {median_val}")
        
        # Categorical columns - fill with mode or 'Unknown'
        categorical_cols = self._schema_config.get('categorical_columns', [])
        for col in categorical_cols:
            if col in df.columns and df[col].isnull().any():
                if not df[col].mode().empty:
                    mode_val = df[col].mode()[0]
                    df[col] = df[col].fillna(mode_val)
                    logging.info(f"Filled {col} missing values with mode: {mode_val}")
                else:
                    df[col] = df[col].fillna('Unknown')
                    logging.info(f"Filled {col} missing values with 'Unknown'")
        
        return df

    def _remove_outliers(self, df):
        """Remove outliers using IQR method (only for training data)"""
        logging.info("Removing outliers using IQR method")
        
        numeric_cols = ['creatinine_max', 'glucose_max', 'ast_max', 'alt_max', 
                       'HR_max', 'NBPs_max', 'NBPd_max', 'NBPm_max']
        
        initial_shape = df.shape[0]
        
        for col in numeric_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                before_count = df.shape[0]
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                removed = before_count - df.shape[0]
                logging.info(f"{col}: removed {removed} rows ({(removed/initial_shape)*100:.2f}%)")
        
        logging.info(f"After outlier removal: {df.shape}")
        return df

    def _map_gender_column(self, df):
        """Map gender to binary values"""
        logging.info("Mapping 'gender' column to binary values")
        if 'gender' in df.columns:
            # Map F=0, M=1, handle any other values as -1
            df['gender'] = df['gender'].map({'F': 0, 'M': 1}).fillna(-1).astype(int)
            logging.info("Gender mapping completed: F=0, M=1")
        return df

    def _encode_anchor_year_group(self, df):
        """Convert anchor_year_group to numeric by extracting start year"""
        logging.info("Encoding anchor_year_group")
        if 'anchor_year_group' in df.columns:
            # Extract start year from format "2008 - 2010"
            df['anchor_year'] = df['anchor_year_group'].str.extract(r'(\d{4})').astype(float)
            df = df.drop(columns=['anchor_year_group'])
            logging.info("Extracted anchor_year from anchor_year_group and dropped original column")
        return df

    def _create_dummy_columns(self, df):
        """Create dummy variables for categorical features"""
        logging.info("Creating dummy variables for categorical features")
        
        # Get categorical columns from schema, excluding gender and anchor_year_group (already handled)
        categorical_cols = ['race', 'marital_status', 'admission_type', 'insurance']
        existing_cats = [col for col in categorical_cols if col in df.columns]
        
        if existing_cats:
            df = pd.get_dummies(df, columns=existing_cats, drop_first=True)
            logging.info(f"Created dummy variables for: {existing_cats}")
        
        return df

    def _drop_id_column(self, df):
        """Drop unnecessary columns like _id"""
        logging.info("Dropping unnecessary columns")
        drop_cols = self._schema_config.get('drop_columns', [])
        drop_cols = [col for col in drop_cols if col in df.columns]
        
        if drop_cols:
            df = df.drop(columns=drop_cols)
            logging.info(f"Dropped columns: {drop_cols}")
        
        return df
    
    def _handle_target_missing_values(self, df, target_col):
        """Handle missing values in target column and ensure numeric dtype"""
        logging.info(f"Checking target column '{target_col}' for missing values")
        
        if target_col in df.columns:
            missing_count = df[target_col].isnull().sum()
            if missing_count > 0:
                logging.warning(f"Found {missing_count} missing values in target column '{target_col}'")
                # Drop rows with missing target (recommended)
                df = df.dropna(subset=[target_col])
                logging.info(f"Dropped {missing_count} rows with missing target values")
            else:
                logging.info(f"No missing values found in target column '{target_col}'")
            
            # CRITICAL: Ensure target is numeric (int/float) not object dtype
            if df[target_col].dtype == 'object':
                logging.warning(f"Target column is object dtype, converting to numeric")
                df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
                # Drop any rows that couldn't be converted (became NaN)
                new_missing = df[target_col].isnull().sum()
                if new_missing > 0:
                    df = df.dropna(subset=[target_col])
                    logging.info(f"Dropped {new_missing} rows with non-numeric target values")
            
            # Convert to int for binary classification
            df[target_col] = df[target_col].astype(int)
            logging.info(f"Target column dtype: {df[target_col].dtype}")
        
        return df

    def initiate_data_transformation(self, apply_outlier_removal=True) -> DataTransformationArtifact:
        try:
            logging.info("DATA TRANSFORMATION STARTED !!!")
            
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            # Load train and test data
            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info(f"Train data loaded: {train_df.shape}")
            logging.info(f"Test data loaded: {test_df.shape}")
            logging.info("TRANSFORMING TRAIN DATA")
            
            # Step 1: Handle missing values in FEATURES
            train_df = self._handle_missing_values(train_df)
            
            # Step 1.5: Handle missing values in TARGET
            train_df = self._handle_target_missing_values(train_df, TARGET_COLUMN)
            
            # Step 2: Remove outliers (only for train data)
            if apply_outlier_removal:
                train_df = self._remove_outliers(train_df)
            
            # Step 3: Separate features and target
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            logging.info(f"Target column separated. Class distribution:\n{target_feature_train_df.value_counts()}")
            
            # Step 4: Apply custom transformations
            input_feature_train_df = self._drop_id_column(input_feature_train_df)
            input_feature_train_df = self._map_gender_column(input_feature_train_df)
            input_feature_train_df = self._encode_anchor_year_group(input_feature_train_df)
            input_feature_train_df = self._create_dummy_columns(input_feature_train_df)
            logging.info(f"Train features after custom transformations: {input_feature_train_df.shape}")

            # ============ TEST DATA TRANSFORMATION ============
            logging.info("TRANSFORMING TEST DATA")
            
            # Step 1: Handle missing values in FEATURES
            test_df = self._handle_missing_values(test_df)
            
            # Step 1.5: Handle missing values in TARGET
            test_df = self._handle_target_missing_values(test_df, TARGET_COLUMN)
            
            # Step 2: Separate features and target
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            logging.info(f"Target column separated. Class distribution:\n{target_feature_test_df.value_counts()}")
            
            # Step 3: Apply custom transformations
            input_feature_test_df = self._drop_id_column(input_feature_test_df)
            input_feature_test_df = self._map_gender_column(input_feature_test_df)
            input_feature_test_df = self._encode_anchor_year_group(input_feature_test_df)
            input_feature_test_df = self._create_dummy_columns(input_feature_test_df)
            logging.info(f"Test features after custom transformations: {input_feature_test_df.shape}")
            
            # Add missing columns to test that exist in train
            missing_cols = set(input_feature_train_df.columns) - set(input_feature_test_df.columns)
            for col in missing_cols:
                input_feature_test_df[col] = 0
                logging.info(f"Added missing column '{col}' to test data with value 0")
            
            # Reorder test columns to match train
            input_feature_test_df = input_feature_test_df[input_feature_train_df.columns]
            logging.info(f"Columns aligned. Train: {input_feature_train_df.shape}, Test: {input_feature_test_df.shape}")

            logging.info("APPLYING MINMAXSCALER")
            
            preprocessor = self.get_data_transformer_object()
            
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            logging.info("MinMaxScaler transformation completed")

            # ============ CONCATENATE FEATURES AND TARGET ============
            target_train_arr = np.array(target_feature_train_df, dtype=np.int32)
            target_test_arr = np.array(target_feature_test_df, dtype=np.int32)
            
            train_arr = np.c_[input_feature_train_arr, target_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_test_arr]
            
            logging.info(f"Final train array: {train_arr.shape}")
            logging.info(f"Final test array: {test_arr.shape}")
            logging.info(f"Train target distribution: {np.unique(target_train_arr, return_counts=True)}")
            logging.info(f"Test target distribution: {np.unique(target_test_arr, return_counts=True)}")

            # ============ SAVE OBJECTS ============
            logging.info("Saving transformation objects and arrays...")
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            logging.info("All objects saved successfully")
            logging.info("DATA TRANSFORMATION COMPLETED SUCCESSFULLY")
            
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            logging.exception("Exception occurred in data transformation")
            raise MyException(e, sys) from e