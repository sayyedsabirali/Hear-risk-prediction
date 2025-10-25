import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
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

            num_features = self._schema_config['num_features']
            mm_columns = self._schema_config['mm_columns']
            
            if isinstance(num_features, list) and len(num_features) > 0 and isinstance(num_features[0], list):
                num_features = num_features[0]  # Extract inner list
            if isinstance(mm_columns, list) and len(mm_columns) > 0 and isinstance(mm_columns[0], list):
                mm_columns = mm_columns[0]  # Extract inner list
            
        
            logging.info(f"Numerical features: {len(num_features)} - {type(num_features)}")
            logging.info(f"MinMax features: {len(mm_columns)} - {type(mm_columns)}")
            
            # Validate column types
            if not all(isinstance(col, (str, int)) for col in num_features):
                logging.error(f"Invalid numerical features: {num_features}")
                raise ValueError("Numerical features must be column names or indices")
            
            if not all(isinstance(col, (str, int)) for col in mm_columns):
                logging.error(f"Invalid MinMax features: {mm_columns}")
                raise ValueError("MinMax features must be column names or indices")

            # Creating preprocessor pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ("StandardScaler", numeric_transformer, num_features),
                    ("MinMaxScaler", min_max_scaler, mm_columns)
                ],
                remainder='passthrough',
                verbose_feature_names_out=False
            )

            # Wrapping everything in a single pipeline
            final_pipeline = Pipeline(steps=[("Preprocessor", preprocessor)])
            logging.info("Final Pipeline Ready!!")
            logging.info("Exited get_data_transformer_object method of DataTransformation class")
            return final_pipeline

        except Exception as e:
            logging.exception("Exception occurred in get_data_transformer_object method of DataTransformation class")
            raise MyException(e, sys) from e

    def _advanced_medical_feature_engineering(self, df):
        """Advanced medical feature engineering with safe mathematical operations"""
        logging.info("Advanced Medical Feature Engineering...")
        
        df_eng = df.copy()
        original_features = len(df_eng.columns)
        
        def safe_log1p(x):
            return np.log1p(np.where(x <= 0, 0.001, x))
        
        def safe_sqrt(x):
            return np.sqrt(np.where(x < 0, 0, x))
        
        # 1. CARDIAC-SPECIFIC FEATURES
        logging.info("   Creating cardiac-specific features...")
        
        if all(col in df_eng.columns for col in ['troponin_t', 'creatine_kinase_mb']):
            # Use safe operations
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
                df_eng[f'{col}_log'] = safe_log1p(df_eng[col])  # Use safe operation
                df_eng[f'{col}_sqrt'] = safe_sqrt(df_eng[col])  # Use safe operation
        
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

    def _handle_missing_values_advanced(self, df):
        """Advanced missing value handling without data removal"""
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
        logging.info("Mapping 'gender' column to binary values")
        if 'gender' in df.columns:
            df['gender'] = df['gender'].map({'F': 0, 'M': 1}).astype(int)
        return df

    def _create_dummy_columns(self, df):
        logging.info("Creating dummy variables for categorical features")
        low_cardinality = ['admission_type']
        for col in low_cardinality:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col])
        return df

    def _label_encode_columns(self, df):
        logging.info("Label encoding high cardinality categorical features")
        high_cardinality = ['insurance', 'marital_status', 'race']
        for col in high_cardinality:
            if col in df.columns:
                df[col] = pd.factorize(df[col])[0]
        return df

    def _drop_id_columns(self, df):
        logging.info("Dropping identifier columns")
        drop_cols = self._schema_config.get('drop_columns', [])
        # Handle list of lists in drop_columns
        if isinstance(drop_cols, list) and len(drop_cols) > 0 and isinstance(drop_cols[0], list):
            drop_cols = drop_cols[0]
        
        drop_cols = [col for col in drop_cols if col in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        return df

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Data Transformation Started !!!")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info("Train-Test data loaded")

            # Apply feature engineering to both train and test
            logging.info("Applying advanced medical feature engineering to train data")
            train_df = self._advanced_medical_feature_engineering(train_df)
            logging.info(f"Train data after feature engineering - Type: {type(train_df)}, Shape: {train_df.shape}")
            
            logging.info("Applying advanced medical feature engineering to test data")
            test_df = self._advanced_medical_feature_engineering(test_df)
            logging.info(f"Test data after feature engineering - Type: {type(test_df)}, Shape: {test_df.shape}")

            # Handle missing values
            logging.info("Handling missing values in train data")
            train_df = self._handle_missing_values_advanced(train_df)
            
            logging.info("Handling missing values in test data")
            test_df = self._handle_missing_values_advanced(test_df)

            # Separate features and target
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            logging.info("Input and Target cols defined for both train and test df.")

            # Apply transformations
            input_feature_train_df = self._drop_id_columns(input_feature_train_df)
            input_feature_train_df = self._map_gender_column(input_feature_train_df)
            input_feature_train_df = self._label_encode_columns(input_feature_train_df)
            input_feature_train_df = self._create_dummy_columns(input_feature_train_df)

            input_feature_test_df = self._drop_id_columns(input_feature_test_df)
            input_feature_test_df = self._map_gender_column(input_feature_test_df)
            input_feature_test_df = self._label_encode_columns(input_feature_test_df)
            input_feature_test_df = self._create_dummy_columns(input_feature_test_df)
            logging.info("Custom transformations applied to train and test data")

            # Align columns
            logging.info("Aligning columns between train and test datasets")
            missing_cols = set(input_feature_train_df.columns) - set(input_feature_test_df.columns)
            for col in missing_cols:
                input_feature_test_df[col] = 0
            
            input_feature_test_df = input_feature_test_df[input_feature_train_df.columns]
            logging.info("Column alignment completed")

            # Apply preprocessing
            logging.info("Starting data transformation with preprocessor")
            preprocessor = self.get_data_transformer_object()
            logging.info("Got the preprocessor object")

            logging.info("Initializing transformation for Training-data")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
        
            logging.info("Initializing transformation for Testing-data")
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            logging.info("Transformation done end to end to train-test df.")

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

            logging.info("Data transformation completed successfully")
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            logging.error(f"Error in data transformation: {str(e)}")
            raise MyException(e, sys) from e