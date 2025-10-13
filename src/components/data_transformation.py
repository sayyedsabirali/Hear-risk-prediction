import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, FEATURE_SELECTION_COUNT
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file

import lightgbm as lgb
from sklearn.feature_selection import SelectKBest, f_classif


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

    def advanced_medical_feature_engineering(self, df):
        """EXACT MLflow feature engineering"""
        logging.info("Advanced Medical Feature Engineering Started")
        
        df_eng = df.copy()
        original_features = len(df_eng.columns)
        
        # Remove identifiers
        temp_df = df_eng.copy()
        drop_columns = ['subject_id', 'hadm_id', '_id']
        for col in drop_columns:
            if col in temp_df.columns:
                temp_df.drop(col, axis=1, inplace=True)
        
        # ✅ 1. CARDIAC FEATURES (MLflow exact)
        if all(col in temp_df.columns for col in ['troponin_t', 'creatine_kinase_mb']):
            temp_df['cardiac_biomarker_product'] = np.log1p(temp_df['troponin_t']) * np.log1p(temp_df['creatine_kinase_mb'])
            temp_df['cardiac_risk_index'] = (temp_df['troponin_t'] * 100) + (temp_df['creatine_kinase_mb'] * 10)
            temp_df['elevated_troponin'] = (temp_df['troponin_t'] > 0.1).astype(int)
            temp_df['elevated_ckmb'] = (temp_df['creatine_kinase_mb'] > 5).astype(int)
            temp_df['both_biomarkers_elevated'] = ((temp_df['troponin_t'] > 0.1) & (temp_df['creatine_kinase_mb'] > 5)).astype(int)
        
        # ✅ 2. HEMODYNAMIC (MLflow exact)
        if all(col in temp_df.columns for col in ['bp_systolic', 'bp_diastolic', 'heart_rate']):
            temp_df['mean_arterial_pressure'] = (temp_df['bp_systolic'] + 2 * temp_df['bp_diastolic']) / 3
            temp_df['pulse_pressure'] = temp_df['bp_systolic'] - temp_df['bp_diastolic']
            temp_df['rate_pressure_product'] = temp_df['heart_rate'] * temp_df['bp_systolic'] / 100
            temp_df['shock_index'] = temp_df['heart_rate'] / temp_df['bp_systolic']
            temp_df['hemodynamic_instability'] = (
                (temp_df['bp_systolic'] < 90) | 
                (temp_df['heart_rate'] > 120) | 
                (temp_df['mean_arterial_pressure'] < 65)
            ).astype(int)
        
        # ✅ 3. OXYGENATION (MLflow exact) - YEH MISSING THA
        if all(col in temp_df.columns for col in ['spo2', 'respiratory_rate', 'hemoglobin']):
            temp_df['oxygen_saturation_ratio'] = temp_df['spo2'] / 100
            temp_df['oxygen_content'] = (temp_df['hemoglobin'] * 1.34 * temp_df['spo2'] / 100) + (0.003 * temp_df['spo2'])
            temp_df['respiratory_distress'] = (
                (temp_df['spo2'] < 92) | (temp_df['respiratory_rate'] > 24)
            ).astype(int)
            temp_df['ventilation_perfusion_ratio'] = temp_df['respiratory_rate'] / (temp_df['spo2'] + 0.1)  # ✅ ADDED
        
        # ✅ 4. METABOLIC (MLflow exact)
        if all(col in temp_df.columns for col in ['creatinine', 'potassium', 'sodium', 'glucose']):
            temp_df['renal_function_score'] = temp_df['creatinine'] * temp_df['potassium']
            temp_df['electrolyte_imbalance'] = (
                (temp_df['sodium'] < 135) | (temp_df['sodium'] > 145) |
                (temp_df['potassium'] < 3.5) | (temp_df['potassium'] > 5.0)
            ).astype(int)
            temp_df['hyperglycemia'] = (temp_df['glucose'] > 180).astype(int)
            temp_df['hypoglycemia'] = (temp_df['glucose'] < 70).astype(int)
            temp_df['bun_creatinine_ratio'] = temp_df['glucose'] / (temp_df['creatinine'] + 0.1)  # ✅ ADDED
        
        # ✅ 5. INFLAMMATORY (MLflow exact)
        if 'white_blood_cells' in temp_df.columns:
            temp_df['wbc_elevated'] = (temp_df['white_blood_cells'] > 11).astype(int)
            temp_df['wbc_low'] = (temp_df['white_blood_cells'] < 4).astype(int)
            temp_df['leukocytosis'] = (temp_df['white_blood_cells'] > 15).astype(int)
        
        if 'hemoglobin' in temp_df.columns:
            temp_df['anemia'] = (temp_df['hemoglobin'] < 12).astype(int)
            temp_df['severe_anemia'] = (temp_df['hemoglobin'] < 8).astype(int)
            temp_df['polycythemia'] = (temp_df['hemoglobin'] > 16).astype(int)
        
        # ✅ 6. AGE FEATURES (MLflow exact)
        if 'anchor_age' in temp_df.columns:
            temp_df['age_squared'] = temp_df['anchor_age'] ** 2
            temp_df['age_cubed'] = temp_df['anchor_age'] ** 3  # ✅ ADDED
            temp_df['elderly'] = (temp_df['anchor_age'] >= 65).astype(int)
            temp_df['very_elderly'] = (temp_df['anchor_age'] >= 80).astype(int)
            temp_df['young_adult'] = ((temp_df['anchor_age'] >= 18) & (temp_df['anchor_age'] <= 40)).astype(int)  # ✅ ADDED
            
            # Age-adjusted thresholds
            if 'heart_rate' in temp_df.columns:
                temp_df['age_adjusted_hr_max'] = 220 - temp_df['anchor_age']
                temp_df['hr_percentage_max'] = temp_df['heart_rate'] / temp_df['age_adjusted_hr_max']
        
        # ✅ 7. VITAL SIGN INTERACTIONS (MLflow exact)
        vital_cols = ['heart_rate', 'respiratory_rate', 'spo2', 'temperature']
        available_vitals = [col for col in vital_cols if col in temp_df.columns]
        
        if len(available_vitals) >= 2:
            temp_df['vital_sign_mean'] = temp_df[available_vitals].mean(axis=1)
            temp_df['vital_sign_std'] = temp_df[available_vitals].std(axis=1)
            temp_df['vital_sign_range'] = temp_df[available_vitals].max(axis=1) - temp_df[available_vitals].min(axis=1)
            
            # Early Warning Score-like features
            temp_df['ews_score'] = 0
            if 'heart_rate' in temp_df.columns:
                temp_df['ews_score'] += ((temp_df['heart_rate'] < 50) | (temp_df['heart_rate'] > 100)).astype(int)
            if 'respiratory_rate' in temp_df.columns:
                temp_df['ews_score'] += ((temp_df['respiratory_rate'] < 12) | (temp_df['respiratory_rate'] > 20)).astype(int)
            if 'spo2' in temp_df.columns:
                temp_df['ews_score'] += (temp_df['spo2'] < 95).astype(int)
            if 'temperature' in temp_df.columns:
                temp_df['ews_score'] += ((temp_df['temperature'] < 36) | (temp_df['temperature'] > 38)).astype(int)
        
        # ✅ 8. CLINICAL RISK SCORES (MLflow exact)
        temp_df['timi_like_score'] = 0
        if 'anchor_age' in temp_df.columns:
            temp_df['timi_like_score'] += (temp_df['anchor_age'] > 65).astype(int)
        if all(col in temp_df.columns for col in ['troponin_t', 'creatine_kinase_mb']):
            temp_df['timi_like_score'] += ((temp_df['troponin_t'] > 0.1) | (temp_df['creatine_kinase_mb'] > 5)).astype(int)
        if 'heart_rate' in temp_df.columns:
            temp_df['timi_like_score'] += (temp_df['heart_rate'] > 100).astype(int)
        if 'bp_systolic' in temp_df.columns:
            temp_df['timi_like_score'] += (temp_df['bp_systolic'] < 100).astype(int)
        
        # ✅ 9. POLYNOMIAL FEATURES (MLflow exact)
        key_continuous = ['troponin_t', 'creatine_kinase_mb', 'hemoglobin', 'heart_rate', 'anchor_age']
        for col in key_continuous:
            if col in temp_df.columns:
                temp_df[f'{col}_squared'] = temp_df[col] ** 2
                temp_df[f'{col}_cubed'] = temp_df[col] ** 3
                temp_df[f'{col}_log'] = np.log1p(temp_df[col])
                temp_df[f'{col}_sqrt'] = np.sqrt(temp_df[col] + 0.1)
        
        # ✅ 10. RATIO FEATURES (MLflow exact)
        if all(col in temp_df.columns for col in ['hemoglobin', 'white_blood_cells']):
            temp_df['hgb_wbc_ratio'] = temp_df['hemoglobin'] / (temp_df['white_blood_cells'] + 0.1)
        
        if all(col in temp_df.columns for col in ['heart_rate', 'respiratory_rate']):
            temp_df['hr_rr_ratio'] = temp_df['heart_rate'] / (temp_df['respiratory_rate'] + 0.1)
        
        if all(col in temp_df.columns for col in ['spo2', 'respiratory_rate']):
            temp_df['spo2_rr_ratio'] = temp_df['spo2'] / (temp_df['respiratory_rate'] + 0.1)
        
        new_features = len(temp_df.columns) - original_features
        logging.info(f"Advanced feature engineering completed: {new_features} new features")
        return temp_df

    def handle_missing_values(self, df):
        """Handle missing values - EXACT MLflow approach"""
        logging.info("Handling Missing Values")
        
        df_imputed = df.copy()
        
        # Categorical columns
        categorical_cols = ['insurance', 'marital_status', 'gender', 'admission_type', 'race']
        for col in categorical_cols:
            if col in df_imputed.columns and df_imputed[col].isna().any():
                mode_val = df_imputed[col].mode()
                df_imputed[col] = df_imputed[col].fillna(mode_val[0] if len(mode_val) > 0 else 'Unknown')
        
        # Numerical columns - use median (same as MLflow)
        numerical_cols = df_imputed.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_imputed[col].isna().any():
                median_val = df_imputed[col].median()
                df_imputed[col] = df_imputed[col].fillna(median_val)
        
        return df_imputed

    def encode_features(self, df):
        """Encode categorical features - EXACT MLflow approach"""
        logging.info("Encoding Features")
        
        df_encoded = df.copy()
        
        # One-hot encoding for low cardinality
        low_cardinality = ['gender', 'admission_type']
        for col in low_cardinality:
            if col in df_encoded.columns:
                dummies = pd.get_dummies(df_encoded[col], prefix=col)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
        
        # Label encoding for high cardinality
        high_cardinality = ['insurance', 'marital_status', 'race']
        for col in high_cardinality:
            if col in df_encoded.columns:
                df_encoded[col] = pd.factorize(df_encoded[col])[0]
        
        # Remove original categorical columns
        df_encoded.drop(low_cardinality, axis=1, inplace=True, errors='ignore')
        
        return df_encoded

    def ensure_numeric_data(self, X):
        """Ensure all data is numeric for LightGBM"""
        logging.info("Ensuring all data is numeric")
        
        X_clean = X.copy()
        
        # Convert object columns to numeric
        for col in X_clean.columns:
            if X_clean[col].dtype == 'object':
                # Try to convert to numeric first
                X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce')
                # If still object, use factorize
                if X_clean[col].dtype == 'object':
                    X_clean[col] = pd.factorize(X_clean[col])[0]
            
            # Fill NaN values with median
            if X_clean[col].isna().any():
                X_clean[col] = X_clean[col].fillna(X_clean[col].median())
        
        return X_clean

    def advanced_feature_selection(self, X, y):
        """Select EXACTLY 30 features - MLflow exact approach"""
        logging.info(f"Advanced Feature Selection for {FEATURE_SELECTION_COUNT} features")
        
        # ✅ FIRST ensure all data is numeric
        X_clean = self.ensure_numeric_data(X)
        
        selected_features = set()
        
        # Method 1: LightGBM importance (12 features)
        lgb_selector = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        lgb_selector.fit(X_clean, y)
        lgb_importance = pd.Series(lgb_selector.feature_importances_, index=X_clean.columns)
        selected_features.update(lgb_importance.nlargest(12).index.tolist())
        # Method 2: Correlation (12 features)  
        correlations = X_clean.corrwith(y).abs()
        selected_features.update(correlations.nlargest(12).index.tolist())
        
        # Method 3: Statistical (12 features)
        selector = SelectKBest(score_func=f_classif, k=min(12, X_clean.shape[1]))
        selector.fit(X_clean, y)
        statistical_scores = pd.Series(selector.scores_, index=X_clean.columns)
        selected_features.update(statistical_scores.nlargest(12).index.tolist())
        
        # ✅ CRITICAL: Ensure exactly 30 features
        if len(selected_features) < FEATURE_SELECTION_COUNT:
            remaining = FEATURE_SELECTION_COUNT - len(selected_features)
            # Add remaining top features from LightGBM
            additional_features = lgb_importance.nlargest(FEATURE_SELECTION_COUNT + 10).index.tolist()
            for feature in additional_features:
                if feature not in selected_features and len(selected_features) < FEATURE_SELECTION_COUNT:
                    selected_features.add(feature)
        
        final_features = list(selected_features)[:FEATURE_SELECTION_COUNT]
        logging.info(f"Selected {len(final_features)} features (Target: 30)")
        return X_clean[final_features]

    def get_data_transformer_object(self) -> Pipeline:
        """EXACT MLflow approach - Simple StandardScaler"""
        logging.info("Creating data transformer object - MLflow Exact Approach")
        try:
            # Simple StandardScaler - no complex ColumnTransformer
            scaler = StandardScaler()
            
            # Simple pipeline
            pipeline = Pipeline([
                ('scaler', scaler)
            ])
            
            logging.info("StandardScaler pipeline ready (MLflow exact approach)")
            return pipeline

        except Exception as e:
            logging.exception("Error creating transformer")
            raise MyException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Starting Data Transformation - EXACT MLflow Approach")
            
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            # Load data
            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)

            # ✅ EXACT MLflow Successful Pipeline
            logging.info("1. Advanced Medical Feature Engineering")
            train_df = self.advanced_medical_feature_engineering(train_df)
            test_df = self.advanced_medical_feature_engineering(test_df)

            logging.info("2. Handling Missing Values")
            train_df = self.handle_missing_values(train_df)
            test_df = self.handle_missing_values(test_df)

            logging.info("3. Encoding Features")
            train_df = self.encode_features(train_df)
            test_df = self.encode_features(test_df)

            # Separate features and target
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            # ✅ 4. Feature Selection (30 features) - EXACT MLflow
            logging.info("4. Advanced Feature Selection")
            input_feature_train_selected = self.advanced_feature_selection(
                input_feature_train_df, target_feature_train_df
            )
            
            # Ensure test data has same features
            common_features = [col for col in input_feature_train_selected.columns if col in input_feature_test_df.columns]
            input_feature_test_selected = input_feature_test_df[common_features]
            
            # Add missing features to test data
            missing_features = set(input_feature_train_selected.columns) - set(common_features)
            for feature in missing_features:
                input_feature_test_selected[feature] = 0
            
            input_feature_test_selected = input_feature_test_selected[input_feature_train_selected.columns]

            # ✅ 5. EXACT MLflow Preprocessing Steps
            logging.info("5. Applying MLflow Exact Preprocessing")
            
            # Apply StandardScaler (same as MLflow)
            preprocessor = self.get_data_transformer_object()
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_selected)
            input_feature_test_arr = preprocessor.transform(input_feature_test_selected)

            # Create final arrays
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save objects
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

            logging.info(f"Transformation completed. Final train shape: {train_arr.shape}")

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            logging.error("Error in data transformation")
            raise MyException(e, sys) from e