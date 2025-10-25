import sys
import numpy as np
import pandas as pd
from src.exception import MyException
from src.logger import logging

class PreprocessingUtils:
    """Shared preprocessing utilities for both training and prediction"""
    
    @staticmethod
    def apply_complete_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply COMPLETE feature engineering (used in both training and prediction)
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
            
            def safe_divide(numerator, denominator, default=0.0):
                """Safe division that handles zero denominator and returns default"""
                result = np.where(denominator == 0, default, numerator / denominator)
                # Replace inf and -inf with default
                result = np.where(np.isinf(result), default, result)
                return result
            
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
                df_eng['shock_index'] = safe_divide(df_eng['heart_rate'], df_eng['bp_systolic'], default=0.0)
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
                df_eng['ventilation_perfusion_ratio'] = safe_divide(df_eng['respiratory_rate'], df_eng['spo2'], default=0.0)
            
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
                df_eng['bun_creatinine_ratio'] = safe_divide(df_eng['glucose'], df_eng['creatinine'], default=0.0)
            
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
                    df_eng['hr_percentage_max'] = safe_divide(df_eng['heart_rate'], df_eng['age_adjusted_hr_max'], default=0.0)
            
            # 7. VITAL SIGN INTERACTIONS
            logging.info("   Creating vital sign interactions...")
            vital_cols = ['heart_rate', 'respiratory_rate', 'spo2', 'temperature']
            available_vitals = [col for col in vital_cols if col in df_eng.columns]
            
            if len(available_vitals) >= 2:
                df_eng['vital_sign_mean'] = df_eng[available_vitals].mean(axis=1)
                df_eng['vital_sign_std'] = df_eng[available_vitals].std(axis=1).fillna(0)
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
                df_eng['hgb_wbc_ratio'] = safe_divide(df_eng['hemoglobin'], df_eng['white_blood_cells'], default=0.0)
            
            if all(col in df_eng.columns for col in ['heart_rate', 'respiratory_rate']):
                df_eng['hr_rr_ratio'] = safe_divide(df_eng['heart_rate'], df_eng['respiratory_rate'], default=0.0)
            
            if all(col in df_eng.columns for col in ['spo2', 'respiratory_rate']):
                df_eng['spo2_rr_ratio'] = safe_divide(df_eng['spo2'], df_eng['respiratory_rate'], default=0.0)
            
            # FINAL CHECK: Replace any remaining inf/nan values
            logging.info("   Final check: Replacing any remaining inf/nan values...")
            df_eng = df_eng.replace([np.inf, -np.inf], np.nan)
            df_eng = df_eng.fillna(0)
            
            new_features = len(df_eng.columns) - original_features
            logging.info(f"Advanced feature engineering completed: {new_features} new features")
            logging.info(f"   Total features: {len(df_eng.columns)}")
            
            return df_eng
            
        except Exception as e:
            logging.error(f"Error in feature engineering: {str(e)}")
            raise MyException(e, sys)

    @staticmethod
    def apply_preprocessing_transformations(df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing transformations (used in both training and prediction)
        """
        try:
            df_processed = df.copy()
            
            # Step 1: Drop identifier columns
            df_processed = PreprocessingUtils._drop_id_columns(df_processed)
            
            # Step 2: Gender mapping
            df_processed = PreprocessingUtils._map_gender_column(df_processed)
            
            # Step 3: Label encoding
            df_processed = PreprocessingUtils._label_encode_columns(df_processed)
            
            # Step 4: Create dummy variables
            df_processed = PreprocessingUtils._create_dummy_columns(df_processed)
            
            # Step 5: Handle missing values
            df_processed = PreprocessingUtils._handle_missing_values(df_processed)
            
            # Step 6: Ensure all columns are numerical
            df_processed = PreprocessingUtils._ensure_numeric_columns(df_processed)
            
            # Step 7: Final safety check - replace inf/nan
            logging.info("Final safety check: Replacing any inf/nan values...")
            df_processed = df_processed.replace([np.inf, -np.inf], np.nan)
            df_processed = df_processed.fillna(0)
            
            return df_processed
            
        except Exception as e:
            logging.error(f"Error in preprocessing: {str(e)}")
            raise MyException(e, sys)

    @staticmethod
    def _drop_id_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Drop identifier columns"""
        logging.info("Dropping identifier columns")
        drop_cols = ['_id', 'subject_id', 'hadm_id']
        for col in drop_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
        return df

    @staticmethod
    def _map_gender_column(df: pd.DataFrame) -> pd.DataFrame:
        """Map gender to binary"""
        logging.info("Mapping 'gender' column to binary values")
        if 'gender' in df.columns:
            df['gender'] = df['gender'].map({'F': 0, 'M': 1})
            # Handle any unmapped values
            df['gender'] = df['gender'].fillna(1).astype(int)
        return df

    @staticmethod
    def _label_encode_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Label encode high cardinality columns"""
        logging.info("Label encoding high cardinality categorical features")
        high_cardinality = ['insurance', 'marital_status', 'race']
        for col in high_cardinality:
            if col in df.columns:
                df[col] = pd.factorize(df[col])[0]
        return df

    @staticmethod
    def _create_dummy_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Create dummy variables for categorical features"""
        logging.info("Creating dummy variables for categorical features")
        
        # Define all possible categories for admission_type (from training data)
        admission_type_categories = [
            'AMBULATORY OBSERVATION',
            'DIRECT EMER.',
            'DIRECT OBSERVATION', 
            'ELECTIVE',
            'EU OBSERVATION',
            'EW EMER.',
            'OBSERVATION ADMIT',
            'SURGICAL SAME DAY ADMISSION',
            'URGENT'
        ]
        
        if 'admission_type' in df.columns:
            # Create dummies for all categories
            dummies = pd.get_dummies(df['admission_type'], prefix='admission_type')
            
            # Ensure all expected columns exist
            for category in admission_type_categories:
                col_name = f'admission_type_{category}'
                if col_name not in dummies.columns:
                    dummies[col_name] = 0
            
            # Drop first column (to match training behavior with drop_first=True)
            # Typically 'admission_type_AMBULATORY OBSERVATION' is dropped
            if 'admission_type_AMBULATORY OBSERVATION' in dummies.columns:
                dummies = dummies.drop(columns=['admission_type_AMBULATORY OBSERVATION'])
            
            # Concat and drop original column
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=['admission_type'])
            
        return df

    @staticmethod
    def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values"""
        logging.info("Handling missing values")
        
        # Fill categorical columns with mode
        categorical_cols = ['insurance', 'marital_status', 'gender', 'admission_type', 'race']
        for col in categorical_cols:
            if col in df.columns and df[col].isna().any():
                mode_val = df[col].mode()
                fill_value = mode_val[0] if len(mode_val) > 0 else 0
                df[col] = df[col].fillna(fill_value)
        
        # Fill numerical columns with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isna().any():
                median_value = df[col].median()
                # If median is nan or inf, use 0
                if pd.isna(median_value) or np.isinf(median_value):
                    median_value = 0
                df[col] = df[col].fillna(median_value)
        
        return df

    @staticmethod
    def _ensure_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all columns are numeric"""
        logging.info("Ensuring all columns are numeric")
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isna().any():
                    df[col] = df[col].fillna(0)
        return df