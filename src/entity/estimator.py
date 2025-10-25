import sys
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.pipeline import Pipeline

from src.exception import MyException
from src.logger import logging

class TargetValueMapping:
    def __init__(self):
        self.yes: int = 0
        self.no: int = 1
    
    def _asdict(self):
        return self.__dict__
    
    def reverse_mapping(self):
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))

class MyModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object



    def _apply_feature_engineering(self, df):
        """Advanced medical feature engineering with safe mathematical operations"""
        logging.info("Advanced Medical Feature Engineering...")
        
        df_eng = df.copy()
        original_features = len(df_eng.columns)
        
        # ✅ FIX: Safe mathematical operations to avoid warnings
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

    def predict(self, dataframe: pd.DataFrame) -> DataFrame:
        try:
            logging.info("Starting prediction process with raw data...")
            
            # STEP 1: Apply EXACT SAME feature engineering as training
            logging.info("Step 1: Applying feature engineering...")
            df_processed = self._apply_feature_engineering(dataframe)
            
            # STEP 2: Apply EXACT SAME preprocessing steps as training
            logging.info("Step 2: Applying preprocessing steps...")
            df_processed = self._apply_preprocessing_steps(df_processed)
            
            # STEP 3: Apply scaling transformations
            logging.info("Step 3: Applying scaling transformations...")
            transformed_feature = self.preprocessing_object.transform(df_processed)
            
            # STEP 4: Make predictions
            logging.info("Step 4: Making predictions...")
            predictions = self.trained_model_object.predict(transformed_feature)
            
            logging.info(f"Prediction completed. Total predictions: {len(predictions)}")
            return predictions

        except Exception as e:
            logging.error("Error occurred in predict method", exc_info=True)
            raise MyException(e, sys) from e