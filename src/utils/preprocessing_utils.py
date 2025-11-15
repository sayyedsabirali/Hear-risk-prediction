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
        REMOVED: Feature engineering as per your requirement
        Now returns the same dataframe without any feature engineering
        """
        try:
            logging.info("SKIPPING feature engineering as per best model configuration...")
            # Return original dataframe without any changes
            return df.copy()
            
        except Exception as e:
            logging.error(f"Error in preprocessing: {str(e)}")
            raise MyException(e, sys)

    @staticmethod
    def apply_preprocessing_transformations(df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing transformations (used in both training and prediction)
        Only basic preprocessing as per your best model parameters
        """
        try:
            df_processed = df.copy()
            
            # Step 1: Drop identifier columns
            df_processed = PreprocessingUtils._drop_id_columns(df_processed)
            
            # Step 2: Gender mapping
            df_processed = PreprocessingUtils._map_gender_column(df_processed)
            
            # Step 3: Handle missing values - EXACTLY as per best model
            df_processed = PreprocessingUtils._handle_missing_values_exact(df_processed)
            
            # Step 4: Label encoding
            df_processed = PreprocessingUtils._label_encode_columns(df_processed)
            
            # Step 5: Create dummy variables
            df_processed = PreprocessingUtils._create_dummy_columns(df_processed)
            
            # Step 6: Ensure all columns are numerical
            df_processed = PreprocessingUtils._ensure_numeric_columns(df_processed)
            
            # Step 7: Final safety check - replace inf/nan
            logging.info("Final safety check: Replacing any inf/nan values...")
            df_processed = df_processed.replace([np.inf, -np.inf], np.nan)
            df_processed = df_processed.fillna(0)
            
            logging.info(f"Preprocessing completed. Final shape: {df_processed.shape}")
            
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
            # Handle any unmapped values with 'unknown' category (2)
            df['gender'] = df['gender'].fillna(2).astype(int)
        return df

    @staticmethod
    def _handle_missing_values_exact(df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values EXACTLY as per best model parameters:
        - missing_categorical: unknown (replace with 'unknown' category)
        - missing_numerical: median (replace with median)
        """
        logging.info("Handling missing values EXACTLY as per best model parameters")
        
        # Categorical columns: Replace with 'unknown' category as per best model
        categorical_cols = ['insurance', 'marital_status', 'race', 'admission_type']
        for col in categorical_cols:
            if col in df.columns and df[col].isna().any():
                logging.info(f"   Filling missing categorical '{col}' with 'unknown' category")
                # For categorical, we'll fill with a string 'unknown' which will be encoded later
                df[col] = df[col].fillna('unknown')
        
        # Numerical columns: Replace with MEDIAN as per best model
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isna().any():
                median_value = df[col].median()
                # If median is nan or inf, use 0
                if pd.isna(median_value) or np.isinf(median_value):
                    median_value = 0
                logging.info(f"   Filling missing numerical '{col}' with median: {median_value}")
                df[col] = df[col].fillna(median_value)
        
        return df

    @staticmethod
    def _label_encode_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Label encode high cardinality columns"""
        logging.info("Label encoding high cardinality categorical features")
        high_cardinality = ['insurance', 'marital_status', 'race']
        for col in high_cardinality:
            if col in df.columns:
                # This will automatically handle 'unknown' category
                df[col] = pd.factorize(df[col])[0]
        return df

    @staticmethod
    def _create_dummy_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Create dummy variables for categorical features"""
        logging.info("Creating dummy variables for categorical features")
        
        # Define all possible categories for admission_type (from training data + unknown)
        admission_type_categories = [
            'AMBULATORY OBSERVATION',
            'DIRECT EMER.',
            'DIRECT OBSERVATION', 
            'ELECTIVE',
            'EU OBSERVATION',
            'EW EMER.',
            'OBSERVATION ADMIT',
            'SURGICAL SAME DAY ADMISSION',
            'URGENT',
            'unknown'  # Added unknown category
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
    def _ensure_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all columns are numeric"""
        logging.info("Ensuring all columns are numeric")
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isna().any():
                    df[col] = df[col].fillna(0)
        return df