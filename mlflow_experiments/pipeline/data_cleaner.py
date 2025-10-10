import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import warnings

class DataCleaner:
    def __init__(self):
        self.cleaning_report = {}
    
    def remove_extreme_outliers(self, df, numerical_cols):
        """Remove physically impossible medical values"""
        df_clean = df.copy()
        rows_removed = 0
        
        # Medical value ranges (realistic limits)
        medical_limits = {
            'creatinine_max': (0.1, 15.0),      # Normal: 0.6-1.2 mg/dL
            'glucose_max': (20, 1000),          # Normal: 70-140 mg/dL
            'ast_max': (5, 500),                # Normal: 10-40 U/L
            'alt_max': (5, 500),                # Normal: 7-56 U/L  
            'HR_max': (30, 250),                # Normal: 60-100 bpm
            'NBPs_max': (60, 250),              # Normal: 90-120 mmHg
            'NBPd_max': (30, 150),              # Normal: 60-80 mmHg
            'NBPm_max': (40, 200),              # Normal: 70-100 mmHg
            'anchor_age': (18, 100)             # Realistic age range
        }
        
        for col in numerical_cols:
            if col in medical_limits:
                lower, upper = medical_limits[col]
                mask = (df_clean[col] >= lower) & (df_clean[col] <= upper)
                outliers = (~mask).sum()
                df_clean = df_clean[mask | df_clean[col].isna()]
                rows_removed += outliers
                print(f"🚫 Removed {outliers} outliers from {col}")
        
        self.cleaning_report['outliers_removed'] = rows_removed
        return df_clean
    
    def handle_missing_values_advanced(self, df, threshold=0.5):
        """Advanced missing value handling"""
        df_clean = df.copy()
        
        # Remove columns with too many missing values
        missing_ratio = df_clean.isnull().sum() / len(df_clean)
        cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
        df_clean = df_clean.drop(columns=cols_to_drop)
        
        print(f"🗑️  Dropped columns with >{threshold*100}% missing: {cols_to_drop}")
        
        # For remaining columns, use different strategies
        numerical_cols = ['creatinine_max', 'glucose_max', 'ast_max', 'alt_max', 
                         'HR_max', 'NBPs_max', 'NBPd_max', 'NBPm_max', 'anchor_age']
        numerical_cols = [col for col in numerical_cols if col in df_clean.columns]
        
        # Use medical knowledge for imputation
        medical_medians = {
            'creatinine_max': 1.0,
            'glucose_max': 100,
            'ast_max': 25,
            'alt_max': 20,
            'HR_max': 75,
            'NBPs_max': 120, 
            'NBPd_max': 80,
            'NBPm_max': 93
        }
        
        for col in numerical_cols:
            if col in medical_medians and col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(medical_medians[col])
                print(f"🏥 Filled {col} with medical median: {medical_medians[col]}")
        
        # For categorical, fill with mode
        categorical_cols = ['gender', 'anchor_year_group', 'race', 
                           'marital_status', 'admission_type', 'insurance']
        categorical_cols = [col for col in categorical_cols if col in df_clean.columns]
        
        for col in categorical_cols:
            if col in df_clean.columns:
                mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Unknown"
                df_clean[col] = df_clean[col].fillna(mode_val)
        
        self.cleaning_report['columns_dropped'] = cols_to_drop
        self.cleaning_report['final_shape'] = df_clean.shape
        return df_clean
    
    def create_clinical_features(self, df):
        """Create clinically meaningful features"""
        df_eng = df.copy()
        
        # Hypertension categories
        if all(col in df.columns for col in ['NBPs_max', 'NBPd_max']):
            df_eng['hypertension_stage'] = np.select([
                (df_eng['NBPs_max'] < 120) & (df_eng['NBPd_max'] < 80),
                (df_eng['NBPs_max'] < 130) & (df_eng['NBPd_max'] < 85), 
                (df_eng['NBPs_max'] < 140) & (df_eng['NBPd_max'] < 90),
                (df_eng['NBPs_max'] >= 140) | (df_eng['NBPd_max'] >= 90)
            ], [0, 1, 2, 3], default=2)
        
        # Diabetes indicators
        if 'glucose_max' in df.columns:
            df_eng['diabetes_risk'] = np.select([
                df_eng['glucose_max'] < 100,
                df_eng['glucose_max'] < 126, 
                df_eng['glucose_max'] >= 126
            ], [0, 1, 2], default=1)
        
        # Kidney function
        if 'creatinine_max' in df.columns:
            df_eng['kidney_dysfunction'] = (df_eng['creatinine_max'] > 1.3).astype(int)
        
        # Liver function
        if all(col in df.columns for col in ['ast_max', 'alt_max']):
            df_eng['elevated_liver_enzymes'] = (
                (df_eng['ast_max'] > 40) | (df_eng['alt_max'] > 56)
            ).astype(int)
        
        # Age risk groups
        if 'anchor_age' in df.columns:
            df_eng['age_risk_group'] = pd.cut(df_eng['anchor_age'], 
                                            bins=[18, 45, 65, 75, 100],
                                            labels=[0, 1, 2, 3])
        
        # Heart rate abnormalities
        if 'HR_max' in df.columns:
            df_eng['hr_abnormal'] = ((df_eng['HR_max'] < 60) | (df_eng['HR_max'] > 100)).astype(int)
        
        print(f"🩺 Created clinical risk features")
        return df_eng
    
    def clean_dataset(self, df):
        """Complete data cleaning pipeline"""
        print("🚀 Starting aggressive data cleaning...")
        print(f"📊 Original shape: {df.shape}")
        
        # Step 1: Remove extreme outliers
        numerical_cols = ['creatinine_max', 'glucose_max', 'ast_max', 'alt_max', 
                         'HR_max', 'NBPs_max', 'NBPd_max', 'NBPm_max', 'anchor_age']
        df_clean = self.remove_extreme_outliers(df, numerical_cols)
        print(f"✅ After outlier removal: {df_clean.shape}")
        
        # Step 2: Handle missing values
        df_clean = self.handle_missing_values_advanced(df_clean, threshold=0.5)
        print(f"✅ After missing value handling: {df_clean.shape}")
        
        # Step 3: Create clinical features
        df_clean = self.create_clinical_features(df_clean)
        print(f"✅ After feature engineering: {df_clean.shape}")
        
        return df_clean