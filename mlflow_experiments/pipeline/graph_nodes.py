import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import PowerTransformer, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score
import warnings
import os

class DataProcessor:
    def __init__(self):
        self.processed_data = None
        self.target_column = None
        
    def set_target_column(self, target_col):
        self.target_column = target_col
        
    def handle_duplicates(self, df: pd.DataFrame, method: str = "keep_first") -> pd.DataFrame:
        """Handle duplicate rows"""
        initial_shape = df.shape[0]
        
        if method == "keep_first":
            df_clean = df.drop_duplicates(keep='first')
        elif method == "keep_last":
            df_clean = df.drop_duplicates(keep='last')
        elif method == "drop_all":
            df_clean = df.drop_duplicates(keep=False)
        else:
            df_clean = df.copy()
            
        final_shape = df_clean.shape[0]
        duplicates_removed = initial_shape - final_shape
        
        return df_clean, {"duplicates_removed": duplicates_removed}
    
    def handle_missing_values(self, df: pd.DataFrame, num_method: str, cat_method: str, 
                            num_cols: list, cat_cols: list) -> pd.DataFrame:
        """Handle missing values with different strategies"""
        df_clean = df.copy()
        missing_info = {}
        
        # Numerical columns - FIXED: Remove inplace=True
        for col in num_cols:
            if col in df_clean.columns and df_clean[col].isna().sum() > 0:
                if num_method == "mean":
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                elif num_method == "median":
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                elif num_method == "knn":
                    # Implement KNN imputation
                    knn_imputer = KNNImputer(n_neighbors=5)
                    df_clean[col] = knn_imputer.fit_transform(df_clean[[col]]).ravel()
                elif num_method == "drop":
                    df_clean = df_clean.dropna(subset=[col])
                
                missing_info[f"{col}_missing_filled"] = df_clean[col].isna().sum()
        
        # Categorical columns - FIXED: Remove inplace=True  
        for col in cat_cols:
            if col in df_clean.columns and df_clean[col].isna().sum() > 0:
                if cat_method == "mode":
                    mode_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Unknown"
                    df_clean[col] = df_clean[col].fillna(mode_value)
                elif cat_method == "unknown":
                    df_clean[col] = df_clean[col].fillna("Unknown")
                elif cat_method == "drop":
                    df_clean = df_clean.dropna(subset=[col])
                
                missing_info[f"{col}_missing_filled"] = df_clean[col].isna().sum()
        
        return df_clean, missing_info
    
    def handle_outliers(self, df: pd.DataFrame, method: str, numerical_cols: list, **kwargs) -> pd.DataFrame:
        """Handle outliers using different methods"""
        df_clean = df.copy()
        outlier_info = {}
        
        for col in numerical_cols:
            if col not in df_clean.columns:
                continue
                
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - kwargs.get('threshold', 1.5) * IQR
            upper_bound = Q3 + kwargs.get('threshold', 1.5) * IQR
            
            outliers_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
            outlier_count = outliers_mask.sum()
            outlier_info[f"{col}_outliers"] = outlier_count
            
            if method == "iqr":
                df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, 
                                    np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col]))
            elif method == "capping":
                # FIX: Use proper parameter names
                limits = kwargs.get('limits', [0.05, 0.95])
                lower_cap = df_clean[col].quantile(limits[0])
                upper_cap = df_clean[col].quantile(limits[1])
                df_clean[col] = np.clip(df_clean[col], lower_cap, upper_cap)
            elif method == "winsorize":
                limits = kwargs.get('limits', [0.05, 0.05])
                lower_limit = df_clean[col].quantile(limits[0])
                upper_limit = df_clean[col].quantile(1 - limits[1])
                df_clean[col] = np.clip(df_clean[col], lower_limit, upper_limit)
            # For "none" method, do nothing
        
        return df_clean, outlier_info
    
    def reduce_skewness(self, df: pd.DataFrame, method: str, numerical_cols: list) -> pd.DataFrame:
        """Reduce skewness in numerical columns"""
        df_clean = df.copy()
        skewness_info = {}
        
        for col in numerical_cols:
            if col not in df_clean.columns:
                continue
                
            original_skew = df_clean[col].skew()
            skewness_info[f"{col}_original_skew"] = original_skew
            
            if method == "log" and original_skew > 0:
                df_clean[col] = np.log1p(df_clean[col] - df_clean[col].min() + 1)
            elif method == "boxcox" and original_skew > 0:
                # Box-Cox requires positive data
                if df_clean[col].min() > 0:
                    pt = PowerTransformer(method='box-cox')
                    df_clean[col] = pt.fit_transform(df_clean[[col]]).ravel()
            elif method == "yeojohnson":
                pt = PowerTransformer(method='yeo-johnson')
                df_clean[col] = pt.fit_transform(df_clean[[col]]).ravel()
            
            final_skew = df_clean[col].skew()
            skewness_info[f"{col}_final_skew"] = final_skew
        
        return df_clean, skewness_info
    
    def encode_categorical(self, df: pd.DataFrame, method: str, categorical_cols: list) -> pd.DataFrame:
        """Encode categorical variables - RETURN ONLY NUMERIC METRICS"""
        df_clean = df.copy()
        encoding_info = {}
        
        for col in categorical_cols:
            if col not in df_clean.columns:
                continue
                
            if method == "onehot":
                dummies = pd.get_dummies(df_clean[col], prefix=col)
                df_clean = pd.concat([df_clean.drop(col, axis=1), dummies], axis=1)
                encoding_info[f"{col}_encoded_categories"] = len(dummies.columns)  # ✅ NUMERIC VALUE
            elif method == "label":
                le = LabelEncoder()
                df_clean[col] = le.fit_transform(df_clean[col])
                encoding_info[f"{col}_encoded_categories"] = len(le.classes_)  # ✅ NUMERIC VALUE
            elif method == "target":
                # Simple target encoding - FIXED: Remove inplace=True
                if self.target_column and self.target_column in df_clean.columns:
                    target_mean = df_clean.groupby(col)[self.target_column].mean()
                    df_clean[col] = df_clean[col].map(target_mean)
                    df_clean[col] = df_clean[col].fillna(target_mean.mean())  # ✅ FIXED
                encoding_info[f"{col}_encoded_categories"] = 1  # ✅ NUMERIC VALUE
        
        return df_clean, encoding_info
    
    def scale_features(self, df: pd.DataFrame, method: str, numerical_cols: list) -> pd.DataFrame:
        """Scale numerical features - RETURN ONLY NUMERIC METRICS"""
        df_clean = df.copy()
        scaling_info = {}
        
        scaler = None
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        elif method == "robust":
            scaler = RobustScaler()
        
        if scaler:
            scaled_count = 0
            for col in numerical_cols:
                if col in df_clean.columns:
                    df_clean[col] = scaler.fit_transform(df_clean[[col]]).ravel()
                    scaled_count += 1
            scaling_info["features_scaled"] = scaled_count  # ✅ NUMERIC VALUE
        
        return df_clean, scaling_info