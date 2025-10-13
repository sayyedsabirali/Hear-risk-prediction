import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import PowerTransformer, StandardScaler, MinMaxScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

class BoxPlotPreprocessor:
    def __init__(self):
        self.numerical_cols = None
        self.categorical_cols = None
        
    def identify_columns(self, df):
        """
        Identify numerical and categorical columns
        """
        self.numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target from numerical if present
        if 'heart_attack' in self.numerical_cols:
            self.numerical_cols.remove('heart_attack')
            
        return self.numerical_cols, self.categorical_cols
    
    def handle_missing_values(self, df, numerical_strategy, categorical_strategy):
        """Handle missing values with different strategies"""
        df_clean = df.copy()
        missing_info = {}
        
        # Numerical columns
        if numerical_strategy == "drop":
            df_clean = df_clean.dropna(subset=self.numerical_cols)
        elif numerical_strategy == "mean":
            imputer = SimpleImputer(strategy='mean')
            df_clean[self.numerical_cols] = imputer.fit_transform(df_clean[self.numerical_cols])
        elif numerical_strategy == "median":
            imputer = SimpleImputer(strategy='median')
            df_clean[self.numerical_cols] = imputer.fit_transform(df_clean[self.numerical_cols])
        elif numerical_strategy == "knn":
            imputer = KNNImputer(n_neighbors=5)
            df_clean[self.numerical_cols] = imputer.fit_transform(df_clean[self.numerical_cols])
        
        # Categorical columns
        if categorical_strategy == "drop":
            df_clean = df_clean.dropna(subset=self.categorical_cols)
        elif categorical_strategy == "mode":
            imputer = SimpleImputer(strategy='most_frequent')
            df_clean[self.categorical_cols] = imputer.fit_transform(df_clean[self.categorical_cols])
        elif categorical_strategy == "unknown":
            df_clean[self.categorical_cols] = df_clean[self.categorical_cols].fillna("Unknown")
        
        missing_info['remaining_rows'] = len(df_clean)
        missing_info['rows_dropped'] = len(df) - len(df_clean)
        missing_info['data_retention_pct'] = (len(df_clean) / len(df)) * 100
        
        return df_clean, missing_info
    
    def handle_outliers_boxplot(self, df, method, threshold=1.5):
        """Handle outliers using PURE BOX PLOT method (IQR)"""
        df_clean = df.copy()
        outlier_info = {}
        
        total_outliers_removed = 0
        
        for col in self.numerical_cols:
            if col in df_clean.columns:
                # Box plot calculations
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                if method == "remove":
                    # Remove outliers completely
                    before = len(df_clean)
                    df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
                    outliers_removed = before - len(df_clean)
                    outlier_info[f'{col}_outliers_removed'] = outliers_removed
                    total_outliers_removed += outliers_removed
                    
                elif method == "cap":
                    # Cap outliers at bounds
                    outliers_before = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
                    df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
                    outlier_info[f'{col}_outliers_capped'] = outliers_before
                    total_outliers_removed += outliers_before
                    
                elif method == "winsorize":
                    # Winsorize - cap at 5th and 95th percentiles
                    lower_percentile = df_clean[col].quantile(0.05)
                    upper_percentile = df_clean[col].quantile(0.95)
                    outliers_before = ((df_clean[col] < lower_percentile) | (df_clean[col] > upper_percentile)).sum()
                    df_clean[col] = np.clip(df_clean[col], lower_percentile, upper_percentile)
                    outlier_info[f'{col}_outliers_winsorized'] = outliers_before
                    total_outliers_removed += outliers_before
        
        outlier_info['final_rows_after_outliers'] = len(df_clean)
        outlier_info['total_outliers_handled'] = total_outliers_removed
        outlier_info['rows_lost_pct'] = ((len(df) - len(df_clean)) / len(df)) * 100
        
        return df_clean, outlier_info
    
    def reduce_skewness(self, df, method):
        """Reduce skewness in numerical columns"""
        df_clean = df.copy()
        skew_info = {}
        
        if method != "none":
            for col in self.numerical_cols:
                if col in df_clean.columns:
                    original_skew = df_clean[col].skew()
                    skew_info[f'{col}_original_skew'] = original_skew
                    
                    if method == "log" and df_clean[col].min() > 0:
                        df_clean[col] = np.log1p(df_clean[col])
                    elif method == "boxcox" and df_clean[col].min() > 0:
                        pt = PowerTransformer(method='box-cox')
                        df_clean[col] = pt.fit_transform(df_clean[[col]]).ravel()
                    elif method == "yeojohnson":
                        pt = PowerTransformer(method='yeo-johnson')
                        df_clean[col] = pt.fit_transform(df_clean[[col]]).ravel()
                    
                    final_skew = df_clean[col].skew()
                    skew_info[f'{col}_final_skew'] = final_skew
                    skew_info[f'{col}_skewness_reduced'] = original_skew - final_skew
        
        return df_clean, skew_info
    
    def scale_features(self, df, method):
        """Scale numerical features"""
        df_clean = df.copy()
        scaling_info = {}
        
        if method != "none":
            if method == "standard":
                scaler = StandardScaler()
            elif method == "minmax":
                scaler = MinMaxScaler()
            elif method == "robust":
                scaler = RobustScaler()
            
            df_clean[self.numerical_cols] = scaler.fit_transform(df_clean[self.numerical_cols])
            # Don't log string values as metrics
            # scaling_info['scaling_method'] = method  # This causes the error
        
        # Instead, log numeric indicators
        scaling_info['scaling_applied'] = 1.0 if method != "none" else 0.0
        
        return df_clean, scaling_info
    
    def encode_categorical(self, df):
        """Encode categorical variables"""
        df_clean = df.copy()
        
        for col in self.categorical_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype('category').cat.codes
        
        return df_clean