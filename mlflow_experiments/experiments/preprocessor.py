import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import PowerTransformer, StandardScaler, MinMaxScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Clean preprocessing with all strategies"""
    
    def __init__(self):
        self.numerical_cols = None
        self.categorical_cols = None
        self.preprocessing_steps = []
        
    def identify_columns(self, df, target_col='heart_attack'):
        """Identify column types"""
        self.numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if target_col in self.numerical_cols:
            self.numerical_cols.remove(target_col)
            
        return self.numerical_cols, self.categorical_cols
    
    def handle_missing_values(self, df, numerical_strategy='median', categorical_strategy='mode'):
        """Handle missing values"""
        df_clean = df.copy()
        step_info = {'step': 'missing_values', 'numerical_strategy': numerical_strategy, 
                     'categorical_strategy': categorical_strategy}
        
        initial_rows = len(df_clean)
        
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
        if self.categorical_cols:
            if categorical_strategy == "drop":
                df_clean = df_clean.dropna(subset=self.categorical_cols)
            elif categorical_strategy == "mode":
                imputer = SimpleImputer(strategy='most_frequent')
                df_clean[self.categorical_cols] = imputer.fit_transform(df_clean[self.categorical_cols])
            elif categorical_strategy == "unknown":
                df_clean[self.categorical_cols] = df_clean[self.categorical_cols].fillna("Unknown")
        
        step_info['rows_after'] = len(df_clean)
        step_info['rows_removed'] = initial_rows - len(df_clean)
        self.preprocessing_steps.append(step_info)
        
        return df_clean
    
    def handle_outliers(self, df, method='cap', threshold=1.5):
        """Handle outliers using IQR method"""
        df_clean = df.copy()
        step_info = {'step': 'outliers', 'method': method, 'threshold': threshold}
        
        initial_rows = len(df_clean)
        outliers_handled = 0
        
        for col in self.numerical_cols:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                if method == "remove":
                    before = len(df_clean)
                    df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
                    outliers_handled += (before - len(df_clean))
                    
                elif method == "cap":
                    outliers_before = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
                    df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
                    outliers_handled += outliers_before
                    
                elif method == "winsorize":
                    lower_percentile = df_clean[col].quantile(0.05)
                    upper_percentile = df_clean[col].quantile(0.95)
                    outliers_before = ((df_clean[col] < lower_percentile) | (df_clean[col] > upper_percentile)).sum()
                    df_clean[col] = np.clip(df_clean[col], lower_percentile, upper_percentile)
                    outliers_handled += outliers_before
        
        step_info['rows_after'] = len(df_clean)
        step_info['outliers_handled'] = int(outliers_handled)
        self.preprocessing_steps.append(step_info)
        
        return df_clean
    
    def reduce_skewness(self, df, method='yeojohnson'):
        """Reduce skewness in numerical columns"""
        df_clean = df.copy()
        step_info = {'step': 'skewness', 'method': method}
        
        if method != "none":
            for col in self.numerical_cols:
                if col in df_clean.columns:
                    if method == "log" and df_clean[col].min() > 0:
                        df_clean[col] = np.log1p(df_clean[col])
                    elif method == "boxcox" and df_clean[col].min() > 0:
                        pt = PowerTransformer(method='box-cox')
                        df_clean[col] = pt.fit_transform(df_clean[[col]]).ravel()
                    elif method == "yeojohnson":
                        pt = PowerTransformer(method='yeo-johnson')
                        df_clean[col] = pt.fit_transform(df_clean[[col]]).ravel()
        
        self.preprocessing_steps.append(step_info)
        return df_clean
    
    def scale_features(self, df, method='standard'):
        """Scale numerical features"""
        df_clean = df.copy()
        step_info = {'step': 'scaling', 'method': method}
        
        if method != "none":
            if method == "standard":
                scaler = StandardScaler()
            elif method == "minmax":
                scaler = MinMaxScaler()
            elif method == "robust":
                scaler = RobustScaler()
            
            df_clean[self.numerical_cols] = scaler.fit_transform(df_clean[self.numerical_cols])
        
        self.preprocessing_steps.append(step_info)
        return df_clean
    
    def encode_categorical(self, df):
        """Encode categorical variables"""
        df_clean = df.copy()
        
        for col in self.categorical_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype('category').cat.codes
        
        self.preprocessing_steps.append({'step': 'encoding', 'method': 'label_encoding'})
        return df_clean
    
    def get_preprocessing_summary(self):
        """Get summary of all preprocessing steps"""
        return self.preprocessing_steps