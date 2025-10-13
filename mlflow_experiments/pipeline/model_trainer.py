import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

class ModelTrainer:
    def __init__(self):
        self.models = {
            'xgboost': xgb.XGBClassifier,
            'lightgbm': lgb.LGBMClassifier,
            'catboost': CatBoostClassifier,
            # 'random_forest': RandomForestClassifier, 
            'gradient_boosting': GradientBoostingClassifier,
            'ada_boost': AdaBoostClassifier,
            'extra_trees': ExtraTreesClassifier,
            'svm': SVC,
            'knn': KNeighborsClassifier,
            'decision_tree': DecisionTreeClassifier,
            'naive_bayes': GaussianNB
        }
    
    def train_and_evaluate(self, df, model_name, model_params, target_col='heart_attack'):
        """Train and evaluate a model - return only key metrics"""
        try:
            # Prepare features and target
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Initialize and train model
            model_class = self.models[model_name]
            model = model_class(**model_params)
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Calculate key metrics - BINARY CLASSIFICATION (use binary averaging)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='binary')  # Changed to binary
            recall = recall_score(y_test, y_pred, average='binary')        # Changed to binary  
            f1 = f1_score(y_test, y_pred, average='binary')               # Changed to binary
            
            # Cross-validation for accuracy only
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            
            results = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean_accuracy': cv_scores.mean(),
                'cv_std_accuracy': cv_scores.std(),
                'model': model,
                'final_rows': len(df)
            }
            
            return results
            
        except Exception as e:
            print(f"❌ Error training {model_name}: {str(e)}")
            return None