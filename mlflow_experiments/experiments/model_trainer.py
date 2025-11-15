import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """Train and evaluate multiple models"""
    
    def __init__(self):
        self.models = {
            'xgboost': xgb.XGBClassifier,
            'lightgbm': lgb.LGBMClassifier,
            'catboost': CatBoostClassifier,
            'random_forest': RandomForestClassifier,
            'logistic_regression': LogisticRegression,
            'gradient_boosting': GradientBoostingClassifier,
            'ada_boost': AdaBoostClassifier,
            'extra_trees': ExtraTreesClassifier,
            'svm': SVC,
            'knn': KNeighborsClassifier,
            'decision_tree': DecisionTreeClassifier,
            'naive_bayes': GaussianNB
        }
    
    def train_and_evaluate(self, df, model_name, model_params, target_col='heart_attack'):
        """Train model and return comprehensive metrics"""
        try:
            # Prepare data
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
            
            # Get probability predictions for AUC (if available)
            try:
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    auc_score = roc_auc_score(y_test, y_pred_proba)
                else:
                    y_pred_proba = y_pred  # Fallback
                    auc_score = 0.0
            except:
                y_pred_proba = y_pred
                auc_score = 0.0
            
            # Calculate all metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Cross-validation scores (5-fold)
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            
            # Feature importance (if available)
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False).head(10)
            
            results = {
                'model': model,
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'auc': float(auc_score),
                'confusion_matrix': cm,
                'y_true': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'feature_importance': feature_importance,
                'train_size': len(X_train),
                'test_size': len(X_test)
            }
            
            return results
            
        except Exception as e:
            print(f"❌ Error training {model_name}: {str(e)}")
            return None