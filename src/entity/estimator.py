import sys
import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline

from src.exception import MyException
from src.logger import logging
from src.utils.preprocessing_utils import PreprocessingUtils

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
        """
        Initialize MyModel with preprocessing pipeline and trained model
        
        Args:
            preprocessing_object: Sklearn Pipeline for scaling/transformations
            trained_model_object: Trained ML model (e.g., LightGBM)
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: pd.DataFrame) -> DataFrame:
        try:
            logging.info("Starting prediction process with raw data...")
            logging.info("Step 1: Applying complete feature engineering...")
            df_processed = PreprocessingUtils.apply_complete_feature_engineering(dataframe)
            logging.info("Step 2: Applying preprocessing transformations...")
            df_processed = PreprocessingUtils.apply_preprocessing_transformations(df_processed)
            logging.info("Step 3: Applying scaling transformations...")
            transformed_feature = self.preprocessing_object.transform(df_processed)
            logging.info("Step 4: Making predictions...")
            predictions = self.trained_model_object.predict(transformed_feature)
            
            logging.info(f"Prediction completed. Total predictions: {len(predictions)}")
            return predictions

        except Exception as e:
            logging.error("Error occurred in predict method", exc_info=True)
            raise MyException(e, sys) from e