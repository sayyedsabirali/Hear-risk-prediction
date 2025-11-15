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
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: pd.DataFrame) -> DataFrame:
        try:
            logging.info("Starting prediction process with raw data...")
            logging.info("Step 1: Applying preprocessing transformations .....")
            df_processed = PreprocessingUtils.apply_preprocessing_transformations(dataframe)
            
            logging.info("Step 2: Applying identity transformation (no scaling)...")
            transformed_feature = self.preprocessing_object.transform(df_processed)
            
            # Step 3: Make predictions
            logging.info("Step 3: Making predictions...")
            predictions = self.trained_model_object.predict(transformed_feature)
            
            logging.info(f"Prediction completed. Total predictions: {len(predictions)}")
            return predictions

        except Exception as e:
            logging.error("Error occurred in predict method", exc_info=True)
            raise MyException(e, sys) from e