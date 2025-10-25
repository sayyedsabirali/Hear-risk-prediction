import sys
import pandas as pd
from src.entity.config_entity import ModelPusherConfig
from src.entity.s3_estimator import Proj1Estimator
from src.exception import MyException
from src.logger import logging
from src.utils.preprocessing_utils import PreprocessingUtils  


class HeartPatientData:
    def __init__(self,
                 gender: str,
                 anchor_age: int,
                 admission_type: str,
                 insurance: str,
                 race: str,
                 marital_status: str,
                 creatinine: float,
                 glucose: float,
                 sodium: float,
                 potassium: float,
                 troponin_t: float,
                 creatine_kinase_mb: float,
                 hemoglobin: float,
                 white_blood_cells: float,
                 heart_rate: float,
                 bp_systolic: float,
                 bp_diastolic: float,
                 spo2: float,
                 respiratory_rate: float,
                 temperature: float):
        try:
            self.gender = gender
            self.anchor_age = anchor_age
            self.admission_type = admission_type
            self.insurance = insurance
            self.race = race
            self.marital_status = marital_status
            self.creatinine = creatinine
            self.glucose = glucose
            self.sodium = sodium
            self.potassium = potassium
            self.troponin_t = troponin_t
            self.creatine_kinase_mb = creatine_kinase_mb
            self.hemoglobin = hemoglobin
            self.white_blood_cells = white_blood_cells
            self.heart_rate = heart_rate
            self.bp_systolic = bp_systolic
            self.bp_diastolic = bp_diastolic
            self.spo2 = spo2
            self.respiratory_rate = respiratory_rate
            self.temperature = temperature

        except Exception as e:
            raise MyException(e, sys) from e

    def get_heart_data_as_dict(self):
        logging.info("Entered get_heart_data_as_dict method as HeartPatientData class")
        try:
            input_data = {
                "gender": [self.gender],
                "anchor_age": [self.anchor_age],
                "admission_type": [self.admission_type],
                "insurance": [self.insurance],
                "race": [self.race],
                "marital_status": [self.marital_status],
                "creatinine": [self.creatinine],
                "glucose": [self.glucose],
                "sodium": [self.sodium],
                "potassium": [self.potassium],
                "troponin_t": [self.troponin_t],
                "creatine_kinase_mb": [self.creatine_kinase_mb],
                "hemoglobin": [self.hemoglobin],
                "white_blood_cells": [self.white_blood_cells],
                "heart_rate": [self.heart_rate],
                "bp_systolic": [self.bp_systolic],
                "bp_diastolic": [self.bp_diastolic],
                "spo2": [self.spo2],
                "respiratory_rate": [self.respiratory_rate],
                "temperature": [self.temperature]
            }

            logging.info("Created heart patient data dict")
            logging.info("Exited get_heart_data_as_dict method as HeartPatientData class")
            return input_data

        except Exception as e:
            raise MyException(e, sys) from e

    def get_heart_input_data_frame(self) -> pd.DataFrame:
        try:
            heart_input_dict = self.get_heart_data_as_dict()
            return pd.DataFrame(heart_input_dict)
        except Exception as e:
            raise MyException(e, sys) from e


class HeartRiskClassifier:
    def __init__(self, prediction_pipeline_config: ModelPusherConfig = ModelPusherConfig()):
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise MyException(e, sys)

    def predict(self, dataframe: pd.DataFrame) -> str:
        try:
            logging.info("Entered predict method of HeartRiskClassifier class")
            logging.info("Applying medical feature engineering...")
            processed_data = PreprocessingUtils.apply_complete_feature_engineering(dataframe)
            logging.info("Applying preprocessing transformations...")
            processed_data = PreprocessingUtils.apply_preprocessing_transformations(processed_data)
            
            # Load model from S3
            model = Proj1Estimator(
                bucket_name=self.prediction_pipeline_config.bucket_name,
                model_path=self.prediction_pipeline_config.s3_model_key_path,
            )
            
            # Make prediction
            result = model.predict(processed_data)
            
            # Convert prediction to meaningful result
            prediction_label = "Patient has heart risk" if result[0] == 1 else "Patient has no heart risk"
            
            logging.info(f"Prediction completed: {prediction_label}")
            return prediction_label
        
        except Exception as e:
            raise MyException(e, sys)