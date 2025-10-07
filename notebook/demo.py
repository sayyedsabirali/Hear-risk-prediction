# # Access logger code
# from src.logger import logger
# logger.info("hi")

# Access Exceptioncode
# from src.exception import MyException
# from src.logger import logger
# import sys 
# try:
#     a=1/0
# except Exception as e:
#     logger.info(e)
#     raise MyException(e,sys)



# Check the data transformation code 
from src.pipline.training_pipeline import TrainPipeline

if __name__ == "__main__":
    pipeline = TrainPipeline()

    # 1️⃣ Data Ingestion
    data_ingestion_artifact = pipeline.start_data_ingestion()

    # 2️⃣ Data Validation
    data_validation_artifact = pipeline.start_data_validation(
        data_ingestion_artifact=data_ingestion_artifact
    )

    # 3️⃣ Data Transformation
    data_transformation_artifact = pipeline.start_data_transformation(
        data_ingestion_artifact=data_ingestion_artifact,
        data_validation_artifact=data_validation_artifact
    )

    # 4️⃣ Model Trainer
    model_trainer_artifact = pipeline.start_model_trainer(
        data_transformation_artifact=data_transformation_artifact
    )
