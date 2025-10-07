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
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact

# Step 1: Data ingestion
data_ingestion = DataIngestion()
data_ingestion_artifact: DataIngestionArtifact = data_ingestion.initiate_data_ingestion()

# Step 2: Load data validation config
data_validation_config = DataValidationConfig()  # Ensure you pass any required params

# Step 3: Create DataValidation object with proper arguments
data_validation = DataValidation(
    data_ingestion_artifact=data_ingestion_artifact,
    data_validation_config=data_validation_config
)

# Step 4: Initiate validation
validation_artifact = data_validation.initiate_data_validation()
