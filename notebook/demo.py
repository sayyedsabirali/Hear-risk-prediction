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
data = DataIngestion()
data.initiate_data_ingestion()

