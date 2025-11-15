import os
from datetime import date
from dotenv import load_dotenv
load_dotenv()

# For MongoDB connection
DATABASE_NAME = "heart"
COLLECTION_NAME = "heart-Data"
password =  os.getenv("MONGO_PASS")
user = os.getenv("MONGO_USER_NAME")
MONGODB_URL_KEY = f"mongodb+srv://{user}:{password}@heart.xfuzvm4.mongodb.net/?retryWrites=true&w=majority&appName=heart"

PIPELINE_NAME: str = ""
ARTIFACT_DIR: str = "artifact"
Path_of_data= "F:\\18. MAJOR PROJECT\\Heart-related-content\\heart_risk_complete_dataset.csv"

MODEL_FILE_NAME = "model.pkl"

TARGET_COLUMN = "heart_attack"

CURRENT_YEAR = date.today().year
PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"

FILE_NAME: str = "data.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")


AWS_ACCESS_KEY_ID_ENV_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY_ENV_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

REGION_NAME = "us-east-1"



# Data Ingestion related constant start with DATA_INGESTION VAR NAME

DATA_INGESTION_COLLECTION_NAME: str = "heart-Data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.25

"""
Data Validation realted contant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME: str = "report.yaml"

"""
Data Transformation related constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

"""
MODEL TRAINER related constant start with MODEL_TRAINER var name
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.2
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")

# CHANGED: LightGBM to XGBoost with tuned parameters
MODEL_TRAINER_MODEL_TYPE: str = "XGBoost"

# CHANGED: Updated to tuned XGBoost parameters from your best model
MODEL_TRAINER_N_ESTIMATORS: int = 400  # Changed from 2000 to 400
MODEL_TRAINER_MAX_DEPTH: int = 10      # Changed from 12 to 10
MODEL_TRAINER_LEARNING_RATE: float = 0.1  # Changed from 0.02 to 0.1
MODEL_TRAINER_SUBSAMPLE: float = 0.8   # Same as tuned
MODEL_TRAINER_COLSAMPLE_BYTREE: float = 0.7  # Changed from 0.8 to 0.7
MODEL_TRAINER_REG_ALPHA: float = 0.1   # Same as tuned (reg_alpha)
MODEL_TRAINER_REG_LAMBDA: float = 0.5  # Changed from 0.1 to 0.5 (reg_lambda)

# CHANGED: Added XGBoost specific parameters
MODEL_TRAINER_GAMMA: float = 0.5       # Added gamma parameter
MODEL_TRAINER_MIN_CHILD_WEIGHT: int = 5  # Added min_child_weight (changed from min_child_samples)

# REMOVED: LightGBM specific parameters
# MODEL_TRAINER_NUM_LEAVES: int = 64
# MODEL_TRAINER_MIN_CHILD_SAMPLES: int = 20

MODEL_TRAINER_RANDOM_STATE: int = 42
MODEL_TRAINER_VERBOSE = -1

# CHANGED: Removed feature engineering since you don't want it
# FEATURE_SELECTION_COUNT: int = 30

"""
MODEL Evaluation related constants
"""
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_BUCKET_NAME = "sabir-project"
MODEL_PUSHER_S3_KEY = "model-registry"


APP_HOST = "0.0.0.0"
APP_PORT = 5000