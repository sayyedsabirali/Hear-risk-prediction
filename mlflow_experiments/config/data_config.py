import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Any
from src.constants import *

import os
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = Path_of_data
TARGET_COL = TARGET_COLUMN
TEST_SIZE = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
RANDOM_STATE = 42
