import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class DataConfig:
    SAMPLE_SIZE: int = 1000
    DATA_PATH: str = r"F:\18. MAJOR PROJECT\Data\DATA_FOR_HEART_RISK_PRED.csv"
    TARGET_COLUMN: str = "heart_flag"
    NUMERICAL_COLUMNS: List[str] = None
    CATEGORICAL_COLUMNS: List[str] = None
    
    def __post_init__(self):
        if self.NUMERICAL_COLUMNS is None:
            self.NUMERICAL_COLUMNS = []
        if self.CATEGORICAL_COLUMNS is None:
            self.CATEGORICAL_COLUMNS = []