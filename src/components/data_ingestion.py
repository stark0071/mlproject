import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformerConfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

# --- Configuration Class ---
@dataclass
class DataIngestionConfig:
    """Defines the paths where the ingestion process will save data artifacts."""
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Replace 'notebook/data/stud.csv' with the actual path to your source dataset
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as dataframe')

            # Create the 'artifacts' directory if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the train and test sets to the artifacts folder
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

# --- Main Entry Point ---
if __name__ == "__main__":
    # 1. Start Data Ingestion
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # 2. Trigger Data Transformation
    # This is where the code you showed earlier gets executed
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # 3. Trigger Model Training
    model_trainer = ModelTrainer()
    score = model_trainer.initiate_model_trainer(train_arr, test_arr)
    print("R2 Score for the best model is: ", score)

    logging.info("Data Ingestion and Model Training process completed successfully.")