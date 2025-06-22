import os
import pandas as pd
from google.cloud import storage
from src.logger import get_logger
from config.paths_config import *
from src.custom_exception import CustomException
from utils.common_functions import read_yaml
from sklearn.model_selection import train_test_split

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self,config):
        self.config = config["data_ingestion"]
        self.bucket_name = self.config['bucket_name']
        self.file_name = self.config["bucket_file_name"]
        self.train_test_ratio = self.config["train_ratio"]

        os.makedirs(RAW_DIR, exist_ok=True)
        
        logger.info(f"DataIngestion started with {self.bucket_name} and file name is {self.file_name}")

    def download_csv_from_gcp(self):
        """
        Downloads a CSV file from Google Cloud Storage to the local raw directory.
        """
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.file_name)

            blob.download_to_filename(RAW_FILE_PATH)

            logger.info(f"Raw file is successfully downloaded from GCP and saved at {RAW_FILE_PATH}.")

        except Exception as e:
            logger.error(f"Error downloading file from GCP: {e}")
            raise CustomException("Failed to download file from GCP", e)
            
    def split_data(self):
        """
        Splits the raw data into training and testing datasets.
        """
        try:
            logger.info(f"Starting the Spliting process of the data.")
            data = pd.read_csv(RAW_FILE_PATH)
            logger.info(f"Raw data loaded successfully from {RAW_FILE_PATH}.")

            train_data, test_data = train_test_split(data, test_size=1-self.train_test_ratio, random_state=42)
            train_data.to_csv(TRAIN_FILE_PATH, index=False)
            test_data.to_csv(TEST_FILE_PATH, index=False)

            logger.info(f"Data split into train and test sets. Train data saved at {TRAIN_FILE_PATH}, Test data saved at {TEST_FILE_PATH}.")

        except Exception as e:
            logger.error(f"Error during data splitting: {e}")
            raise CustomException("Failed to split data", e)
        
    def run(self):
        """
        Executes the data ingestion process.
        """
        try:
            logger.info("Starting data ingestion process.")
            self.download_csv_from_gcp()
            self.split_data()
            logger.info("Data ingestion process completed successfully.")

        except Exception as e:
            logger.error(f"Error in data ingestion process: {e}")
            raise CustomException("Data ingestion process failed", e)
        
        finally:
            logger.info("Data ingestion process finished.")

if __name__ == "__main__":
    config = read_yaml(CONFIG_PATH)
    data_ingestion = DataIngestion(config)
    data_ingestion.run()