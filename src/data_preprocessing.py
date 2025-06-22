import os
import pandas as pd
from utils.common_functions import read_yaml, load_data
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

logger = get_logger(__name__)

class DataProcessor:

    def __init__(self,train_path,test_path,processed_dir,config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir

        self.config = read_yaml(config_path)

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
            logger.info(f"Created directory: {self.processed_dir}")

    def preprocess_data(self,df):

        try:
            logger.info("Starting data preprocessing...")

            logger.info("Droping id columns")
            df.drop(columns=["Booking_ID"], inplace=True)
            df.drop_duplicates(inplace=True)

            cat_cols = self.config['data_processing']['categorical_columns']
            num_cols = self.config['data_processing']['numerical_columns']

            label_encoder = LabelEncoder()
            mappings = {}

            for col in cat_cols:
                df[col] = label_encoder.fit_transform(df[col])
                mappings[col] = {label:code for code, label in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}

            logger.info("Categorical columns encoded.")
            logger.info(f"Mappings are :")

            for col, mapping in mappings.items():
                logger.info(f"{col}: {mapping}")

            logger.info("Doing Skewness handling")
            
            skewness_threshold = self.config['data_processing']['skewness_threshold']
            for col in num_cols:
                skewness = df[col].skew()
                if skewness > skewness_threshold:
                    df[col] = np.log1p(df[col])
                    logger.info(f"Applied log transformation to {col} due to skewness: {skewness}")
            
            logger.info("Skewness handling completed.")
            return df
        
        except Exception as e:
            logger.error(f"Error during data preprocessing: {e}")
            raise CustomException("Data preprocessing failed", e)
    
    def handle_imbalance(self, df):
        try:
            logger.info("Handling class imbalance using SMOTE...")
            X = df.drop(columns=["booking_status"])
            y = df["booking_status"]

            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)

            df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            df_resampled["booking_status"] = y_resampled

            logger.info("Class imbalance handled successfully.")
            return df_resampled
        
        except Exception as e:
            logger.error(f"Error during class imbalance handling: {e}")
            raise CustomException("Class imbalance handling failed", e)
    
    def feature_selection(self,df):
        try:
            logger.info("Starting feature selection...")
    
            X = df.drop(columns=["booking_status"])
            y = df["booking_status"]
            rf_model = RandomForestClassifier(random_state=42)
            rf_model.fit(X, y)
            feature_importances = rf_model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': feature_importances
            })
            feature_importance_df.sort_values(by='importance', ascending=False, inplace=True)

            top_features = feature_importance_df['feature'].head(self.config["data_processing"]["number_of_features"]).values
            logger.info(f"Top features selected: {top_features}")

            top_df = df[top_features.tolist()+ ['booking_status']]

            return top_df
        
        except Exception as e:
            logger.error(f"Error during feature selection: {e}")
            raise CustomException("Feature selection failed", e)
    
    def save_data(self, df, file_path):
        try:
            logger.info(f"Saving processed data to {file_path}...")
            df.to_csv(file_path, index=False)
            logger.info(f"Data saved successfully to {file_path}.")
        except Exception as e:
            logger.error(f"Error saving data to {file_path}: {e}")
            raise CustomException("Failed to save processed data", e)
    
    def process(self):
        try:
            logger.info("Starting data processing pipeline...")

            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)

            train_df = self.handle_imbalance(train_df)
            test_df = self.handle_imbalance(test_df)

            train_df = self.feature_selection(train_df)
            test_df = test_df[train_df.columns]

            self.save_data(train_df, PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df, PROCESSED_TEST_DATA_PATH)

            logger.info("Data processing pipeline completed successfully.")
        
        except Exception as e:
            logger.error(f"Error in data processing pipeline: {e}")
            raise CustomException("Data processing pipeline failed", e)
        
if __name__ == "__main__":
    processor = DataProcessor(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH)
    processor.process()