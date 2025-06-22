import os
import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.common_functions import read_yaml
from scipy.stats import randint

import mlflow
import mlflow.sklearn

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self,train_path,test_path,model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path

        self.params_dist = LIGHTGBM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS
    
    def load_and_split_data(self):
        try:
            logger.info("Loading and splitting data...")
            train_data = pd.read_csv(self.train_path)
            test_data = pd.read_csv(self.test_path)

            X_train = train_data.drop('booking_status',axis=1)
            y_train = train_data['booking_status']
            X_test = test_data.drop('booking_status',axis=1)
            y_test = test_data['booking_status']

            logger.info("Data loaded and split successfully.")
            return X_train, y_train, X_test, y_test
        
        except Exception as e:
            logger.error(f"Error occurred while loading and splitting data: {str(e)}")
            raise CustomException(f"Error in loading and splitting data: {e}" ,e)
    
    def train_lgbm(self, X_train, y_train):
        try:
            logger.info("Initiating LightGBM model training...")
            model = LGBMClassifier(random_state=self.random_search_params['random_state'])
            
            logger.info("Starting Randomized Search for hyperparameter tuning...")
            random_search = RandomizedSearchCV(
                model,
                param_distributions=self.params_dist,
                **self.random_search_params
            )
            logger.info("Fitting the model...")
            random_search.fit(X_train, y_train)
            logger.info("Model training completed successfully.")
            logger.info(f"Best parameters found: {random_search.best_params_}")

            return random_search.best_estimator_
        
        except Exception as e:  
            logger.error(f"Error occurred during model training: {str(e)}")
            raise CustomException(f"Error in model training: {e}" ,e)

    def evaluate_model(self, model, X_test, y_test):
        try:
            logger.info("Evaluating the model...")
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            logger.info(f"Model evaluation metrics: Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        
        except Exception as e:
            logger.error(f"Error occurred during model evaluation: {str(e)}")
            raise CustomException(f"Error in model evaluation: {e}" , e)
    
    def save_model(self, model):
        try:
            logger.info(f"Saving the trained model to {self.model_output_path}...")
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)

            if not self.model_output_path.endswith('.pkl'):
                self.model_output_path += '.pkl'
            
            joblib.dump(model, self.model_output_path)
            logger.info("Model saved successfully.")
        
        except Exception as e:
            logger.error(f"Error occurred while saving the model: {str(e)}")
            raise CustomException(f"Error in saving model: {e}" , e)

    def run(self) -> dict:
        try:
            with mlflow.start_run():
                logger.info("Starting model training pipeline...")
                
                logger.info("MLflow run started for model training.")
                logger.info("Logging the training and testing datasets to MLflow...")

                mlflow.log_artifact(self.train_path)

                logger.info("Logging the test dataset to MLflow...")

                mlflow.log_artifact(self.test_path)

                X_train, y_train, X_test, y_test = self.load_and_split_data()
                model = self.train_lgbm(X_train, y_train)
                evaluation_metrics = self.evaluate_model(model, X_test, y_test)
                self.save_model(model)

                logger.info("Model training and evaluation completed successfully.")
                logger.info("Logging the model to MLflow...")
                

                mlflow.log_artifact(self.model_output_path)
                mlflow.log_params(model.get_params())
                mlflow.log_metrics(evaluation_metrics)

                logger.info("Model training pipeline completed successfully.")
                return evaluation_metrics
        
        except Exception as e:
            logger.error(f"Error in model training pipeline: {str(e)}")
            raise CustomException(f"Model training pipeline failed: {e}", e)
if __name__ == "__main__":
    model_trainer = ModelTraining(
        train_path=PROCESSED_TRAIN_DATA_PATH,
        test_path=PROCESSED_TEST_DATA_PATH,
        model_output_path=MODEL_OUTPUT_PATH
    )
    evaluation_results = model_trainer.run()
    logger.info(f"Model evaluation results: {evaluation_results}")