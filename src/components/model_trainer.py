import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Optimization 1: n_jobs=-1 uses all available CPU cores
            models = {
                "Random Forest": RandomForestRegressor(n_jobs=-1), 
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(n_jobs=-1),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False, thread_count=-1),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            # Optimization 2: Reduced parameter grid (Focused on high-impact keys)
            params = {
                "Random Forest": {
                    'n_estimators': [64, 128, 256],
                    'max_depth': [10, 20, None],
                    'max_features': ['sqrt', 'log2']
                    # Removed 'absolute_error' as it is computationally expensive
                },
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse'],
                    'max_depth': [5, 10, 15]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01],
                    'n_estimators': [100, 200],
                    'subsample': [0.8, 1.0]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01],
                    'n_estimators': [100, 200],
                    'max_depth': [5, 6]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [100, 200]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01],
                    'n_estimators': [50, 100]
                }
            }

            logging.info("Starting model evaluation and hyperparameter tuning")
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                models=models, param=params
            )

            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with acceptable R2 score")

            logging.info(f"Best Model: {best_model_name} with R2: {best_model_score}")

            # Fit the best model on full training data before saving
            best_model.fit(X_train, y_train)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)