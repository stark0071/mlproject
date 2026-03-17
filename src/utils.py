import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV # Changed from GridSearchCV

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        model_report = {}

        for model_name, model in models.items():
            param_grid = param.get(model_name, {})

            logging.info(f"Started tuning for: {model_name}")

            # Optimization 1: Use RandomizedSearchCV instead of GridSearchCV
            # Optimization 2: n_jobs=-1 uses all CPU cores
            # Optimization 3: n_iter=10 limits the search to 10 random combinations per model
            rs = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=5, # You can increase this for better accuracy or decrease for speed
                cv=3,
                n_jobs=-1, 
                verbose=0,
                random_state=42
            )

            rs.fit(X_train, y_train)

            # Optimization 4: Use the best estimator directly (no need to refit manually)
            best_model = rs.best_estimator_
            
            # Predict
            y_test_pred = best_model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)

            model_report[model_name] = test_model_score
            logging.info(f"Model: {model_name} | R2 Score: {test_model_score}")

        return model_report

    except Exception as e:
        raise CustomException(e, sys)