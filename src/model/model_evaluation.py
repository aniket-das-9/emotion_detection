import numpy as np
import pandas as pd

import pickle
import json

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_evaluation.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# clf = pickle.load(open('model.pkl','rb'))
def load_model(model_path: str) -> Any:
    # """Load the trained model from a pickle file."""
    try:
        with open(model_path, 'rb') as file:
            clf = pickle.load(file)
        logger.info("Model loaded successfully.")
        return clf
    except FileNotFoundError:
        logger.error(f"Model file {model_path} not found.", exc_info=True)
        raise
    except pickle.UnpicklingError:
        logger.error("Error unpickling the model file.", exc_info=True)
        raise
    except Exception as e:
        logger.critical("Unexpected error occurred while loading the model.", exc_info=True)
        raise

# Fetch the data from data/features

# test_data = pd.read_csv('./data/features/test_bow.csv')

# X_test = test_data.iloc[:,0:-1].values
# y_test = test_data.iloc[:,-1].values
def load_data(test_path: str) -> Tuple[np.ndarray, np.ndarray]:
    # """Load testing data and split it into features (X_test) and labels (y_test)."""
    try:
        test_data = pd.read_csv(test_path)
        X_test = test_data.iloc[:, 0:-1].values
        y_test = test_data.iloc[:, -1].values
        logger.info("Successfully loaded testing data.")
        return X_test, y_test
    except FileNotFoundError:
        logger.error(f"Testing data file {test_path} not found.", exc_info=True)
        raise
    except pd.errors.ParserError:
        logger.error("Error parsing the testing CSV file.", exc_info=True)
        raise
    except Exception as e:
        logger.critical("Unexpected error occurred while loading data.", exc_info=True)
        raise

# Make predictions
# y_pred = clf.predict(X_test)
# y_pred_proba = clf.predict_proba(X_test)[:, 1]

# # Calculate evaluation metrics
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# auc = roc_auc_score(y_test, y_pred_proba)

# metrics_dict = {
#     'accuracy':accuracy,
#     'precision':precision,
#     'recall':recall,
#     'auc':auc
# }
def evaluate_model(clf: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Evaluate the model on test data and return the calculated metrics."""
    try:
        # Make predictions
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }

        logger.info("Model evaluation completed successfully.")
        return metrics_dict
    except ValueError:
        logger.error("Invalid input data for model evaluation.", exc_info=True)
        raise
    except Exception as e:
        logger.critical("Unexpected error occurred during model evaluation.", exc_info=True)
        raise


# with open('metrics.json','w') as file:
#     json.dump(metrics_dict,file,indent=4)
def save_metrics(metrics: Dict[str, float], metrics_path: str) -> None:
    """Save evaluation metrics to a JSON file."""
    try:
        with open(metrics_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.info(f"Metrics successfully saved to {metrics_path}.")
    except Exception as e:
        logger.critical("Error occurred while saving evaluation metrics.", exc_info=True)
        raise

def main() -> None:
    try:
        logger.info("Starting model evaluation process.")

        # Load the trained model
        clf = load_model('./models/model.pkl')

        # Load test data
        X_test, y_test = load_data('./data/processed/test_tfidf.csv')

        # Evaluate the model
        metrics_dict = evaluate_model(clf, X_test, y_test)

        # Save evaluation metrics
        save_metrics(metrics_dict, 'reports/metrics.json')

        logger.info("Model evaluation stage completed successfully.")
    except Exception as e:
        logger.critical("Model evaluation process failed.", exc_info=True)
        raise

if __name__ == "__main__":
    main()