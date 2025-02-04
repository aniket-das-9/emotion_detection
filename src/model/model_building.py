import numpy as np
import pandas as pd

import pickle

from sklearn.ensemble import GradientBoostingClassifier

import yaml
import logging
from typing import Tuple, Dict, Any

# params = yaml.safe_load(open('params.yaml','r'))['model_building']
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_building.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> Dict[str, Any]:
    # """Load model hyperparameters from the YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)['model_building']
        logger.info("Successfully loaded model hyperparameters.")
        return params
    except FileNotFoundError:
        logger.error(f"Parameters file {params_path} not found.", exc_info=True)
        raise
    except KeyError:
        logger.error("Missing key in parameters file.", exc_info=True)
        raise
    except yaml.YAMLError:
        logger.error("Error while parsing YAML file.", exc_info=True)
        raise
    except Exception as e:
        logger.critical("Unexpected error occurred while loading parameters.", exc_info=True)
        raise

# Fetch the data from data/features

# train_data = pd.read_csv('./data/features/train_bow.csv')

# X_train = train_data.iloc[:,0:-1].values
# y_train = train_data.iloc[:,-1].values
def load_data(train_path: str) -> Tuple[np.ndarray, np.ndarray]:
    # """Load training data and split it into features (X_train) and labels (y_train)."""
    try:
        train_data = pd.read_csv(train_path)
        X_train = train_data.iloc[:, 0:-1].values
        y_train = train_data.iloc[:, -1].values
        logger.info("Successfully loaded training data.")
        return X_train, y_train
    except FileNotFoundError:
        logger.error("Training data file not found.", exc_info=True)
        raise
    except pd.errors.ParserError:
        logger.error("Error parsing the training CSV file.", exc_info=True)
        raise
    except Exception as e:
        logger.critical("Unexpected error occurred while loading data.", exc_info=True)
        raise

# Define and train the GradientBoosting model
# clf = GradientBoostingClassifier(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'])
# clf.fit(X_train, y_train)
def train_model(X_train: np.ndarray, y_train: np.ndarray, params: Dict[str, Any]) -> GradientBoostingClassifier:
    """Train a GradientBoostingClassifier model."""
    try:
        clf = GradientBoostingClassifier(
            n_estimators=params['n_estimators'], 
            learning_rate=params['learning_rate']
        )
        clf.fit(X_train, y_train)
        logger.info("Model training completed successfully.")
        return clf
    except KeyError:
        logger.error("Missing hyperparameters in parameters dictionary.", exc_info=True)
        raise
    except ValueError as e:
        logger.error("Invalid input data for model training.", exc_info=True)
        raise
    except Exception as e:
        logger.critical("Unexpected error occurred during model training.", exc_info=True)
        raise

# save the model

# pickle.dump(clf,open('model.pkl','wb'))
def save_model(model: GradientBoostingClassifier, model_path: str) -> None:
    # """Save the trained model as a pickle file."""
    try:
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
        logger.info(f"Model successfully saved to {model_path}.")
    except Exception as e:
        logger.critical("Error occurred while saving the model.", exc_info=True)
        raise

def main() -> None:
    try:
        logger.info("Starting model building process.")

        # Load parameters
        params = load_params('params.yaml')

        # Load training data
        X_train, y_train = load_data('./data/processed/train_tfidf.csv')

        # Train model
        clf = train_model(X_train, y_train, params)

        # Save trained model
        save_model(clf, 'models/model.pkl')

        logger.info("Model building stage completed successfully.")
    except Exception as e:
        logger.critical("Model building process failed.", exc_info=True)
        raise

if __name__ == "__main__":
    main()