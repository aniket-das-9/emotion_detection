import numpy as np
import pandas as pd

import os

from sklearn.feature_extraction.text import TfidfVectorizer

import yaml
import logging
from typing import Tuple

logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('feature_engineering.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# max_features = yaml.safe_load(open('params.yaml','r'))['feature_engineering']['max_features']

def load_params(params_path: str) -> int:
    # Load max_features parameter from the YAML file.
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        max_features = params['feature_engineering']['max_features']
        logger.info("Successfully loaded max_features parameter.")
        return max_features
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

# Fetch the data from data/processed

# train_data = pd.read_csv('./data/processed/train_processed.csv')
# test_data = pd.read_csv('./data/processed/test_processed.csv')

# train_data.fillna('',inplace=True)
# test_data.fillna('',inplace=True)

def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # """Load preprocessed training and testing data."""
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        train_data.fillna('', inplace=True)
        test_data.fillna('', inplace=True)
        logger.info("Successfully loaded processed training and testing data.")
        return train_data, test_data
    except FileNotFoundError:
        logger.error("Processed data files not found.", exc_info=True)
        raise
    except pd.errors.ParserError:
        logger.error("Error parsing the CSV files.", exc_info=True)
        raise
    except Exception as e:
        logger.critical("Unexpected error occurred while loading data.", exc_info=True)
        raise

# Apply BOW

# X_train = train_data['content'].values
# y_train = train_data['sentiment'].values

# X_test = test_data['content'].values
# y_test = test_data['sentiment'].values

# # Apply Bag of Words (CountVectorizer)
# vectorizer = CountVectorizer(max_features=max_features)

# # Fit the vectorizer on the training data and transform it
# X_train_bow = vectorizer.fit_transform(X_train)

# # Transform the test data using the same vectorizer
# X_test_bow = vectorizer.transform(X_test)

# train_df = pd.DataFrame(X_train_bow.toarray())

# train_df['label'] = y_train

# test_df = pd.DataFrame(X_test_bow.toarray())

# test_df['label'] = y_test

def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # """Apply Tfidf to the data"""
    try:
        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values

        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values

        vectorizer = TfidfVectorizer(max_features=max_features)

        # Fit and transform training data
        X_train_bow = vectorizer.fit_transform(X_train)

        # Transform testing data
        X_test_bow = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test

        logger.info("Successfully applied Bag of Words transformation.")
        return train_df, test_df
    except KeyError:
        logger.error("Missing 'content' or 'sentiment' column in the dataset.", exc_info=True)
        raise
    except Exception as e:
        logger.critical("Unexpected error occurred while applying Bag of Words.", exc_info=True)
        raise

# Store the data inside data/features

# data_path = os.path.join('data','features')

# os.makedirs(data_path)

# train_df.to_csv(os.path.join(data_path,'train_bow.csv'))

# test_df.to_csv(os.path.join(data_path,'test_bow.csv'))

def save_features(train_df: pd.DataFrame, test_df: pd.DataFrame, data_path: str) -> None:
    """Save transformed train and test data to CSV files."""
    try:
        os.makedirs(data_path, exist_ok=True)
        train_df.to_csv(os.path.join(data_path, 'train_tfidf.csv'), index=False)
        test_df.to_csv(os.path.join(data_path, 'test_tfidf.csv'), index=False)
        logger.info("Feature-engineered data saved successfully.")
    except Exception as e:
        logger.critical("Error occurred while saving feature-engineered data.", exc_info=True)
        raise

def main() -> None:
    try:
        logger.info("Starting feature engineering process.")

        # Load parameters
        max_features = load_params('params.yaml')

        # Load preprocessed data
        train_data, test_data = load_data('./data/interim/train_processed.csv', './data/interim/test_processed.csv')

        # Apply Bag of Words transformation
        train_df, test_df = apply_tfidf(train_data, test_data, max_features)

        # Save transformed data
        save_features(train_df, test_df, os.path.join('./data', 'processed'))

        logger.info("Feature engineering stage completed successfully.")
    except Exception as e:
        logger.critical("Feature engineering process failed.", exc_info=True)
        raise

if __name__ == "__main__":
    main()