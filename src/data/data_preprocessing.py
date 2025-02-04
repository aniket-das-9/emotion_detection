import numpy as np
import pandas as pd

import os

import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

import logging

logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('data_preprocessing.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)
# Fetch the data from data/raw

# train_data = pd.read_csv('./data/raw/train.csv')
# test_data = pd.read_csv('./data/raw/test.csv')

# Transform the data

nltk.download('wordnet')
nltk.download('stopwords')

def lemmatization(text:str) -> str:
    try:
        lemmatizer= WordNetLemmatizer()
        text = text.split()
        text=[lemmatizer.lemmatize(y) for y in text]
        return " " .join(text)
    except Exception as e:
        logger.error("Error in lemmatization function.", exc_info=True)
        raise

def remove_stop_words(text:str) -> str:
    try:
        stop_words = set(stopwords.words("english"))
        Text=[i for i in str(text).split() if i not in stop_words]
        return " ".join(Text)
    except Exception as e:
        logger.error("Error in remove_stop_words function.", exc_info=True)
        raise

def removing_numbers(text:str) -> str:
    try:
        text=''.join([i for i in text if not i.isdigit()])
        return text
    except Exception as e:
        logger.error("Error in removing_numbers function.", exc_info=True)
        raise

def lower_case(text:str) -> str:
    try:
        text = text.split()
        text=[y.lower() for y in text]
        return " " .join(text)
    except Exception as e:
        logger.error("Error in lower_case function.", exc_info=True)
        raise

def removing_punctuations(text:str) -> str:
    try:
        ## Remove punctuations
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
        text = text.replace('؛',"", )

        ## remove extra whitespace
        text = re.sub('\s+', ' ', text)
        text =  " ".join(text.split())
        return text.strip()
    except Exception as e:
        logger.error("Error in removing_punctuations function.", exc_info=True)
        raise

def removing_urls(text:str) -> str:
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    except Exception as e:
        logger.error("Error in removing_urls function.", exc_info=True)
        raise

def remove_small_sentences(df: pd.DataFrame) -> None:
    try:
        for i in range(len(df)):
            if len(df.text.iloc[i].split()) < 3:
                df.text.iloc[i] = np.nan
    except Exception as e:
        logger.error("Error in remove_small_sentences function.", exc_info=True)
        raise

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.content=df.content.apply(lambda content : lower_case(content))
        df.content=df.content.apply(lambda content : remove_stop_words(content))
        df.content=df.content.apply(lambda content : removing_numbers(content))
        df.content=df.content.apply(lambda content : removing_punctuations(content))
        df.content=df.content.apply(lambda content : removing_urls(content))
        df.content=df.content.apply(lambda content : lemmatization(content))
        return df
    except Exception as e:
        logger.error("Error in normalize_text function.", exc_info=True)
        raise

# train_processed_data = normalize_text(train_data)
# test_processed_data = normalize_text(test_data)

def save_data(train_processed_data: pd.DataFrame, test_processed_data: pd.DataFrame, data_path: str) -> None:
    try:
        # Store the data inside data/processed
        interim_data_path = os.path.join(data_path,'interim')
        os.makedirs(interim_data_path, exist_ok=True)
        train_processed_data.to_csv(os.path.join(interim_data_path,'train_processed.csv'))
        test_processed_data.to_csv(os.path.join(interim_data_path,'test_processed.csv'))
        logger.info("Processed data saved successfully.")
    except Exception as e:
        print(f"Error: An unexpected error occurred while saving the data.")
        print(e)
        raise


def main() -> None:
    try:
        # Fetch data from data/raw
        logger.info("Loading raw data...")
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.info("Raw data loaded successfully.")

        # Normalize and preprocess the data
        logger.info("Starting data normalization...")
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)
        logger.info("Data normalization completed successfully.")

        # Store the data in data/processed
        save_data(train_processed_data,test_processed_data,data_path='./data')

    except Exception as e:
        logger.critical("Data preprocessing failed.", exc_info=True)
        raise

if __name__ == "__main__":
    main()