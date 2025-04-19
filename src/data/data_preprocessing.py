# src/data/data_preprocessing.py

import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

# Logging configuration
logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('preprocessing_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Download required NLTK data
nltk.download('wordnet')
nltk.download('stopwords')

# Preprocessing helper functions
def lower_case(df: pd.Series) -> pd.Series:
    try:
        return df.str.lower()
    except Exception as e:
        logger.error(f"Error in lower_case: {e}")
        return df

def remove_special_characters(df: pd.Series) -> pd.Series:
    try:
        return df.str.replace(r"[^a-zA-Z0-9]", " ", regex=True)
    except Exception as e:
        logger.error(f"Error in remove_special_characters: {e}")
        return df

def remove_extra_spaces(df: pd.Series) -> pd.Series:
    try:
        return df.str.replace(r"\s+", " ", regex=True)
    except Exception as e:
        logger.error(f"Error in remove_extra_spaces: {e}")
        return df

def remove_leading_trailing_spaces(df: pd.Series) -> pd.Series:
    try:
        return df.str.strip()
    except Exception as e:
        logger.error(f"Error in remove_leading_trailing_spaces: {e}")
        return df

def remove_numbers(df: pd.Series) -> pd.Series:
    try:
        return df.str.replace(r"\d+", "", regex=True)
    except Exception as e:
        logger.error(f"Error in remove_numbers: {e}")
        return df

def remove_stopwords(df: pd.Series) -> pd.Series:
    try:
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        return df.apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))
    except Exception as e:
        logger.error(f"Error in remove_stopwords: {e}")
        return df

def remove_punctuation(df: pd.Series) -> pd.Series:
    try:
        return df.str.replace(r"[^\w\s]", "", regex=True)
    except Exception as e:
        logger.error(f"Error in remove_punctuation: {e}")
        return df

def lemmatize_words(df: pd.Series) -> pd.Series:
    try:
        lemmatizer = WordNetLemmatizer()
        return df.apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
    except Exception as e:
        logger.error(f"Error in lemmatize_words: {e}")
        return df

def remove_urls(df: pd.Series) -> pd.Series:
    try:
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return df.str.replace(url_pattern, '', regex=True)
    except Exception as e:
        logger.error(f"Error in remove_urls: {e}")
        return df

def remove_null_values(df: pd.DataFrame) -> pd.DataFrame:
    try:
        return df.dropna()
    except Exception as e:
        logger.error(f"Error in remove_null_values: {e}")
        return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    try:
        return df.drop_duplicates()
    except Exception as e:
        logger.error(f"Error in remove_duplicates: {e}")
        return df

def remove_empty_rows(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Remove rows where specified column is empty after stripping."""
    try:
        df = df[~(df[column_name].str.strip() == "")]
        return df
    except Exception as e:
        logger.error(f"Error in remove_empty_rows: {e}")
        return df

# Main preprocessing pipeline
def preprocess_comment(comment):
    """Old-style single comment preprocessing (retained for use if needed)."""
    try:
        comment = comment.lower()
        comment = comment.strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])
        return comment
    except Exception as e:
        logger.error(f"Error in preprocess_comment: {e}")
        return comment

def normalize_text(df):
    """Apply full preprocessing pipeline to text."""
    try:
        logger.debug('Starting text normalization...')

        # Assume column name is 'clean_comment'
        df['clean_comment'] = lower_case(df['clean_comment'])
        df['clean_comment'] = remove_urls(df['clean_comment'])
        df['clean_comment'] = remove_special_characters(df['clean_comment'])
        df['clean_comment'] = remove_numbers(df['clean_comment'])
        df['clean_comment'] = remove_punctuation(df['clean_comment'])
        df['clean_comment'] = remove_stopwords(df['clean_comment'])
        df['clean_comment'] = lemmatize_words(df['clean_comment'])
        df['clean_comment'] = remove_extra_spaces(df['clean_comment'])
        df['clean_comment'] = remove_leading_trailing_spaces(df['clean_comment'])

        logger.debug('Text normalization finished successfully.')
        return df
    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        raise

def final_cleaning(df):
    """Final step to clean dataset: remove empty rows, nulls, and duplicates."""
    try:
        logger.debug('Starting final cleaning: remove empty rows, nulls, and duplicates...')
        df = remove_empty_rows(df, 'clean_comment')
        df = remove_null_values(df)
        df = remove_duplicates(df)
        logger.debug('Final cleaning completed.')
        return df
    except Exception as e:
        logger.error(f"Error in final_cleaning: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        interim_data_path = os.path.join(data_path, 'interim')
        logger.debug(f"Creating directory {interim_data_path}")

        os.makedirs(interim_data_path, exist_ok=True)
        logger.debug(f"Directory {interim_data_path} created or already exists")

        train_data.to_csv(os.path.join(interim_data_path, "train_processed.csv"), index=False)
        test_data.to_csv(os.path.join(interim_data_path, "test_processed.csv"), index=False)

        logger.debug(f"Processed data saved to {interim_data_path}")
    except Exception as e:
        logger.error(f"Error occurred while saving data: {e}")
        raise

def main():
    try:
        logger.debug("Starting data preprocessing...")

        # Load raw data
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Raw data loaded successfully.')

        # Preprocessing
        train_processed = normalize_text(train_data)
        test_processed = normalize_text(test_data)

        # Final cleaning
        train_processed = final_cleaning(train_processed)
        test_processed = final_cleaning(test_processed)

        # Save processed data
        save_data(train_processed, test_processed, data_path='./data')

    except Exception as e:
        logger.error(f"Failed to complete data preprocessing: {e}")
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
