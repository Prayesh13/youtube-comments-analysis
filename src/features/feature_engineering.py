# src/feature_engineering/feature_engineering.py

import os
import re
import pickle
import torch
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import logging
import yaml
import pickle

# Logging setup
logger = logging.getLogger('feature_engineering')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('feature_engineering_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# 1. Tokenizer function
def tokenizer(text):
    return re.findall(r'\b\w+\b', text.lower())

# 2. Build vocabulary
def build_vocab(texts, min_freq=1):
    counter = Counter()
    for text in texts:
        counter.update(tokenizer(text))
    vocab = {"<unk>": 0, "<pad>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

# 3. Custom Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, tokenizer, max_len=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.texts[idx])
        ids = [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens][:self.max_len]
        ids += [self.vocab["<pad>"]] * (self.max_len - len(ids))
        return torch.tensor(ids), torch.tensor(self.labels[idx])

# 4. Create DataLoaders
def create_loaders(X_train, y_train, X_test, y_test, vocab, tokenizer, batch_size=32, max_len=100):
    train_dataset = TextDataset(X_train.tolist(), y_train.tolist(), vocab, tokenizer, max_len)
    test_dataset = TextDataset(X_test.tolist(), y_test.tolist(), vocab, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader

# 5. Load params.yaml
def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except Exception as e:
        logger.error('Failed to load params: %s', e)
        raise

# 6. Get root directory
def get_root_directory() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))

# 7. Save object
def save_object(obj, file_path: str):
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
        logger.debug(f"Object saved to {file_path}")
    except Exception as e:
        logger.error('Failed to save object: %s', e)
        raise

# 8. Load data
def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug('Data loaded successfully from %s', file_path)
        return df
    except Exception as e:
        logger.error('Failed to load data: %s', e)
        raise

# 9. Main function
def main():
    try:
        root_dir = get_root_directory()
        params = load_params(os.path.join(root_dir, 'params.yaml'))
        
        batch_size = params['feature_engineering']['batch_size']
        max_len = params['feature_engineering']['max_len']

        # Load training and testing data
        train_data = load_data(os.path.join(root_dir, 'data/interim/train_processed.csv'))
        test_data = load_data(os.path.join(root_dir, 'data/interim/test_processed.csv'))

        X_train = train_data['clean_comment']
        y_train = train_data['category']
        X_test = test_data['clean_comment']
        y_test = test_data['category']

        # Build vocabulary
        logger.info("Building vocabulary...")
        vocab = build_vocab(X_train)
        logger.info(f"Vocabulary size: {len(vocab)}")

        # Create data loaders
        train_loader, test_loader = create_loaders(X_train, y_train, X_test, y_test, vocab, tokenizer, batch_size, max_len)

        # Save vocab, tokenizer, loaders
        os.makedirs(os.path.join(root_dir, 'data/processed'), exist_ok=True)

        save_object(vocab, os.path.join(root_dir, 'data/processed/vocab.pkl'))
        save_object(tokenizer, os.path.join(root_dir, 'data/processed/tokenizer.pkl'))
        save_object(train_loader, os.path.join(root_dir, 'data/processed/train_loader.pkl'))
        save_object(test_loader, os.path.join(root_dir, 'data/processed/test_loader.pkl'))

    except Exception as e:
        logger.error('Feature engineering process failed: %s', e)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
