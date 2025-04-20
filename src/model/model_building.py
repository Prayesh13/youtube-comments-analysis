# src/model/model_building.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import re
from collections import Counter
import numpy as np
import pandas as pd
import os
import logging
import yaml
import nltk
import json
import pickle
nltk.download('punkt')

# logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


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
        return torch.tensor(ids), torch.tensor(self.labels[idx], dtype=torch.long)

# 4. Create DataLoaders
def create_loaders(X_train, y_train, X_test, y_test, vocab, tokenizer, batch_size=32, max_len=100):
    train_dataset = TextDataset(X_train.tolist(), y_train.tolist(), vocab, tokenizer, max_len)
    test_dataset = TextDataset(X_test.tolist(), y_test.tolist(), vocab, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_output):
        attn_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context = torch.sum(attn_weights * lstm_output, dim=1)
        return context


class BiLSTMAttentionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, vocab):
        super(BiLSTMAttentionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab["<pad>"])
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        attn_output = self.attention(lstm_out)
        return self.fc(attn_output)


def train_model(model, train_loader, val_loader, optimizer, criterion, device):
    """Train the model."""
    best_val_loss = np.inf
    patience_counter = 0
    patience = 3
    num_epochs = 1

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                loss = criterion(output, y_batch)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            logger.debug('Best model at epoch %d', epoch)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info('Early stopping at epoch %d', epoch)
                break

    return model


def train_on_best_params(best_params, vocab, train_loader, test_loader, y_train):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build model with best hyperparameters
        model = BiLSTMAttentionModel(
            vocab_size=len(vocab),
            embed_dim=best_params['embed_dim'],
            hidden_dim=best_params['hidden_dim'],
            output_dim=y_train.nunique(),
            vocab=vocab
        ).to(device)

        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])

        # Train the model
        model = train_model(model, train_loader, test_loader, optimizer, criterion, device)

        logger.debug(f"Model trained with best parameters: {best_params}")

        return model

    except Exception as e:
        logger.debug(f"Error occurred during retraining and saving best model: {e}")
        raise


def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug('Data loaded successfully from %s', file_path)
        return df
    except Exception as e:
        logger.error('Failed to load data: %s', e)
        raise


def save_model(model, file_path: str) -> None:
    """Save the full model to a file."""
    try:
        # Save the entire model (architecture + weights)
        torch.save(model, file_path)
        logger.debug('Model saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the full model: %s', e)
        raise

def save_vocab(vocab, vocab_path):
    """Save vocabulary to a pickle file."""
    try:
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab, f)
        logger.debug(f"Vocabulary saved successfully to {vocab_path}")
    except Exception as e:
        logger.error(f'Error occurred while saving vocabulary: {e}')
        raise


def get_root_directory() -> str:
    """Get the root directory (two levels up from this script's location)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))


def main():
    try:
        root_dir = get_root_directory()
        params = load_params(os.path.join(root_dir, 'params.yaml'))

        # Load training and testing data
        train_data = load_data(os.path.join(root_dir, 'data/interim/train_processed.csv'))
        test_data = load_data(os.path.join(root_dir, 'data/interim/test_processed.csv'))

        X_train = train_data['clean_comment']
        y_train = train_data['category']
        X_test = test_data['clean_comment']
        y_test = test_data['category']

        label_mapping = {-1: 0, 0: 1, 1: 2}
        y_train = y_train.map(label_mapping)
        y_test = y_test.map(label_mapping)

        logger.debug('Data loaded successfully and output labels mapped.')

        # Build vocabulary
        logger.info("Building vocabulary...")
        vocab = build_vocab(X_train)
        logger.info(f"Vocabulary size: {len(vocab)}")

        save_vocab(vocab, os.path.join(root_dir, 'models/vocab.pkl'))
        logger.debug('Vocabulary saved successfully')

        max_len = params['model_building']['max_len']
        embed_dim = params['model_building']['embed_dim']
        hidden_dim = params['model_building']['hidden_dim']
        learning_rate = params['model_building']['learning_rate']
        batch_size = params['model_building']['batch_size']

        # Create data loaders
        train_loader, test_loader = create_loaders(X_train, y_train, X_test, y_test, vocab, tokenizer, batch_size, max_len)
        
        best_params = {
            'embed_dim': embed_dim,
            'hidden_dim': hidden_dim,
            'learning_rate': learning_rate,
            'batch_size': batch_size
        }


        # Train the LightGBM model using hyperparameters from params.yaml
        logger.info('Training started')
        best_model = train_on_best_params(
            best_params=best_params,
            vocab=vocab,
            train_loader=train_loader,
            test_loader=test_loader,
            y_train=y_train
        )

        # Save the trained model in the models directory
        os.makedirs(os.path.join(root_dir, 'models'), exist_ok=True)
        model_path = os.path.join(root_dir, 'models')
        save_model(best_model, os.path.join(model_path, 'bilstm_attention_model.pth'))

        # Save the best parameters in the models directory
        # save_params(best_params, os.path.join(model_path, 'best_params.json'))

    except Exception as e:
        logger.error('Failed to complete the feature engineering and model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
