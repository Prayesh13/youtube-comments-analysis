# src/model/model_building.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from collections import Counter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow
import mlflow.pytorch
import numpy as np
import os
import logging
import yaml
import nltk
import pickle
import json
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


def load_object(file_path: str):
    """General function to load a pickled object (e.g., model, vectorizer, etc.)."""
    try:
        if not os.path.exists(file_path):
            logger.error('File does not exist: %s', file_path)
            raise FileNotFoundError(f"File does not exist: {file_path}")

        with open(file_path, 'rb') as file:
            obj = pickle.load(file)
        logger.debug('Object loaded successfully from %s', file_path)
        return obj

    except FileNotFoundError as e:
        logger.error('File not found error: %s', e)
        raise
    except pickle.UnpicklingError as e:
        logger.error('Error unpickling file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error loading object: %s', e)
        raise


def load_vocab_tokenizer_dataloaders(root_dir: str):
    """Load vocab, tokenizer, and dataloaders from pickled files."""
    try:
        vocab = load_object(os.path.join(root_dir, 'data/processed/vocab.pkl'))
        tokenizer = load_object(os.path.join(root_dir, 'data/processed/tokenizer.pkl'))
        
        train_loader = load_object(os.path.join(root_dir, 'data/processed/train_loader.pkl'))
        test_loader = load_object(os.path.join(root_dir, 'data/processed/test_loader.pkl'))

        logger.debug('Vocab, tokenizer, and dataloaders loaded successfully.')
        
        return vocab, tokenizer, train_loader, test_loader

    except Exception as e:
        logger.error('Error loading vocab, tokenizer, or dataloaders: %s', e)
        raise


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
    num_epochs = 2

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
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

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
        mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            logger.debug('Saved best model at epoch %d', epoch)
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


def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        # Save the model's state_dict (weights and parameters)
        torch.save(model.state_dict(), file_path)
        logger.debug('Model saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise


def save_params(best_params, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            json.dump(best_params, file)
        logger.debug('Parameters saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the parameters as json: %s', e)
        raise


def get_root_directory() -> str:
    """Get the root directory (two levels up from this script's location)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))


def main():
    try:
        # Get root directory and resolve the path for params.yaml
        root_dir = get_root_directory()

        # Load parameters from the root directory
        params = load_params(os.path.join(root_dir, 'params.yaml'))

        vocab, tokenizer, train_loader, val_loader, test_loader = load_vocab_tokenizer_dataloaders(root_dir)

        embed_dim = params['model_building']['embed_dim']
        hidden_dim = params['model_building']['hidden_dim']
        learning_rate = params['model_building']['learning_rate']
        batch_size = params['model_building']['batch_size']

        best_params = {
            'embed_dim': embed_dim,
            'hidden_dim': hidden_dim,
            'learning_rate': learning_rate,
            'batch_size': batch_size
        }


        # Train the LightGBM model using hyperparameters from params.yaml
        best_model = train_lgbm(X_train_tfidf, y_train, learning_rate, max_depth, n_estimators)

        # Save the trained model in the root directory
        save_model(best_model, os.path.join(root_dir, 'lgbm_model.pkl'))

    except Exception as e:
        logger.error('Failed to complete the feature engineering and model building process: %s', e)
        print(f"Error: {e}")

def main():
    try:
        # Load params
        params = load_params('params.yaml')

        # Model, Dataset and DataLoader initialization should happen here
        # Example (dummy placeholders):
        # model = YourModelClass()
        # train_loader = DataLoader(...)
        # val_loader = DataLoader(...)
        # test_loader = DataLoader(...)
        # optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        # criterion = nn.CrossEntropyLoss()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        with mlflow.start_run():
            logger.info('Training started')
            train_model(model, train_loader, val_loader, optimizer, criterion, device)
            logger.info('Training completed')

            logger.info('Evaluating model')
            evaluate_model(model, test_loader, criterion, device)
            mlflow.pytorch.log_model(model, "model")

    except Exception as e:
        logger.error('Error in main execution: %s', e)
        print(f"Error occurred: {e}")

if __name__ == '__main__':
    main()



if __name__ == '__main__':
    main()