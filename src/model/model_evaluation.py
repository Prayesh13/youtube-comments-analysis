# src/model/model_evaluation.py
import os
import json
import yaml
import torch
import logging
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import mlflow
import mlflow.pytorch
import dagshub

# Initialize DagsHub and MLflow
dagshub.init(repo_owner='Prayesh13', repo_name='youtube-comments-analysis', mlflow=True)

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


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


def tokenizer(text):
    return re.findall(r'\b\w+\b', text.lower())


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.texts[idx].split()
        ids = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens][:self.max_len]
        ids += [self.vocab['<pad>']] * (self.max_len - len(ids))
        return torch.tensor(ids), torch.tensor(self.labels[idx], dtype=torch.long)


def create_loader(texts, labels, vocab, batch_size, max_len):
    dataset = TextDataset(texts.tolist(), labels.tolist(), vocab, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


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


def load_model(file_path: str):
    """Load the full model from a file."""
    try:
        # Add the custom class to the safe globals if you're confident in the checkpoint's source
        # torch.serialization.add_safe_globals([BiLSTMAttentionModel])
        
        model = torch.load(file_path, map_location=torch.device('cpu'), weights_only=False)
        model.eval()
        logger.debug('Model loaded from %s', file_path)
        return model
    except Exception as e:
        logger.error('Error occurred while loading the full model: %s', e)
        raise



def evaluate_model(model, test_loader, device):
    """Evaluate the model on the test data."""
    try:
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                _, preds = torch.max(output, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, target_names=['Category -1', 'Category 0', 'Category 1'])

        return accuracy, cm, report
    except Exception as e:
        logger.error('Error occurred during model evaluation: %s', e)
        raise


def save_results(accuracy, cm, report, reports_dir):
    """Save evaluation results to the reports directory."""
    try:
        if accuracy is None or cm is None or report is None:
            raise ValueError("Accuracy, confusion matrix, or report is None. Cannot save results.")

        # Ensure reports directory exists
        os.makedirs(reports_dir, exist_ok=True)

        # Save accuracy, confusion matrix, and classification report to a JSON file
        evaluation_results = {
            'accuracy': float(accuracy),  # Make sure it's serializable
            'confusion_matrix': cm.tolist(),  # Convert numpy array to list
            'classification_report': report  # Assume this is a dict
        }

        # Define the path
        evaluation_file = os.path.join(reports_dir, 'evaluation_results.json')

        # Save as JSON
        with open(evaluation_file, 'w') as f:
            json.dump(evaluation_results, f, indent=4)

        logger.debug(f"Evaluation results saved successfully to {evaluation_file}")

    except Exception as e:
        logger.error(f'Error occurred while saving evaluation results: {e}')
        raise

    
def log_confusion_matrix(cm, report, output_dir):
    """Log the confusion matrix and save it as an image file."""
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1', 'Class 2'], yticklabels=['Class 0', 'Class 1', 'Class 2'])
        plt.title("Confusion Matrix")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        cm_image_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(cm_image_path)
        plt.close()

        logger.debug(f"Confusion Matrix image saved and logged at: {cm_image_path}")
        return cm_image_path
    except Exception as e:
        logger.error('Error while logging confusion matrix: %s', e)
        raise


def save_model_info(run_id, model_path, output_file):
    """Save model information to a JSON file."""
    try:
        model_info = {
            'run_id': run_id,
            'model_path': model_path
        }

        # Define the path
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        output_file_path = os.path.join(root_dir, output_file)

        # Save as JSON
        with open(output_file_path, 'w') as f:
            json.dump(model_info, f, indent=4)

        logger.debug(f"Model info saved successfully to {output_file_path}")

    except Exception as e:
        logger.error(f'Error occurred while saving model info: {e}')
        raise


def load_vocab(vocab_path):
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab


def main():

    mlflow.set_tracking_uri("https://dagshub.com/Prayesh13/youtube-comments-analysis.mlflow")
    mlflow.set_experiment("dvc-pipeline")

    try:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        params = load_params(os.path.join(root_dir, 'params.yaml'))
        vocab = load_vocab(os.path.join(root_dir, 'models/vocab.pkl'))

        test_data = pd.read_csv(os.path.join(root_dir, 'data/interim/test_processed.csv'))
        test_data.fillna('', inplace=True)

        X_test = test_data['clean_comment']
        y_test = test_data['category'].map({-1: 0, 0: 1, 1: 2})

        max_len = params['model_building']['max_len']
        batch_size = params['model_building']['batch_size']

        test_loader = create_loader(X_test, y_test, vocab, batch_size, max_len)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the trained model
        model_path = os.path.join(root_dir, 'models', 'bilstm_attention_model.pth')
        best_params = params['model_building']
        
        model = load_model(model_path)
        model = model.to(device)

        # Start MLflow logging
        with mlflow.start_run() as run:
            # Log model-building parameters
            mlflow.log_params(params['model_building'])
            logger.debug('Logged model-building parameters to MLflow.')

            # Take a small example input for logging
            example_batch = next(iter(test_loader))[0]  # Only X_batch
            example_batch = example_batch.to(device)
            input_example = example_batch[0].unsqueeze(0).cpu().numpy()  # Single input, make batch dimension

            # Log the PyTorch model along with input_example
            model_name = "bilstm_attention_model"
            mlflow.pytorch.log_model(model, model_name, input_example=input_example)
            logger.debug('Logged PyTorch model to MLflow with input_example.')

            model_path = model_name
            logger.debug(f"Model path: {model_path}")

            # Save model info
            save_model_info(run.info.run_id, model_path, 'experiment_info.json')
            logger.debug('Saved model info to JSON.')   

            # mlflow.pytorch.log_model(model, "model")
            logger.debug('Logged PyTorch model to MLflow.')

            mlflow.log_artifact(os.path.join(root_dir, 'models/vocab.pkl'))
            logger.debug('Logged vocabulary to MLflow.')

            accuracy, cm, report = evaluate_model(model, test_loader, device)

            # Save evaluation results to JSON
            reports_dir = os.path.join(root_dir, 'reports')
            os.makedirs(reports_dir, exist_ok=True)  # Ensure directory exists
            save_results(accuracy, cm, report, reports_dir)
            logger.debug('Saved evaluation results to JSON.')

            # Log evaluation metrics
            mlflow.log_metric("accuracy", accuracy)
            logger.debug(f"Logged accuracy: {accuracy}")

            # Log confusion matrix and classification report as artifacts
            cm_image_path = log_confusion_matrix(cm, report, os.path.join(root_dir, 'reports'))
            mlflow.log_artifact(os.path.join(root_dir, 'reports', 'evaluation_results.json'))
            mlflow.log_artifact(cm_image_path)
            logger.debug('Logged confusion matrix image and classification report to MLflow.')


        logger.info(f"Test Accuracy: {accuracy * 100:.2f}%")
        logger.info(f"Confusion Matrix: \n{cm}")
        logger.info(f"Classification Report: \n{report}")

    except Exception as e:
        logger.error('Critical error occurred in main: %s', e)
        print(f"Critical error: {e}")


if __name__ == '__main__':
    main()
