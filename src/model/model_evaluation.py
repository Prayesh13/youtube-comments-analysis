# src/model/model_evaluation.py

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
import logging
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import json
import pickle
from model_building import BiLSTMAttentionModel, tokenizer, build_vocab, TextDataset, create_loaders
import dagshub
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

def load_trained_model(model_path: str, model_class: nn.Module, vocab_size: int, embed_dim: int, hidden_dim: int, output_dim: int, vocab) -> nn.Module:
    """Load the trained model from a file."""
    try:
        model = model_class(vocab_size, embed_dim, hidden_dim, output_dim, vocab)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        logger.debug('Model loaded successfully from %s', model_path)
        return model
    except Exception as e:
        logger.error('Error occurred while loading model: %s', e)
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


def save_vocab(vocab, vocab_path):
    """Save vocabulary to a pickle file."""
    try:
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab, f)
        logger.debug(f"Vocabulary saved successfully to {vocab_path}")
    except Exception as e:
        logger.error(f'Error occurred while saving vocabulary: {e}')
        raise

def main():
    mlflow.set_tracking_uri("https://dagshub.com/Prayesh13/youtube-comments-analysis.mlflow") 

    mlflow.set_experiment("dvc-pipeline")

    try:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

        # Load parameters from YAML file
        try:
            params = load_params(os.path.join(root_dir, 'params.yaml'))
        except Exception as e:
            logger.error('Error occurred while loading params: %s', e)
            raise

        # Load test data
        try:
            train_data = pd.read_csv(os.path.join(root_dir, 'data/interim/train_processed.csv'))
            train_data.fillna('', inplace=True)
            test_data = pd.read_csv(os.path.join(root_dir, 'data/interim/test_processed.csv'))
            test_data.fillna('', inplace=True)
            logger.debug('Test data loaded successfully')
        except Exception as e:
            logger.error('Error occurred while loading test data: %s', e)
            raise
        
        X_train = train_data['clean_comment']
        X_test = test_data['clean_comment']
        y_test = test_data['category']

        label_mapping = {-1: 0, 0: 1, 1: 2}
        y_test = y_test.map(label_mapping)

        # Build vocabulary from training data (or load from file if available)
        try:
            vocab = build_vocab(X_train)
            logger.debug('Vocabulary built successfully')
        except Exception as e:
            logger.error('Error occurred while building vocabulary: %s', e)
            raise

        try:
            save_vocab(vocab, os.path.join(root_dir, 'models/vocab.pkl'))
            logger.debug('Vocabulary saved successfully')
        except Exception as e:
            logger.error('Error occurred while saving vocabulary: %s', e)
            raise

        # Create test data loader
        try:
            max_len = params['model_building']['max_len']
            batch_size = params['model_building']['batch_size']
            test_loader = create_loaders(X_test, y_test, X_test, y_test, vocab, tokenizer, batch_size, max_len)[1]
            logger.debug('Test data loader created successfully')
        except Exception as e:
            logger.error('Error occurred while creating data loaders: %s', e)
            raise

        # Load the trained model
        model_path = os.path.join(root_dir, 'models', 'bilstm_attention_model.pth')
        best_params = params['model_building']
        try:
            model = load_trained_model(
                model_path=model_path,
                model_class=BiLSTMAttentionModel,
                vocab_size=len(vocab),
                embed_dim=best_params['embed_dim'],
                hidden_dim=best_params['hidden_dim'],
                output_dim=len(label_mapping),
                vocab=vocab
            )
            logger.debug('Model loaded successfully')
        except Exception as e:
            logger.error('Error occurred while loading the model: %s', e)
            raise

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)


        # Start MLflow logging
        with mlflow.start_run() as run:
            # Log model-building parameters
            mlflow.log_params(params['model_building'])
            logger.debug('Logged model-building parameters to MLflow.')

            # Log the trained model
            example_input = torch.randint(0, len(vocab), (1, max_len))  # batch_size=1, sequence_length=max_len

            # Take a small example input for logging
            example_batch = next(iter(test_loader))[0]  # Only X_batch
            example_batch = example_batch.to(device)
            input_example = example_batch[0].unsqueeze(0).cpu().numpy()  # Single input, make batch dimension

            # Log the PyTorch model along with input_example
            model_name = "bilstm_attention_model"
            mlflow.pytorch.log_model(model, model_name, input_example=input_example)
            logger.debug('Logged PyTorch model to MLflow with input_example.')

            artifact_uri = mlflow.get_artifact_uri()
            logger.debug(f"Artifact URI: {artifact_uri}")

            model_path = os.path.join(artifact_uri, model_name)
            logger.debug(f"Model path: {model_path}")

            # Save model info
            save_model_info(run.info.run_id, model_path, 'experiment_info.json')
            logger.debug('Saved model info to JSON.')

            # mlflow.pytorch.log_model(model, "model")
            logger.debug('Logged PyTorch model to MLflow.')

            mlflow.log_artifact(os.path.join(root_dir, 'models/vocab.pkl'))
            logger.debug('Logged vocabulary to MLflow.')

            # Evaluate the model
            try:
                accuracy, cm, report = evaluate_model(model, test_loader, device)
            except Exception as e:
                logger.error('Error occurred while evaluating the model: %s', e)
                raise

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
