import pytest
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.pytorch
import torch
from mlflow.tracking import MlflowClient
import os
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_YOUTUBE_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_YOUTUBE_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "Prayesh13"
repo_name = "youtube-comments-analysis"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

def load_latest_model(model_name):
    try:
        # Create an MLflow Client
        client = MlflowClient()

        # Get the latest version of the model
        latest_version_info = client.get_latest_versions(model_name, stages=["None"])  # You can add stages like 'production', 'staging', etc.
        
        if latest_version_info:
            # Get the latest version number
            latest_version = latest_version_info[0].version
            print(f"The latest version of the model '{model_name}' is {latest_version}")
            
            # Construct the model URI
            model_uri = f"models:/{model_name}/{latest_version}"
            
            # Load the model from MLflow using PyTorch
            model = mlflow.pytorch.load_model(model_uri)
            print(f"Model '{model_name}' version {latest_version} successfully loaded.")
            
            return model
        else:
            print(f"No model found with name '{model_name}'")
            return None
            
    except Exception as e:
        print(f"An error occurred while fetching and loading the latest model '{model_name}': {str(e)}")
        return None

@pytest.mark.parametrize("model_name, holdout_data_path, vocab_path", [
    ("yt_chrome_plugin_model", "data/interim/test_processed.csv", "models/vocab.pkl"),  # Replace with your actual paths
])
def test_model_performance(model_name, holdout_data_path, vocab_path):
    try:
        # Load the vocabulary (for tokenization)
        with open(vocab_path, 'rb') as file:
            vocab = pickle.load(file)

        # Load the holdout test data
        holdout_data = pd.read_csv(holdout_data_path)
        X_holdout_raw = holdout_data.iloc[:, :-1].squeeze()  # Raw text features (assuming text is in the first column)
        y_holdout = holdout_data.iloc[:, -1]  # Labels

        # Handle NaN values in the text data
        X_holdout_raw = X_holdout_raw.fillna("")

        # Tokenization process using the vocabulary (dummy tokenization here, replace with actual tokenization)
        X_holdout_tokenized = [vocab.get(word, 0) for word in X_holdout_raw.str.split().sum()]  # Convert text to token indices

        # Padding sequences
        max_length = 100  # Define the max length of the sequence
        X_holdout_tokenized = [seq[:max_length] + [0] * (max_length - len(seq)) for seq in X_holdout_tokenized]

        # Convert to PyTorch tensors
        X_holdout_tensor = torch.tensor(X_holdout_tokenized)
        y_holdout_tensor = torch.tensor(y_holdout.values)

        # Create DataLoader for batching
        batch_size = 32
        dataset = TensorDataset(X_holdout_tensor, y_holdout_tensor)
        data_loader = DataLoader(dataset, batch_size=batch_size)

        # Load the latest model from MLflow
        model = load_latest_model(model_name)
        assert model is not None, f"No model found or failed to load: '{model_name}'"

        model.eval()  # Set the model to evaluation mode

        # Define performance metrics
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                inputs, labels = batch
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate performance metrics
        accuracy_new = accuracy_score(all_labels, all_preds)
        precision_new = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
        recall_new = recall_score(all_labels, all_preds, average='weighted', zero_division=1)
        f1_new = f1_score(all_labels, all_preds, average='weighted', zero_division=1)

        # Define expected thresholds for the performance metrics
        expected_accuracy = 0.40
        expected_precision = 0.40
        expected_recall = 0.40
        expected_f1 = 0.40

        # Assert that the new model meets the performance thresholds
        assert accuracy_new >= expected_accuracy, f'Accuracy should be at least {expected_accuracy}, got {accuracy_new}'
        assert precision_new >= expected_precision, f'Precision should be at least {expected_precision}, got {precision_new}'
        assert recall_new >= expected_recall, f'Recall should be at least {expected_recall}, got {recall_new}'
        assert f1_new >= expected_f1, f'F1 score should be at least {expected_f1}, got {f1_new}'

        print(f"Performance test passed for model '{model_name}'")

    except Exception as e:
        pytest.fail(f"Model performance test failed with error: {e}")