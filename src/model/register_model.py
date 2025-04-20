# register_model.py

import json
import mlflow
import logging
import dagshub
import os

# # Set up DagsHub credentials for MLflow tracking
# dagshub_token = os.getenv("DAGSHUB_PAT")
# if not dagshub_token:
#     raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# after cicd remove this two lines
import dagshub
dagshub.init(repo_owner='Prayesh13', repo_name='youtube-comments-analysis', mlflow=True)


# Define DagsHub URL and repository details
dagshub_url = "https://dagshub.com"
repo_owner = "Prayesh13"
repo_name = "youtube-comments-analysis"


# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

# Configure logging
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)

        client = mlflow.tracking.MlflowClient()

        # Get all versions of the model
        all_versions = client.search_model_versions(f"name='{model_name}'")
        all_versions_sorted = sorted(all_versions, key=lambda v: int(v.version), reverse=True)

        latest_version = int(model_version.version)
        previous_versions = [v for v in all_versions_sorted if int(v.version) < latest_version]

        # ✅ DO NOT set 'latest' alias manually anymore

        # Assign alias 'previous{version}' to the most recent previous version
        if previous_versions:
            previous_version = int(previous_versions[0].version)
            alias_name = f"previous{previous_version}"
            client.set_registered_model_alias(model_name, alias_name, previous_version)
            logger.debug(f"Alias '{alias_name}' set to version {previous_version}")

        logger.debug(f"Model {model_name} version {model_version.version} registered successfully.")

    except Exception as e:
        logger.error("Error during model registration: %s", e)
        raise

def main():
    try:
        model_info_path = 'experiment_info.json'
        model_info = load_model_info(model_info_path)

        model_name = "Yt_Chrome_plugin_model"
        register_model(model_name, model_info)
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
