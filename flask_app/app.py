import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import dagshub
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
import numpy as np
import os
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment

# Initialize DagsHub and MLflow
dagshub.init(repo_owner='Prayesh13', repo_name='youtube-comments-analysis', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Prayesh13/youtube-comments-analysis.mlflow")

# Load the model info from model registry
def load_model_vocab_from_registry(model_name, model_version, vocab_path) -> dict:
    try:
        model_uri = f"models:/{model_name}/{model_version}"
        model = mlflow.pytorch.load_model(model_uri)

        # Load the vocab.pkl
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        return model, vocab
    except Exception as e:
        print(f"Error loading model or vocabulary: {e}")
        raise

# Load the model and vocabulary from the registry
model, vocab = load_model_vocab_from_registry("Yt_Chrome_plugin_model", "2", "models/vocab.pkl")
print("Model and vocabulary loaded successfully.")

def tokenizer(text):
    return re.findall(r'\b\w+\b', text.lower())

# 3. Preprocess text into tensor
def preprocess_text(text, vocab, max_len=100):
    tokens = tokenizer(text)
    ids = [vocab.get(token, vocab["<unk>"]) for token in tokens]
    ids = ids[:max_len] + [vocab["<pad>"]] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to predict sentiment of YouTube comments."""
    try:
        # Get the JSON data from the request
        data = request.get_json()

        # Check if 'comments' key is provided in the request
        comments = data.get('comments', [])
        
        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        # Full preprocessing pipeline
        processed_comments = [preprocess_comment(comment) for comment in comments]
        inputs = [preprocess_text(comment, vocab) for comment in processed_comments]

        # Stack into a batch
        input_tensor = torch.stack(inputs)

        # Set model to eval mode
        model.eval()

        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            predictions = torch.argmax(outputs, dim=1).tolist()

        # Map predictions to labels
        label_mapping = {0: -1, 1: 0, 2: 1}
        mapped_predictions = [label_mapping[pred] for pred in predictions]

        # Prepare the response: pair each comment with its prediction
        results = [
            {"comment": comment, "sentiment": sentiment}
            for comment, sentiment in zip(comments, mapped_predictions)
        ]

        # Return results
        return jsonify({"results": results}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
