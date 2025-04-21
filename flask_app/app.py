import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import dagshub
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
import io
import matplotlib.dates as mdates
import numpy as np
import os
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import matplotlib
matplotlib.use('Agg')

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


@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    """Endpoint to predict sentiment of YouTube comments."""
    try:
        data = request.json
        comments_data = data.get('comments')
        
        if not comments_data:
            return jsonify({"error": "No comments provided"}), 400
        
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]

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

        # Return results
        response = [{"comment": comment, "sentiment": sentiment, "timestamp": timestamp} for comment, sentiment, timestamp in zip(comments, mapped_predictions, timestamps)]
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


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


        # Return results
        response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, mapped_predictions)]
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')
        
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        # Prepare data for the pie chart
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")
        
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']  # Blue, Gray, Red

        # Generate the pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Save the chart to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments')

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        # Preprocess comments
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Combine all comments into a single string
        text = ' '.join(preprocessed_comments)

        # Generate the word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        # Save the word cloud to a BytesIO object
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500

@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        print(sentiment_data)

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        # Convert sentiment_data to DataFrame
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Set the timestamp as the index
        df.set_index('timestamp', inplace=True)

        # Ensure the 'sentiment' column is numeric
        df['sentiment'] = df['sentiment'].astype(int)

        # Map sentiment values to labels
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        # Resample the data over monthly intervals and count sentiments
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)

        # Calculate total counts per month
        monthly_totals = monthly_counts.sum(axis=1)

        # Calculate percentages
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure all sentiment columns are present
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        # Sort columns by sentiment value
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        # Plotting
        plt.figure(figsize=(12, 6))

        colors = {
            -1: 'red',     # Negative sentiment
            0: 'gray',     # Neutral sentiment
            1: 'green'     # Positive sentiment
        }

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)

        # Format the x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        # Save the trend graph to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
