from flask import Flask, request, jsonify, render_template
import joblib
import torch
import os
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from huggingface_hub import snapshot_download

# Load traditional models and vectorizer
log_reg_model = joblib.load('models/logistic_regression_model.pkl')
nb_model = joblib.load('models/naive_bayes_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Load DistilBERT model and tokenizer from Hugging Face
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check if model exists locally, otherwise download
model_path = "distilbert-fake-news-model"
if not os.path.exists(model_path):
    model_path = snapshot_download("Barhomy/distilbert-fake-news-model", 
                                  cache_dir="./distilbert-fake-news-model")

distilbert_model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
distilbert_tokenizer = DistilBertTokenizer.from_pretrained(model_path)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form.get('news_text')
    model_choice = request.form.get('model_choice')

    if not news_text or not model_choice:
        return jsonify({'error': 'No text or model selected'})

    try:
        if model_choice == 'logistic_regression':
            news_tfidf = vectorizer.transform([news_text])
            prediction = log_reg_model.predict(news_tfidf)[0]
            probabilities = log_reg_model.predict_proba(news_tfidf)[0]
            confidence = max(probabilities)
            
        elif model_choice == 'naive_bayes':
            news_tfidf = vectorizer.transform([news_text])
            prediction = nb_model.predict(news_tfidf)[0]
            probabilities = nb_model.predict_proba(news_tfidf)[0]
            confidence = max(probabilities)
            
        elif model_choice == 'distilbert':
            inputs = distilbert_tokenizer(news_text, return_tensors="pt", 
                                        truncation=True, padding=True, 
                                        max_length=512).to(device)
            with torch.no_grad():
                outputs = distilbert_model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                prediction = torch.argmax(logits).item()
                confidence = probabilities[0][prediction].item()
        else:
            return jsonify({'error': 'Invalid model choice'})

        result = "FAKE" if prediction == 1 else "TRUE"
        return jsonify({
            'prediction': result,
            'confidence': f"{confidence:.2%}"
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)