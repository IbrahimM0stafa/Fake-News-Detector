# 📰 Fake News Detector

A Flask-based web application that uses machine learning and a fine-tuned DistilBERT model to classify news text as **FAKE** or **TRUE**.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Flask](https://img.shields.io/badge/flask-v2.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow.svg)

---

## 🔍 Features

- **Logistic Regression** model with TF-IDF vectorization
- **Naive Bayes** classifier for high recall detection
- **Fine-tuned DistilBERT** model (hosted on Hugging Face Hub)
- Real-time text classification through a simple web interface
- Multiple model comparison with confidence scores
- Responsive web design with modern UI

---

## ⚙️ Technologies Used

- **Backend**: Python, Flask
- **Machine Learning**: Scikit-learn, PyTorch, Hugging Face Transformers
- **Models**: Pre-trained `.pkl` models + Fine-tuned DistilBERT
- **Frontend**: HTML, CSS, JavaScript
- **Deployment**: Hugging Face Hub integration

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Barhomy/fake-news-detector.git
cd fake-news-detector
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

<details>
<summary><strong>📋 View requirements.txt</strong></summary>

```txt
flask==2.3.3
joblib==1.3.2
scikit-learn==1.3.0
torch==2.0.1
transformers==4.33.2
huggingface_hub==0.16.4
numpy==1.24.3
pandas==1.5.3
```

</details>

### 3. Run the Flask Application
```bash
python app.py
```

### 4. Open in Browser
Navigate to `http://localhost:5000` to use the application.

---

## 📦 Model Information

### What happens on first run:
- **Traditional ML models** (`.pkl` files) are included in the repository and load instantly
- **DistilBERT model** automatically downloads from Hugging Face Hub and caches locally
- **No additional setup required** - everything works out of the box!

🔗 **Model Repository**: [Barhomy/distilbert-fake-news-model](https://huggingface.co/Barhomy/distilbert-fake-news-model)

### Available Models

| Model | Type | Accuracy | Speed | Notes |
|-------|------|----------|-------|-------|
| **Logistic Regression + TF-IDF** | Traditional ML | Good | ⚡ Fast | Reliable baseline |
| **Naive Bayes + TF-IDF** | Traditional ML | Good | ⚡ Fast | High recall rate |
| **DistilBERT Transformer** | Deep Learning | ⭐ Best | 🐌 Slower | Default model |

---

## 🏗️ Project Structure

```
fake-news-detector/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── README.md                      # Project documentation
├── .gitignore                     # Git ignore file
├── Fake_news.code-workspace       # VS Code workspace
│
├── models/                        # Pre-trained ML models (included)
│   ├── logistic_regression_model.pkl
│   ├── naive_bayes_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── distilbert-fake-news-model/    # Auto-downloaded (ignored by git)
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   └── vocab.txt
│
├── templates/                     # HTML templates
│   └── index.html
│
└── static/                        # Static files
    ├── css/
    │   └── style.css
    └── images/
        └── [image files]
```

---

## 🎯 How It Works

1. **Text Input**: User enters news text or article content
2. **Model Selection**: Choose from 3 available AI models
3. **Preprocessing**: 
   - Traditional models use TF-IDF vectorization
   - DistilBERT uses transformer tokenization
4. **Classification**: Returns prediction with confidence scores
5. **Results Display**: Shows FAKE/TRUE with confidence percentage

---

## 🔧 API Usage

### Programmatic Access
```python
import requests

# Send POST request to classify text
response = requests.post('http://localhost:5000/predict', 
                        data={
                            'news_text': 'Your news text here',
                            'model_choice': 'distilbert'
                        })
result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']}")
```

### Response Format
```json
{
    "prediction": "FAKE",
    "confidence": "87.50%"
}
```

---

## 📊 Model Performance

The fine-tuned DistilBERT model achieves:
- **Accuracy**: ~94% on test dataset
- **Precision**: ~93% for fake news detection
- **Recall**: ~95% for fake news detection
- **F1-Score**: ~94% overall performance

---

## 🛠️ Development

### Project Setup for Development
```bash
# Clone and setup
git clone https://github.com/Barhomy/fake-news-detector.git
cd fake-news-detector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run in development mode
python app.py
```

### File Structure Notes
- **`.pkl` models are included** in the repository for immediate functionality
- **DistilBERT model downloads automatically** on first run
- **No additional model setup required**

---

## 🚀 Deployment

### Local Production
```bash
# Install production server
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

```bash
# Build and run
docker build -t fake-news-detector .
docker run -p 5000:5000 fake-news-detector
```

---

## 🔒 Model Files & Git

- **Traditional ML models** (`.pkl` files) are included in the repository
- **DistilBERT model** downloads automatically from Hugging Face Hub
- **No Git LFS required** - all essential files are under GitHub's size limits
- **Clone and run** - no additional downloads needed for basic functionality

---

## 📈 Future Enhancements

- [ ] Add support for multiple languages
- [ ] Implement real-time news scraping and fact-checking
- [ ] Add LIME/SHAP explanations for model decisions
- [ ] Mobile app development
- [ ] API rate limiting and authentication
- [ ] Ensemble voting system combining all models
- [ ] User feedback system for model improvement

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation for new functionality
- Ensure models work with the existing API structure

---

## 🐛 Troubleshooting

### Common Issues:

**Model loading errors:**
```bash
# Make sure all dependencies are installed
pip install -r requirements.txt

# Check if model files exist
ls models/  # Should show .pkl files
```

**DistilBERT download issues:**
- Ensure stable internet connection
- The model downloads automatically on first use
- Download size: ~250MB

**CUDA/GPU errors:**
- The app works with CPU-only PyTorch
- GPU acceleration is optional for better performance

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Hugging Face** for model hosting and transformers library
- **Scikit-learn** for traditional ML algorithms
- **Flask** for the lightweight web framework
- **PyTorch** for deep learning capabilities
- **Open source community** for the tools and libraries

---

## 📞 Contact

**Author**: [@Barhomy](https://github.com/Barhomy)

**Model**: [Barhomy/distilbert-fake-news-model](https://huggingface.co/Barhomy/distilbert-fake-news-model)

**Issues**: [GitHub Issues](https://github.com/Barhomy/fake-news-detector/issues)

**Email**: [Contact via GitHub](https://github.com/Barhomy)

---

## ⭐ Star this repository if you find it helpful!

---

*Built with ❤️ for combating misinformation through AI*
