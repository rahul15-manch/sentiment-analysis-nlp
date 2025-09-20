# utils.py
import re
import string
import joblib
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

# Ensure required NLTK resources are downloaded
nltk.download("stopwords", quiet=True)

STOPWORDS = set(stopwords.words("english"))

# ---------------------------
# 🔹 Text Cleaning Function
# ---------------------------
def clean_text(text):
    """
    Cleans input text by:
    - Lowercasing
    - Removing HTML tags
    - Removing punctuation/numbers/special characters
    - Removing stopwords
    Returns cleaned string.
    """
    # Lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # Remove punctuation & numbers
    text = re.sub(r"[^a-z\s]", "", text)
    
    # Remove extra whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    
    # Remove stopwords
    tokens = [word for word in text.split() if word not in STOPWORDS]
    
    return " ".join(tokens)


# ---------------------------
# 🔹 Tokenizer
# ---------------------------
def tokenize(text):
    """
    Splits text into tokens after cleaning.
    Returns list of words.
    """
    text = clean_text(text)
    return text.split()


# ---------------------------
# 🔹 Model Persistence
# ---------------------------
def save_model(model, vectorizer, model_path="models/sentiment_model.pkl", vec_path="models/vectorizer.pkl"):
    """
    Save model and vectorizer using joblib.
    """
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)
    print(f"[INFO] Model saved to {model_path}, Vectorizer saved to {vec_path}")


def load_model(model_path="models/sentiment_model.pkl", vec_path="models/vectorizer.pkl"):
    """
    Load model and vectorizer from disk.
    """
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer


# ---------------------------
# 🔹 Prediction Pipeline
# ---------------------------
def predict_sentiment(text, model, vectorizer):
    """
    Predict sentiment of a single text string.
    Returns 'positive' or 'negative'.
    """
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    return prediction
