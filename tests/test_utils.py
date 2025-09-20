import os
import sys
import pytest

# 🔹 Add project root to sys.path so Python can find utils.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import clean_text, tokenize, save_model, load_model, predict_sentiment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


@pytest.fixture
def sample_data():
    texts = ["I love this movie", "This film was terrible"]
    labels = ["positive", "negative"]
    return texts, labels


def test_clean_text():
    text = "<b>This movie was AMAZING!!!</b> 🤩"
    cleaned = clean_text(text)
    assert isinstance(cleaned, str)
    assert "movie" in cleaned
    assert "<b>" not in cleaned  # HTML removed


def test_tokenize():
    tokens = tokenize("This is great")
    assert isinstance(tokens, list)
    assert all(isinstance(t, str) for t in tokens)
    assert "great" in tokens


def test_model_pipeline(sample_data):
    texts, labels = sample_data

    # Train a tiny model
    vectorizer = TfidfVectorizer(preprocessor=clean_text)
    X = vectorizer.fit_transform(texts)
    model = LogisticRegression().fit(X, labels)

    # Save & Load
    save_model(model, vectorizer, "models/test_model.pkl", "models/test_vectorizer.pkl")
    loaded_model, loaded_vec = load_model("models/test_model.pkl", "models/test_vectorizer.pkl")

    # Prediction should be valid (positive/negative)
    pred = predict_sentiment("What a fantastic performance!", loaded_model, loaded_vec)
    assert pred in ["positive", "negative"]

    # Clean up
    os.remove("models/test_model.pkl")
    os.remove("models/test_vectorizer.pkl")
