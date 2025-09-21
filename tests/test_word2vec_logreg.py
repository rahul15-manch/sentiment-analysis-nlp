# tests/test_word2vec_logreg.py

import pytest
import pandas as pd
from word2vec_logreg import train_word2vec, prepare_word2vec_data, train_logreg, texts_to_vectors
from utils import clean_text

@pytest.fixture(scope="module")
def imdb_data():
    """Load IMDB dataset once for all tests."""
    df = pd.read_csv("/Users/rahulmanchanda/Desktop/sentimen-analysis-nlp/data/IMDB Dataset.csv")  # Make sure this file exists
    texts = df["review"].tolist()
    labels = df["sentiment"].tolist()
    return texts, labels

def test_word2vec_pipeline(imdb_data):
    texts, labels = imdb_data

    # Prepare train/test split
    X_train, X_test, y_train, y_test = prepare_word2vec_data(texts, labels)

    # Train Word2Vec
    w2v_model = train_word2vec(X_train)

    # Train Logistic Regression
    clf, _ = train_logreg(X_train, y_train, X_test, y_test, w2v_model, model_path="models/test_word2vec_logreg.pkl")

    # Check if model predicts
    X_test_vec = texts_to_vectors(X_test[:5], w2v_model)
    y_pred = clf.predict(X_test_vec)
    assert len(y_pred) == 5
    assert set(y_pred).issubset(set(y_test))

def test_single_prediction(imdb_data):
    texts, labels = imdb_data

    # Train Word2Vec
    w2v_model = train_word2vec(texts)

    # Train Logistic Regression
    clf, _ = train_logreg(texts, labels, texts, labels, w2v_model, model_path="models/test_word2vec_logreg.pkl")

    # Test single review
    single_review = "I absolutely loved this movie! Amazing performances."
    vec = texts_to_vectors([single_review], w2v_model)
    pred = clf.predict(vec)
    assert pred[0] in labels
