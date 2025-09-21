import pandas as pd
import pytest
from tfidf_xgboost import prepare_tfidf_data, train_xgboost

@pytest.fixture(scope="module")
def imdb_data():
    """Load IMDB dataset once for all tests."""
    df = pd.read_csv("/Users/rahulmanchanda/Desktop/sentimen-analysis-nlp/data/IMDB Dataset.csv")
    texts = df["review"].values
    labels = df["sentiment"].map({"negative": 0, "positive": 1}).values
    return texts, labels

def test_tfidf_xgboost_pipeline(imdb_data):
    texts, labels = imdb_data

    # Prepare data
    X_train, X_test, y_train, y_test, vectorizer = prepare_tfidf_data(texts, labels)

    # Train + evaluate
    model, vec = train_xgboost(X_train, y_train, X_test, y_test, vectorizer)

    # Check accuracy threshold
    sample_acc = model.score(X_test, y_test)
    assert sample_acc > 0.80, f"Accuracy too low: {sample_acc}"

def test_single_prediction(imdb_data):
    texts, labels = imdb_data
    X_train, X_test, y_train, y_test, vectorizer = prepare_tfidf_data(texts, labels)
    model, vec = train_xgboost(X_train, y_train, X_test, y_test, vectorizer)

    # Prediction sanity check
    sample = ["The movie was absolutely wonderful, I loved it!"]
    sample_vec = vec.transform(sample)
    pred = model.predict(sample_vec)[0]

    assert pred in [0, 1], f"Unexpected prediction: {pred}"

