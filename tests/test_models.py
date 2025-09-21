# tests/test_model_comparison.py
import pytest
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.clean_text import clean_text

# ------------------------------
# Load models
# ------------------------------

@pytest.fixture(scope="module")
def lstm_model_fixture():
    model = load_model("models/lstm_full_model.h5")
    with open("models/lstm_full_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("models/lstm_full_labelencoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, tokenizer, le

@pytest.fixture(scope="module")
def tfidf_xgb_fixture():
    with open("models/tfidf_xgb.pkl", "rb") as f:
        model, vectorizer = pickle.load(f)
    return model, vectorizer

@pytest.fixture(scope="module")
def word2vec_lr_fixture():
    with open("models/test_word2vec_logreg.pkl", "rb") as f:
        model, w2v_model = pickle.load(f)
    return model, w2v_model

@pytest.fixture(scope="module")
def sentiment_model_fixture():
    with open("models/sentiment_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

# ------------------------------
# Prediction functions
# ------------------------------
def predict_lstm(text, lstm_tuple):
    model, tokenizer, le = lstm_tuple
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=200)
    pred = model.predict(padded)[0]
    label = le.inverse_transform([np.argmax(pred)])[0]
    return label, float(np.max(pred))

def predict_tfidf_xgb(text, xgb_tuple):
    model, vectorizer = xgb_tuple
    vect = vectorizer.transform([clean_text(text)])
    pred = model.predict(vect)[0]
    proba = np.max(model.predict_proba(vect))
    return pred, float(proba)

def predict_word2vec_lr(text, w2v_tuple):
    model, w2v_model = w2v_tuple
    tokens = clean_text(text).split()
    vec = np.mean([w2v_model.wv[t] for t in tokens if t in w2v_model.wv] or [np.zeros(w2v_model.vector_size)], axis=0).reshape(1, -1)
    pred = model.predict(vec)[0]
    proba = np.max(model.predict_proba(vec))
    return pred, float(proba)

def predict_sentiment_model(text, model):
    pred = model.predict([clean_text(text)])[0]
    return pred, 1.0  # placeholder probability

# ------------------------------
# Test
# ------------------------------
sample_texts = [
    "I love this product! It works perfectly.",
    "This is the worst experience I’ve ever had.",
    "It’s okay, nothing special."
]

def test_models_return_valid_prediction(lstm_model_fixture, tfidf_xgb_fixture, word2vec_lr_fixture, sentiment_model_fixture):
    for text in sample_texts:
        lstm_label, lstm_prob = predict_lstm(text, lstm_model_fixture)
        xgb_label, xgb_prob = predict_tfidf_xgb(text, tfidf_xgb_fixture)
        w2v_label, w2v_prob = predict_word2vec_lr(text, word2vec_lr_fixture)
        sent_label, sent_prob = predict_sentiment_model(text, sentiment_model_fixture)

        # Assert probabilities are within 0-1
        assert 0 <= lstm_prob <= 1
        assert 0 <= xgb_prob <= 1
        assert 0 <= w2v_prob <= 1
        assert 0 <= sent_prob <= 1

        # Assert labels are non-empty
        assert lstm_label is not None
        assert xgb_label is not None
        assert w2v_label is not None
        assert sent_label is not None
