import os
import sys 
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lstm_model import (
    prepare_data,
    build_lstm_model,
    train_lstm_model,
    save_lstm_model,
    load_lstm_model,
    predict_sentiment_lstm
)

# ---------------------------
# 🔹 Tiny Toy Dataset
# ---------------------------
texts = [
    "I love this movie",
    "This film was terrible",
    "What a fantastic performance",
    "Worst acting ever"
]
labels = ["positive", "negative", "positive", "negative"]

# ---------------------------
# 🔹 Test LSTM Pipeline
# ---------------------------
def test_lstm_pipeline():
    # Prepare data
    X_train, X_test, y_train, y_test, tokenizer, le = prepare_data(texts, labels, test_size=0.5)
    
    # Build model
    model = build_lstm_model()
    
    # Train briefly
    history = train_lstm_model(model, X_train, y_train, X_test, y_test, epochs=1, batch_size=2)
    assert history is not None

    # Save model
    os.makedirs("models", exist_ok=True)
    save_lstm_model(model, tokenizer, le, model_path="models/test_lstm.h5",
                    tok_path="models/test_tokenizer.pkl", le_path="models/test_labelencoder.pkl")

    # Load model
    loaded_model, loaded_tok, loaded_le = load_lstm_model("models/test_lstm.h5",
                                                          "models/test_tokenizer.pkl",
                                                          "models/test_labelencoder.pkl")
    assert loaded_model is not None
    assert loaded_tok is not None
    assert loaded_le is not None

    # Predict
    pred = predict_sentiment_lstm("I really enjoyed this movie", loaded_model, loaded_tok, loaded_le)
    assert pred in ["positive", "negative"]

    # Clean up
    os.remove("models/test_lstm.h5")
    os.remove("models/test_tokenizer.pkl")
    os.remove("models/test_labelencoder.pkl")
