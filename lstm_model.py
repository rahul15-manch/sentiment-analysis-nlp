# lstm_model.py

import os
import numpy as np
import joblib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils import clean_text

# ---------------------------
# 🔹 LSTM Model Parameters
# ---------------------------
MAX_VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 100
BATCH_SIZE = 64
EPOCHS = 5  # increase for real training

# ---------------------------
# 🔹 Prepare Data
# ---------------------------
def prepare_data(texts, labels, test_size=0.2, random_state=42):
    # Clean texts
    cleaned_texts = [clean_text(t) for t in texts]
    
    # Tokenizer
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(cleaned_texts)
    sequences = tokenizer.texts_to_sequences(cleaned_texts)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    
    # Encode labels
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences, encoded_labels, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, tokenizer, le

# ---------------------------
# 🔹 Build LSTM Model
# ---------------------------
def build_lstm_model(vocab_size=MAX_VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length))
    model.add(Bidirectional(LSTM(64, return_sequences=False)))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # binary classification
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# ---------------------------
# 🔹 Train Model
# ---------------------------
def train_lstm_model(model, X_train, y_train, X_val, y_val, batch_size=BATCH_SIZE, epochs=EPOCHS):
    es = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[es]
    )
    return history

# ---------------------------
# 🔹 Save/Load LSTM Model
# ---------------------------
def save_lstm_model(model, tokenizer, label_encoder, model_path="models/lstm_model.h5", tok_path="models/lstm_tokenizer.pkl", le_path="models/lstm_labelencoder.pkl"):
    model.save(model_path)
    joblib.dump(tokenizer, tok_path)
    joblib.dump(label_encoder, le_path)
    print(f"[INFO] Model saved to {model_path}, tokenizer to {tok_path}, label encoder to {le_path}")


def load_lstm_model(model_path="models/lstm_model.h5", tok_path="models/lstm_tokenizer.pkl", le_path="models/lstm_labelencoder.pkl"):
    model = load_model(model_path)
    tokenizer = joblib.load(tok_path)
    label_encoder = joblib.load(le_path)
    return model, tokenizer, label_encoder

# ---------------------------
# 🔹 Predict Single Text
# ---------------------------
def predict_sentiment_lstm(text, model, tokenizer, label_encoder):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    pred = model.predict(padded)[0][0]
    return label_encoder.inverse_transform([int(round(pred))])[0]
