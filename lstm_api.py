# lstm_api.py
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI()

class TextRequest(BaseModel):
    text: str

# Load LSTM model, tokenizer, label encoder
model = load_model("models/lstm_full_model.h5")
with open("models/lstm_full_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("models/lstm_full_labelencoder.pkl", "rb") as f:
    le = pickle.load(f)

@app.post("/predict")
def predict_lstm(request: TextRequest):
    seq = tokenizer.texts_to_sequences([request.text])
    padded = pad_sequences(seq, maxlen=200)
    pred = model.predict(padded)[0]
    label = le.inverse_transform([np.argmax(pred)])[0]
    return {"model": "LSTM", "prediction": label, "confidence": float(np.max(pred))}


