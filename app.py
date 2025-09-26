# app.py
import streamlit as st
import pickle
import numpy as np
from utils import clean_text  # Keep for any further preprocessing

st.set_page_config(page_title="Sentiment Analysis", layout="wide")

# ------------------------------
# Helper function to load model
# ------------------------------
@st.cache_resource
def load_tfidf_xgb_model():
    model_path = "models/tfidf_xgb_v2.pkl"  # updated model
    with open(model_path, "rb") as f:
        model, vectorizer = pickle.load(f)
    return model, vectorizer

# ------------------------------
# Negation-aware preprocessing
# ------------------------------
negations = ["not", "no", "never", "n't"]

def clean_text_with_negation(text):
    text = text.lower()
    tokens = text.split()
    new_tokens = []
    skip_next = False
    for i, token in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        if token in negations and i + 1 < len(tokens):
            new_tokens.append("not_" + tokens[i + 1])
            skip_next = True
        else:
            new_tokens.append(token)
    text = " ".join(new_tokens)
    # remove any unwanted characters
    text = "".join([c if c.isalnum() or c=="_" or c==" " else "" for c in text])
    return text

# ------------------------------
# Prediction function
# ------------------------------
def predict_tfidf_xgb(text):
    model, vectorizer = load_tfidf_xgb_model()
    text_clean = clean_text_with_negation(text)
    vect = vectorizer.transform([text_clean])
    pred = model.predict(vect)[0]
    proba = np.max(model.predict_proba(vect))
    return pred, float(proba)

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("📊 Sentiment Analysis (TF-IDF + XGBoost)")

review = st.text_area("Enter your text for sentiment analysis:", height=150)

if review:
    label, prob = predict_tfidf_xgb(review)
    sentiment = "positive" if label == 1 else "negative"
    st.markdown("### Prediction")
    st.write(f"**Sentiment:** {sentiment}")
    st.write(f"**Confidence:** {prob:.2f}")
