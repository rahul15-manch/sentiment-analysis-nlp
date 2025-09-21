# app.py
import streamlit as st
import pickle
import numpy as np
import re

st.set_page_config(page_title="Sentiment Analysis", layout="wide")

# ------------------------------
# Helper functions
# ------------------------------

@st.cache_resource
def load_tfidf_xgb_model():
    with open("models/tfidf_xgb.pkl", "rb") as f:
        model, vectorizer = pickle.load(f)
    return model, vectorizer

# ------------------------------
# Safe preprocessing
# ------------------------------

def clean_text_safe(text):
    """Lowercase and remove non-alphanumeric characters."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text

# ------------------------------
# Prediction function
# ------------------------------

def predict_tfidf_xgb(text):
    model, vectorizer = load_tfidf_xgb_model()
    text_clean = clean_text_safe(text)
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
