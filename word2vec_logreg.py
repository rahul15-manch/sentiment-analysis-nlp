# word2vec_logreg.py

import pickle
import numpy as np
import gensim
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from utils import clean_text

def train_word2vec(texts, vector_size=100, window=5, min_count=2, sg=1):
    """
    Train Word2Vec embeddings on the provided texts.
    """
    # Clean & tokenize texts
    tokenized_texts = [clean_text(t).split() for t in texts]

    # Train Word2Vec
    model = gensim.models.Word2Vec(
        sentences=tokenized_texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        workers=4,
        epochs=10
    )

    return model

def texts_to_vectors(texts, w2v_model):
    """
    Convert texts to averaged Word2Vec vectors.
    """
    vectors = []
    for t in texts:
        tokens = clean_text(t).split()
        word_vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
        if len(word_vectors) > 0:
            vectors.append(np.mean(word_vectors, axis=0))
        else:
            # fallback: zero vector if no words are in vocab
            vectors.append(np.zeros(w2v_model.vector_size))
    return np.array(vectors)

def prepare_word2vec_data(texts, labels, test_size=0.2, random_state=42):
    """
    Prepare train/test split for Word2Vec + Logistic Regression.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def train_logreg(X_train_texts, y_train, X_test_texts, y_test, w2v_model, model_path="models/word2vec_logreg.pkl"):
    """
    Train Logistic Regression classifier using Word2Vec embeddings.
    """
    # Convert texts to vectors
    X_train_vec = texts_to_vectors(X_train_texts, w2v_model)
    X_test_vec = texts_to_vectors(X_test_texts, w2v_model)

    # Train Logistic Regression
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_vec, y_train)

    # Evaluate
    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"[INFO] Word2Vec + Logistic Regression Test Accuracy: {acc:.4f}\n")
    print("[INFO] Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save model + Word2Vec embeddings
    with open(model_path, "wb") as f:
        pickle.dump((clf, w2v_model), f)

    print(f"[INFO] Logistic Regression model + Word2Vec embeddings saved at {model_path}")
    return clf, w2v_model
