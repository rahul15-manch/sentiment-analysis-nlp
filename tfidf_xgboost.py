import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from utils import clean_text


def prepare_tfidf_data(texts, labels, max_features=20000, test_size=0.2, random_state=42):
    """
    Convert raw text into TF-IDF features for training XGBoost.
    """
    # Clean the texts
    texts_cleaned = [clean_text(t) for t in texts]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        texts_cleaned, labels, test_size=test_size, random_state=random_state
    )

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer


def train_xgboost(X_train, y_train, X_test, y_test, vectorizer, model_path="models/tfidf_xgb.pkl"):
    """
    Train XGBoost classifier with TF-IDF features.
    """
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    print(f"[INFO] XGBoost Test Accuracy: {acc:.4f}\n")
    print("[INFO] Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save model + vectorizer
    with open(model_path, "wb") as f:
        pickle.dump((model, vectorizer), f)

    print(f"[INFO] XGBoost model + TF-IDF vectorizer saved at {model_path}")
    return model, vectorizer
