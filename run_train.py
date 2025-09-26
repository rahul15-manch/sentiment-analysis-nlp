import pandas as pd
from tfidf_xgboost import prepare_tfidf_data, train_xgboost  # your training functions

def main():
    dataset_path = "data/IMDB Dataset.csv"
    print(f"[INFO] Loading dataset from {dataset_path}...")
    data = pd.read_csv(dataset_path)

    # Rename / map columns
    if "review" in data.columns and "sentiment" in data.columns:
        texts = data["review"].astype(str).tolist()
        labels = data["sentiment"].map({"positive": 1, "negative": 0}).tolist()
    else:
        raise ValueError("Dataset must have 'review' and 'sentiment' columns.")

    # Prepare TF-IDF features
    print("[INFO] Preparing TF-IDF features (with bigrams)...")
    X_train, X_test, y_train, y_test, vectorizer = prepare_tfidf_data(
        texts, labels, max_features=20000, ngram_range=(1, 2)
    )

    # Train + evaluate XGBoost
    print("[INFO] Training XGBoost...")
    model, vectorizer = train_xgboost(
        X_train, y_train, X_test, y_test, vectorizer,
        model_path="models/tfidf_xgb_v2.pkl"
    )

if __name__ == "__main__":
    main()
