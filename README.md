# 📊 Multi-Model Sentiment Analysis

## 📝 Project Overview
This project implements a **multi-model sentiment analysis system** capable of predicting sentiment from text.  
The **TF-IDF + XGBoost model** is selected as the primary model for deployment due to its **lightweight, fast, and reliable predictions**.

- **Dataset**: IMDB movie reviews  
- **Labels**: `positive` / `negative`  
- **TF-IDF features**: unigrams + bigrams  
- **XGBoost classifier**: 300 estimators, max_depth=6  

---

## ⚡ Model Performance

### 1️⃣ Using CountVectorizer

| Model                  | Accuracy | Precision | Recall | F1-score |
|------------------------|----------|-----------|--------|----------|
| Logistic Regression    | 86.99%   | 0.87      | 0.87   | 0.87     |
| Naive Bayes            | 84.55%   | 0.85      | 0.85   | 0.85     |
| SVM                    | 86.14%   | 0.86      | 0.86   | 0.86     |

### 2️⃣ Using TF-IDF Vectorizer

| Model                  | Accuracy | Precision | Recall | F1-score |
|------------------------|----------|-----------|--------|----------|
| Logistic Regression    | 88.66%   | 0.89      | 0.89   | 0.89     |
| Naive Bayes            | 84.92%   | 0.85      | 0.85   | 0.85     |
| SVM                    | 88.06%   | 0.88      | 0.88   | 0.88     |

> ✅ **Observation:** Logistic Regression with TF-IDF features performs best (~88.66% accuracy). TF-IDF generally boosts performance compared to CountVectorizer.

---

## 🧠 Models in the Project

| Model                              | File                                                    | Input Representation                  | Output                              | Notes                                                                    |
| ---------------------------------- | ------------------------------------------------------- | ------------------------------------- | ----------------------------------- | ------------------------------------------------------------------------ |
| **TF-IDF + XGBoost**               | `tfidf_xgboost.py`, `models/tfidf_xgb_v2.pkl`           | TF-IDF (unigrams + bigrams)           | Binary (0 = negative, 1 = positive) | Fast, lightweight, good baseline, deployed in `app.py`                   |
| **LSTM (Deep Learning)**           | `lstm_model.py`, `models/lstm_full_model.h5`            | Tokenized & padded sequences          | Softmax over classes                | Handles sequential info; heavier, requires TensorFlow/Keras              |
| **Word2Vec + Logistic Regression** | `word2vec_logreg.py`, `models/test_word2vec_logreg.pkl` | Word embeddings (average over tokens) | Binary                              | Captures semantic similarity; lightweight, embedding quality matters     |
| **Custom Sentiment Model**         | `models/sentiment_model.pkl`                            | Cleaned text (vectorized)             | Binary                              | Any sklearn-based model; easy to update/replace                          |

---

## 🔮 Example Predictions

| Sentence                                              | TF-IDF + XGBoost | LSTM  | Word2Vec + LR | Custom Model |
|------------------------------------------------------|-----------------|-------|---------------|--------------|
| I do not love movies but love webseries             | Positive (0.77) | 0.58  | 0.65          | 0.70         |
| The first half was good but second half was terrible| Negative (0.75) | 0.72  | 0.70          | 0.74         |
| Thought the movie would be terrible, but surprisingly good | Positive (0.68) | 0.60 | 0.62        | 0.65         |

> **Observation:** TF-IDF + XGBoost provides strong and consistent predictions. LSTM captures sequence dependencies but sometimes has lower confidence. Word2Vec + LR is lightweight and semantic-aware. Custom model can be tuned for project-specific needs.

---

## ⚖️ Pros & Cons of Each Model

| Model            | Pros                                              | Cons                                                       |
| ---------------- | ------------------------------------------------- | ---------------------------------------------------------- |
| TF-IDF + XGBoost | Fast, lightweight, high accuracy, simple          | Loses sequence info, struggles with negations              |
| LSTM             | Captures sequence info, handles long dependencies | Heavy, needs GPU for faster inference, slower in Streamlit |
| Word2Vec + LR    | Semantic info, lightweight, flexible              | Sensitive to embedding quality, averaging loses context    |
| Custom Model     | Easy to update, modular                           | Dependent on chosen features & vectorizer                  |

---

## 🗂 Project Structure
```
.
├── __init__.py
├── __pycache__
│   ├── __init__.cpython-310.pyc
│   ├── api_server.cpython-310.pyc
│   ├── api.cpython-310.pyc
│   ├── lstm_api.cpython-310.pyc
│   ├── lstm_model.cpython-310.pyc
│   ├── tfidf_xgboost.cpython-310.pyc
│   ├── train_tfidf_xgboost.cpython-310.pyc
│   ├── utils.cpython-310.pyc
│   └── word2vec_logreg.cpython-310.pyc
├── app.py
├── data
│   └── IMDB Dataset.csv
├── lstm_api.py
├── lstm_model.py
├── models
│   ├── lstm_full_labelencoder.pkl
│   ├── lstm_full_model.h5
│   ├── lstm_full_tokenizer.pkl
│   ├── sentiment_model.pkl
│   ├── test_word2vec_logreg.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── tfidf_xgb_v2.pkl
│   └── tfidf_xgb.pkl
├── notebooks
│   ├── experiments.ipynb
│   └── lstm_model.ipynb
├── README.md
├── requirements.txt
├── run_train.py
├── tests
│   ├── __pycache__
│   ├── test_lstm_model.py
│   ├── test_models.py
│   ├── test_tfidf_xgboost.py
│   ├── test_utils.py
│   └── test_word2vec_logreg.py
├── tfidf_xgboost.py
├── utils.py
├── venv
│   ├── bin
│   ├── etc
│   ├── include
│   ├── lib
│   ├── pyvenv.cfg
│   └── share
└── word2vec_logreg.py
```

---

## 🚀 Deployment Strategy

- **Single-process Streamlit**: Load only TF-IDF + XGBoost → avoids threading/mutex issues.  
- **API-based multimodal inference**:  
  - Run LSTM & Word2Vec models on FastAPI (`lstm_api.py`, `api_server.py`)  
  - Streamlit sends requests and displays predictions side by side  
  - Keeps UI responsive and avoids heavy models blocking main thread  

> **Recommendation:**  
> - **Production / single-process apps** → TF-IDF + XGBoost  
> - **Research / nuanced analysis** → Compare all models using the same sentences

---

## 🛠 Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![NLTK](https://img.shields.io/badge/NLTK-3.8-orange?logo=nltk)
![Pandas](https://img.shields.io/badge/Pandas-1.6-green?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-1.26-blue?logo=numpy)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2-lightgrey?logo=scikit-learn)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12-orange?logo=seaborn)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8-red?logo=matplotlib)
![BeautifulSoup](https://img.shields.io/badge/BS4-4.12-yellow?logo=beautifulsoup)
![PyTest](https://img.shields.io/badge/PyTest-7.4-lightblue?logo=pytest)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-2.14-red?logo=keras)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-lightgrey?logo=xgboost)
![Gensim](https://img.shields.io/badge/Gensim-4.3-lightblue?logo=gensim)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-orange?logo=streamlit)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-blue?logo=fastapi)
![Pydantic](https://img.shields.io/badge/Pydantic-2.3-lightgrey?logo=pydantic)

---

## 🏃‍♂️ Installation

```
git clone <your-repo-url>
cd sentimen-analysis-nlp

python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```
### 🏃‍♂️ Running the Streamlit App
```
streamlit run app.py
```
- Enter text in the UI to get sentiment predictions with confidence scores.

### 📈 Training New Model
```
python run_train.py
Load your dataset (e.g., data/IMDB Dataset.csv)
```
### Prepare TF-IDF features

- Train XGBoost classifier
- Save the model as models/tfidf_xgb_v2.pkl

🧪 Testing
```
pytest tests/
```
- Run test cases for all models.

## 🔮 Example Predictions

| Sentence                                              | Predicted Sentiment | Confidence |
|------------------------------------------------------|-------------------|------------|
| I do not love movies but love webseries             | Positive          | 0.77       |
| The first half was good but second half was terrible| Negative          | 0.75       |
| I thought movie was bad but it was not that bad     | Positive          | 0.58       |


## ⚡ Notes
- TF-IDF + XGBoost is chosen as the primary model for deployment due to its balance of speed and accuracy.

- Streamlit app works in single-process mode to avoid threading/mutex issues.

- Additional models (LSTM, Word2Vec + LR, custom) can be used for research and detailed comparison.


## 👤 Author

**Rahul Manchanda**  
## 👤 Author

**Rahul Manchanda**  
![Email](https://img.shields.io/badge/Email-rahulmanchanda015%40gmail.com-red?style=flat&logo=gmail&logoColor=white)  
![GitHub](https://img.shields.io/badge/GitHub-rahul15--manch-black?style=flat&logo=github&logoColor=white)  
![LinkedIn](https://img.shields.io/badge/LinkedIn-rahul--manchanda-blue?style=flat&logo=linkedin&logoColor=white)


