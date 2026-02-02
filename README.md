# IMDB Sentiment Analysis (NLP)

Sentiment classification on IMDB movie reviews using **TF-IDF**, **Logistic Regression**, **LightGBM**, and **spaCy** preprocessing.

## Project Overview
This project builds a machine learning pipeline to automatically classify movie reviews as **positive (1)** or **negative (0)**.

**Primary metric:** F1-score  
**Result:** F1 ≈ 0.88–0.89 on the test set (meets the target threshold)

## Dataset
- File: `imdb_reviews.tsv` (not included in this repo)
- Columns used:
  - `review`: raw text
  - `pos`: target label (0/1)
  - `ds_part`: train/test indicator

## Methods
- Text normalization: lowercasing + basic cleanup
- Vectorization: TF-IDF (fit on train only → avoids leakage)
- Models:
  - DummyClassifier (baseline)
  - Logistic Regression (TF-IDF)
  - Logistic Regression (spaCy lemmatization + TF-IDF)
  - LightGBM (TF-IDF)

## How to run
```bash
pip install -r requirements.txt

