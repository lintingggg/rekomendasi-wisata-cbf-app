import pandas as pd
import difflib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_model():
    df = pd.read_csv('data/destinasi-wisata-preprocessed.csv')

    # Kolom wajib
    required_cols = ['Place_Name', 'Category', 'City', 'clean_text']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan di dataset.")

    df[required_cols] = df[required_cols].fillna('')

    # TF-IDF langsung dari teks bersih
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=2
    )

    vectors = vectorizer.fit_transform(df['clean_text'])
    similarity = cosine_similarity(vectors)

    return df, similarity


def get_places_by_category(model, category_keyword):
    df, _ = model
    category_keyword = category_keyword.lower()

    filtered = df[
        df['Place_Name'].str.lower().str.contains(category_keyword) |
        df['Category'].str.lower().str.contains(category_keyword)
    ]

    return sorted(filtered['Place_Name'].unique().tolist())


def recommend(place_name, model, n=5):
    df, similarity = model

    place_name = place_name.strip()
    names = df['Place_Name'].tolist()

    match = difflib.get_close_matches(place_name, names, n=1, cutoff=0.6)
    if not match:
        return []

    idx = df[df['Place_Name'] == match[0]].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]

    return [
        {
            "name": df.iloc[i]['Place_Name'],
            "category": df.iloc[i]['Category'],
            "city": df.iloc[i]['City'],
            "score": round(float(score), 3)
        }
        for i, score in scores
    ]
