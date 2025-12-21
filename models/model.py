import pandas as pd
import difflib
import re
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

def load_model():
    df = pd.read_csv('data/destinasi-wisata-indonesia.csv')

    selected_features = ['Place_Name', 'Description', 'Category', 'City']
    for f in selected_features:
        df[f] = df[f].fillna('')

    stopword_factory = StopWordRemoverFactory()
    stopwords = set(stopword_factory.get_stop_words())
    stemmer = StemmerFactory().create_stemmer()

    def preprocess(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        tokens = text.split()
        tokens = [t for t in tokens if t not in stopwords]
        tokens = [stemmer.stem(t) for t in tokens]
        return ' '.join(tokens)

    df['combined_features'] = (
        df['Place_Name'] + ' ' +
        df['Description'] + ' ' +
        df['Category'] + ' ' +
        df['City']
    ).apply(preprocess)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=2
    )

    vectors = vectorizer.fit_transform(df['combined_features'])
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
