import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Data
df = pd.read_csv('../data/destinasi-wisata-indonesia.csv')
print(df.head)

# Fitur yg digunakan
selected_features = ['Description','Category','City']
for feature in selected_features:
   df[feature] = df[feature].fillna('')

# Vectorization
combined_features = df['Description']+ ' '+ df['Category']+ '' + df['City']
vectorizer = TfidfVectorizer()

feature_vectors = vectorizer.fit_transform(combined_features)
print(feature_vectors)

# Similarity
similarity = cosine_similarity(feature_vectors, feature_vectors)
print (similarity)