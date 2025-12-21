import streamlit as st
import pandas as pd
from models.model import load_model

st.set_page_config(
    page_title="Evaluasi Top-K Recommender",
    layout="centered"
)

@st.cache_resource
def get_model():
    return load_model()

df, similarity = get_model()

st.title("📊 Evaluasi Top-K Sistem Rekomendasi Wisata")
st.write(
    """
    Evaluasi dilakukan menggunakan **Top-K Recommendation Evaluation**
    dengan asumsi relevansi berdasarkan **kategori atau kota yang sama**.
    """
)

k = st.selectbox("Pilih nilai K", options=[3, 5, 10], index=1)

sample_size = 80

def evaluate_top_k(df, similarity, k, sample_size):
    precisions = []
    recalls = []

    test_indices = df.sample(sample_size, random_state=42).index.tolist()

    for idx in test_indices:
        query_item = df.iloc[idx]

        scores = list(enumerate(similarity[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:k+1]

        recommended_indices = [i for i, _ in scores]

        relevant_items = df[
            (df['Category'] == query_item['Category']) |
            (df['City'] == query_item['City'])
        ].index.tolist()

        if idx in relevant_items:
            relevant_items.remove(idx)

        if not relevant_items:
            continue

        hits = len(set(recommended_indices) & set(relevant_items))

        precision = hits / k
        recall = hits / len(relevant_items)

        precisions.append(precision)
        recalls.append(recall)

    return (
        round(sum(precisions) / len(precisions), 4),
        round(sum(recalls) / len(recalls), 4),
        len(precisions)
    )

if st.button("▶️ Jalankan Evaluasi"):
    with st.spinner("Melakukan evaluasi Top-K..."):
        mean_precision, mean_recall, valid_samples = evaluate_top_k(
            df, similarity, k, sample_size
        )

    st.success("Evaluasi selesai")

    col1, col2, col3 = st.columns(3)
    col1.metric("Precision@K", mean_precision)
    col2.metric("Recall@K", mean_recall)
    col3.metric("Data train", valid_samples)

st.markdown(
    """
    ### 🧠 Interpretasi
    - **Precision@K** menunjukkan seberapa relevan rekomendasi yang diberikan.
    - **Recall@K** menunjukkan seberapa banyak item relevan yang berhasil ditemukan.
    - Nilai tidak harus mendekati 1 karena dataset **tidak memiliki label eksplisit**.
    """
)
