import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# =========================
# INIT SASTRAWI
# =========================
stemmer = StemmerFactory().create_stemmer()
stopword_remover = StopWordRemoverFactory().create_stop_word_remover()


# =========================
# Case Folding
# =========================
def case_folding(text: str) -> str:
    return text.lower()


# =========================
# Tokenizing
# =========================
def tokenizing(text: str) -> list:
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


# =========================
# Stopword Removal (Sastrawi)
# =========================
def stopword_removal(tokens: list) -> list:
    cleaned_text = " ".join(tokens)
    cleaned_text = stopword_remover.remove(cleaned_text)
    return cleaned_text.split()


# =========================
# Stemming (Sastrawi)
# =========================
def stemming(tokens: list) -> list:
    return [stemmer.stem(word) for word in tokens]


# =========================
# Full Preprocessing Pipeline
# =========================
def preprocess_text(text: str) -> str:
    cf = case_folding(text)
    tk = tokenizing(cf)
    sw = stopword_removal(tk)
    st = stemming(sw)
    return " ".join(st)

def preprocess_text_steps(text: str) -> dict:
    cf = case_folding(text)
    tk = tokenizing(cf)
    sw = stopword_removal(tk)
    st = stemming(sw)

    return {
        "original": text,
        "case_folding": cf,
        "tokenizing": tk,
        "stopword_removal": sw,
        "stemming": st,
        "clean_text": " ".join(st)
    }



# =========================
# Gabungkan fitur teks
# =========================
def combine_text_features(df: pd.DataFrame, columns: list) -> pd.Series:
    return (
        df[columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
    )


# =========================
# Preprocess Dataset & Save
# =========================
def preprocess_and_save(
    input_path: str,
    output_path: str,
    text_columns: list
) -> pd.DataFrame:
    """
    Melakukan preprocessing dataset wisata dan menyimpan hasilnya ke file CSV.
    """

    df = pd.read_csv(input_path)

    # Gabungkan fitur teks
    df["combined_text"] = combine_text_features(df, text_columns)

    # Preprocessing teks
    df["clean_text"] = df["combined_text"].apply(preprocess_text)

    # Simpan hasil preprocessing
    df.to_csv(output_path, index=False)

    return df
