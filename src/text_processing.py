"""Nettoyage des textes d'avis."""

import html
import re

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

import config

_HTML_TAG = re.compile(r"<[^>]+>")
_NON_WORD = re.compile(r"[\W_]+", flags=re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def clean_text(text):
    """Normalise un texte pour sa vectorisation : HTML, minuscules et ponctuation."""
    if text is None:
        return ""

    text = html.unescape(str(text))
    text = _HTML_TAG.sub(" ", text)
    text = _NON_WORD.sub(" ", text.lower())
    return _WHITESPACE.sub(" ", text).strip()

def build_tfidf_pipeline(max_features=2000, n_components=30):
    """Construit un pipeline TF-IDF + réduction de dimension SVD."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.95,
            sublinear_tf=True,
        )),
        ("svd", TruncatedSVD(
            n_components=n_components,
            random_state=config.RANDOM_STATE,
        )),
    ])


def compute_text_features(df):
    """
    Extrait les caractéristiques stylistiques et structurelles sur le texte BRUT
    (avant mise en minuscules et suppression de la ponctuation).
    """
    texts = df["Text"].fillna("").astype(str)
    summaries = df["Summary"].fillna("").astype(str) if "Summary" in df.columns else pd.Series("", index=df.index)

    features = pd.DataFrame(index=df.index)

    # 1. Longueur et structure
    features["char_count"] = texts.str.len()
    features["word_count"] = texts.str.split().str.len()
    features["avg_word_len"] = features["char_count"] / (features["word_count"] + 1)
    features["sentence_count"] = texts.str.count(r"[.!?]+").clip(lower=1)
    features["avg_sentence_len"] = features["word_count"] / features["sentence_count"]
    features["summary_len"] = summaries.str.len()

    # 2. Signaux stylistiques et d'emphase (majuscules, ponctuation expressive)
    upper_counts = texts.map(lambda s: sum(1 for c in s if c.isupper()))
    features["upper_ratio"] = upper_counts / (features["char_count"] + 1)
    features["exclamation_count"] = texts.str.count("!")
    features["question_count"] = texts.str.count(r"\?")
    features["punctuation_density"] = (features["exclamation_count"] + features["question_count"]) / (features["word_count"] + 1)

    # 3. Richesse lexicale (ratio de mots uniques)
    features["unique_word_ratio"] = texts.map(
        lambda s: len(set(s.lower().split())) / (len(s.split()) + 1)
    )

    return features


def compute_readability(text_series):
    """Calcule un indice de lisibilité (approximation Flesch Reading Ease)."""
    def _flesch_score(text):
        if not isinstance(text, str) or not text.strip():
            return 0.0
        words = re.findall(r"[a-zA-Z]+", text)
        n_words = len(words)
        if n_words == 0:
            return 0.0
        n_sentences = max(1, len(re.findall(r"[.!?]+", text)))
        # Estimation du nombre de syllabes (groupes de voyelles)
        n_syllables = sum(max(1, len(re.findall(r"[aeiouyAEIOUY]+", w))) for w in words)
        score = 206.835 - 1.015 * (n_words / n_sentences) - 84.6 * (n_syllables / n_words)
        return float(np.clip(score, -50.0, 120.0))

    return text_series.fillna("").map(_flesch_score)