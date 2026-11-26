"""
Module 3 : Extraction d'aspects et sentiment par aspect.
==========================================================
Méthodes : dictionnaire, embeddings, zero-shot.
"""
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def split_sentences(text):
    """Découpe un texte en phrases."""
    if not isinstance(text, str):
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 5]


def detect_aspects_keywords(sentence, aspect_keywords=None):
    """
    Détecte les aspects dans une phrase par dictionnaire de mots-clés.
    Retourne une liste d'aspects détectés.
    """
    if aspect_keywords is None:
        aspect_keywords = config.ASPECT_KEYWORDS

    sentence_lower = sentence.lower()
    detected = []
    for aspect, keywords in aspect_keywords.items():
        if any(kw in sentence_lower for kw in keywords):
            detected.append(aspect)
    return detected


def detect_aspects_embeddings(sentence, aspect_keywords=None, model=None, threshold=0.4):
    """
    Détecte les aspects par similarité cosinus avec les embeddings.
    Plus flexible que le dictionnaire.
    """
    if aspect_keywords is None:
        aspect_keywords = config.ASPECT_KEYWORDS
    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(config.EMBEDDING_MODEL)

    sent_emb = model.encode([sentence], normalize_embeddings=True)[0]

    detected = []
    for aspect, keywords in aspect_keywords.items():
        kw_embs = model.encode(keywords, normalize_embeddings=True)
        kw_mean = kw_embs.mean(axis=0)
        kw_mean /= np.linalg.norm(kw_mean)

        similarity = np.dot(sent_emb, kw_mean)
        if similarity >= threshold:
            detected.append((aspect, float(similarity)))

    detected.sort(key=lambda x: x[1], reverse=True)
    return [(a, s) for a, s in detected]


def build_aspect_matrix(df, method="keywords"):
    """
    Construit la matrice avis × aspect × polarité.

    Parameters
    ----------
    df : DataFrame avec colonnes Text et SentimentClass
    method : "keywords" ou "embeddings"

    Returns
    -------
    aspect_data : DataFrame
    """
    aspect_data = []
    detect_fn = detect_aspects_keywords if method == "keywords" else detect_aspects_embeddings

    for idx, row in df.iterrows():
        sentences = split_sentences(row["Text"])
        for sent in sentences:
            aspects = detect_fn(sent)
            if method == "keywords":
                for aspect in aspects:
                    aspect_data.append({
                        "review_idx": idx,
                        "sentence": sent,
                        "aspect": aspect,
                        "review_sentiment": row.get("SentimentClass", "inconnu"),
                    })
            else:
                for aspect, score in aspects:
                    aspect_data.append({
                        "review_idx": idx,
                        "sentence": sent,
                        "aspect": aspect,
                        "similarity": score,
                        "review_sentiment": row.get("SentimentClass", "inconnu"),
                    })

    return pd.DataFrame(aspect_data)


def summarize_aspects(aspect_df, groupby="aspect"):
    """Résume les aspects par fréquence et sentiment."""
    if aspect_df.empty:
        return pd.DataFrame()

    summary = aspect_df.groupby(groupby).agg(
        count=("sentence", "count"),
        pct_negative=("review_sentiment", lambda x: (x == "negatif").mean()),
        pct_positive=("review_sentiment", lambda x: (x == "positif").mean()),
    ).sort_values("count", ascending=False)

    return summary
