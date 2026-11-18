"""Chargement et nettoyage initial des avis Amazon."""

import html
import re
from pathlib import Path

import pandas as pd

import config


def load_raw_reviews(path):
    """Charge le CSV brut et ajoute les informations temporelles."""
    path = Path(path)
    df = pd.read_csv(path)

    df["DateTime"] = pd.to_datetime(df["Time"], unit="s", errors="coerce")
    df["Year"] = df["DateTime"].dt.year
    df["Month"] = df["DateTime"].dt.month
    return df


def detect_duplicates(df):
    """Repere les doublons du meme utilisateur, meme timestamp et meme texte."""
    return df.duplicated(subset=["UserId", "Time", "Text"], keep="first")


def fix_helpfulness_inconsistencies(df):
    """Corrige les nombres de votes positifs superieurs au nombre total de votes."""
    df = df.copy()
    invalid = df["HelpfulnessNumerator"] > df["HelpfulnessDenominator"]
    df.loc[invalid, "HelpfulnessNumerator"] = df.loc[invalid, "HelpfulnessDenominator"]
    return df, int(invalid.sum())


def clean_text_html(df, columns=("Summary", "Text")):
    """Supprime les balises HTML et decode les entites HTML."""
    df = df.copy()
    pattern = re.compile(r"<[^>]+>")

    for column in columns:
        if column in df.columns:
            df[column] = (
                df[column]
                .fillna("")
                .astype(str)
                .map(lambda value: html.unescape(pattern.sub(" ", value)))
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )
    return df


def compute_helpfulness_ratio(df):
    """Calcule le ratio d'utilite et la cible binaire."""
    df = df.copy()
    numerator = df["HelpfulnessNumerator"].astype(float)
    denominator = df["HelpfulnessDenominator"].astype(float)

    df["HasVotes"] = denominator > 0
    df["HelpfulnessRatio"] = numerator.div(denominator).where(denominator > 0)
    df["IsHelpful"] = (
        (denominator >= config.MIN_VOTES)
        & (df["HelpfulnessRatio"] >= config.HELPFULNESS_THRESHOLD)
    ).astype(int)
    return df


def add_sentiment_labels(df):
    """Ajoute une classe sentiment a partir de la note."""
    df = df.copy()

    def sentiment(score):
        if score <= 2:
            return "negatif"
        if score == 3:
            return "neutre"
        return "positif"

    df["SentimentClass"] = df["Score"].map(sentiment)
    return df


def temporal_split(df, train_end_year=None, test_start_year=None):
    """Separe les donnees par annees sans melanger train et test."""
    train_end_year = train_end_year or config.TRAIN_END_YEAR
    test_start_year = test_start_year or config.TEST_START_YEAR
    train = df[df["Year"] <= train_end_year].copy()
    test = df[df["Year"] >= test_start_year].copy()
    return train, test


def stratified_sample(df, n, stratify_col, random_state=42):
    """Tire un echantillon avec proportions conservees."""
    n = min(n, len(df))
    if n <= 0:
        return df.iloc[0:0].copy()

    counts = df[stratify_col].value_counts()
    if len(counts) < 2:
        return df.sample(n=n, random_state=random_state)

    proportions = df[stratify_col].value_counts(normalize=True)
    exact = proportions * n
    allocation = exact.astype(int)
    remaining = n - int(allocation.sum())
    remainders = (exact - allocation).sort_values(ascending=False)
    for category in remainders.index[:remaining]:
        allocation.loc[category] += 1
    allocation = allocation.where(allocation <= counts, counts).astype(int)

    pieces = []
    for category, category_count in allocation.items():
        if category_count > 0:
            piece = df[df[stratify_col] == category].sample(
                n=category_count,
                random_state=random_state,
            )
            pieces.append(piece)
    return pd.concat(pieces).sample(frac=1, random_state=random_state).reset_index(drop=True)


def run_full_pipeline(path=None):
    """Execute le chargement, l'audit et le nettoyage complets du module 0."""
    path = path or config.RAW_REVIEWS
    df_raw = load_raw_reviews(path)

    duplicates = detect_duplicates(df_raw)
    df = df_raw.loc[~duplicates].copy()

    df, _ = fix_helpfulness_inconsistencies(df)
    df = clean_text_html(df)
    df = compute_helpfulness_ratio(df)
    df = add_sentiment_labels(df)

    train, test = temporal_split(df)
    if len(train) == 0 or len(test) == 0:
        raise ValueError("Le jeu nettoye ne contient pas assez d'années pour le train/test.")

    output_path = config.DATA_PROCESSED / "reviews_clean.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
