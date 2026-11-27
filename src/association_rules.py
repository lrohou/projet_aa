"""Découverte non supervisée : extraction d'aspects négatifs et règles d'association (FP-Growth)."""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
import config


def extract_negative_aspect_transactions(df, aspect_keywords=None):
    """
    Transforme les avis ou la sortie du Module 3 en matrice booléenne (Avis x Aspects négatifs).
    """
    if aspect_keywords is None:
        aspect_keywords = config.ASPECT_KEYWORDS

    cols_lower = {c.lower(): c for c in df.columns}

    # Cas 1 : Format long issu du Module 3 (ex: review_id, aspect, sentiment / polarity / label)
    if "aspect" in cols_lower:
        asp_col = cols_lower["aspect"]
        sent_col = next(
            (cols_lower[k] for k in ["sentiment", "polarity", "aspect_sentiment", "label"] if k in cols_lower),
            None,
        )
        id_col = next(
            (cols_lower[k] for k in ["review_id", "id", "index"] if k in cols_lower),
            df.columns[0],
        )

        if sent_col is not None:
            # Filtre les mentions négatives (string "negatif"/"negative" ou score < 0)
            if pd.api.types.is_numeric_dtype(df[sent_col]):
                neg = df[df[sent_col] < 0].copy()
            else:
                neg = df[df[sent_col].astype(str).str.lower().str.startswith("neg")].copy()
        else:
            neg = df.copy()

        if neg.empty:
            return pd.DataFrame()

        basket = (
            neg.assign(val=1)
            .pivot_table(index=id_col, columns=asp_col, values="val", fill_value=0)
            .astype(bool)
        )
        basket.columns = [f"{col}_neg" for col in basket.columns]
        return basket

    # Cas 2 : DataFrame brut/nettoyé avec colonne Text (reviews_clean.csv)
    if "Text" in df.columns:
        texts = df["Text"].fillna("").str.lower()
        basket = pd.DataFrame(index=df.index)
        is_neg_review = df["Score"] <= 2 if "Score" in df.columns else pd.Series(True, index=df.index)

        for aspect, keywords in aspect_keywords.items():
            pattern = r"\b(?:" + "|".join(keywords) + r")\b"
            basket[f"{aspect}_neg"] = is_neg_review & texts.str.contains(pattern, regex=True)

        if "Score" in df.columns:
            basket["note_1_etoile"] = df["Score"] == 1

        basket = basket[basket.any(axis=1)]
        return basket.astype(bool)

    raise ValueError(f"Format de DataFrame non reconnu. Colonnes disponibles : {list(df.columns)}")


def analyze_negative_patterns(df, min_support=0.01, min_confidence=0.3, use_fpgrowth=True):
    """
    Extrait les itemsets fréquents et les règles d'association sur les plaintes négatives.
    Compatible avec l'appel de run_pipeline.py --module 4.
    """
    basket = extract_negative_aspect_transactions(df)

    if basket.empty:
        return pd.DataFrame(), pd.DataFrame()

    if use_fpgrowth:
        frequent_itemsets = fpgrowth(basket, min_support=min_support, use_colnames=True)
    else:
        frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)

    if frequent_itemsets.empty:
        return frequent_itemsets, pd.DataFrame()

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    if not rules.empty:
        rules = rules.sort_values(by=["lift", "confidence"], ascending=False).reset_index(drop=True)

    return frequent_itemsets, rules