"""Assemblage des caractéristiques (texte + métadonnées) sans fuite de données."""

import numpy as np
import pandas as pd


def compute_metadata_features(df, train_product_means=None):
    """
    Calcule les features métadonnées de l'avis et du produit.
    Si train_product_means est None (mode Train), apprend les agrégats par produit.
    Sinon (mode Test), réutilise les agrégats du Train pour éviter toute fuite.
    """
    meta = pd.DataFrame(index=df.index)

    # Note et extrémité de la note (les avis 1★ et 5★ reçoivent plus de votes)
    meta["score"] = df["Score"].astype(float)
    meta["score_extremity"] = np.abs(df["Score"] - 3.0)
    meta["is_extreme_score"] = (df["Score"].isin([1, 5])).astype(int)

    # Temporalité
    if "Year" in df.columns:
        meta["year"] = df["Year"].astype(float)
    if "Month" in df.columns:
        meta["month"] = df["Month"].astype(float)

    # Agrégats produit calculés uniquement sur le Train
    if train_product_means is None:
        prod_stats = df.groupby("ProductId").agg(
            prod_avg_score=("Score", "mean"),
            prod_review_count=("Score", "count"),
        )
        global_mean_score = float(df["Score"].mean())
        train_product_means = {
            "stats": prod_stats,
            "global_mean": global_mean_score,
        }

    stats = train_product_means["stats"]
    global_mean = train_product_means["global_mean"]

    mapped_avg = df["ProductId"].map(stats["prod_avg_score"]).fillna(global_mean)
    mapped_count = df["ProductId"].map(stats["prod_review_count"]).fillna(0.0)

    meta["prod_avg_score"] = mapped_avg.values
    meta["prod_review_count_log"] = np.log1p(mapped_count.values)
    # Écart entre la note de l'avis et la moyenne du produit (signal d'avis atypique)
    meta["score_diff_from_prod"] = meta["score"] - meta["prod_avg_score"]

    return meta, train_product_means


def build_feature_matrix(df, X_tfidf, X_embeddings=None, text_features=None, meta_features=None):
    """Concatène toutes les matrices de features en un tableau NumPy propre et liste leurs noms."""
    blocks = []
    feature_names = []

    if X_tfidf is not None:
        blocks.append(np.asarray(X_tfidf))
        feature_names.extend([f"tfidf_svd_{i}" for i in range(X_tfidf.shape[1])])

    if X_embeddings is not None:
        blocks.append(np.asarray(X_embeddings))
        feature_names.extend([f"emb_{i}" for i in range(X_embeddings.shape[1])])

    if text_features is not None:
        blocks.append(text_features.values)
        feature_names.extend(list(text_features.columns))

    if meta_features is not None:
        blocks.append(meta_features.values)
        feature_names.extend(list(meta_features.columns))

    X = np.hstack(blocks).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, feature_names