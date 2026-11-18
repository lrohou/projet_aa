"""
ReviewScope - Pipeline d'exécution principal
============================================
Permet d'exécuter l'ensemble du projet ou des modules spécifiques en ligne de commande.

Usage :
    python run_pipeline.py --all
    python run_pipeline.py --module 0
    python run_pipeline.py --module 1
    python run_pipeline.py --module 2 --sample 20000
    python run_pipeline.py --module 6
"""
import argparse
import sys
from pathlib import Path

import config

def run_module_0():
    print("\n" + "="*70)
    print("MODULE 0 : Audit et Nettoyage des Données (cours 2)")
    print("="*70)
    from src.data_loader import run_full_pipeline
    df = run_full_pipeline()
    print(f"[OK] Module 0 terminé. {len(df):,} avis nettoyés enregistrés dans data/processed/reviews_clean.csv")
    return df

def run_module_1(df=None, sample_size=10000):
    print("\n" + "="*70)
    print("MODULE 1 : Cohérence Texte / Note (L'anomalie comme signal)")
    print("="*70)
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict

    clean_path = config.DATA_PROCESSED / "reviews_clean.csv"
    if df is None:
        if not clean_path.exists():
            df = run_module_0()
        else:
            df = pd.read_csv(clean_path)

    sample = df.sample(min(sample_size, len(df)), random_state=config.RANDOM_STATE).copy()
    print(f"Échantillon sélectionné : {len(sample)} avis")

    # Modélisation
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    X = tfidf.fit_transform(sample["Text"].fillna(""))
    y = sample["SentimentClass"]

    clf = LogisticRegression(max_iter=500, random_state=config.RANDOM_STATE)
    print("Calcul des prédictions hors pli (cross_val_predict)...")
    y_pred = cross_val_predict(clf, X, y, cv=3)

    sample["pred_sentiment"] = y_pred
    
    incoherents = sample[
        ((sample["Score"] <= 2) & (sample["pred_sentiment"] == "positif")) |
        ((sample["Score"] >= 4) & (sample["pred_sentiment"] == "negatif"))
    ]
    print(f"[Signal] Avis incohérents détectés (anomalies) : {len(incoherents)} ({len(incoherents)/len(sample)*100:.1f}%)")
    if len(incoherents) > 0:
        print("\nExemple d'anomalie détectée :")
        ex = incoherents.iloc[0]
        print(f"  Note réelle : {ex['Score']}/5 | Sentiment prédit : {ex['pred_sentiment']}")
        print(f"  Résumé : {ex['Summary']}")
        print(f"  Extrait : {ex['Text'][:200]}...")
    print("[OK] Module 1 terminé.")

def run_module_2(df=None, sample_size=20000):
    print("\n" + "="*70)
    print("MODULE 2 : Prédiction d'Utilité (Supervisé, Classes Déséquilibrées)")
    print("="*70)
    import pandas as pd
    from src.data_loader import temporal_split, stratified_sample
    from src.text_processing import build_tfidf_pipeline, compute_text_features, compute_readability
    from src.feature_engineering import compute_metadata_features, build_feature_matrix
    from src.models import compare_models

    clean_path = config.DATA_PROCESSED / "reviews_clean.csv"
    if df is None:
        if not clean_path.exists():
            df = run_module_0()
        else:
            df = pd.read_csv(clean_path)
            df["DateTime"] = pd.to_datetime(df["Time"], unit="s")

    df_eval = df[df["HelpfulnessDenominator"] >= config.MIN_VOTES].copy()
    print(f"Avis éligibles (>= {config.MIN_VOTES} votes) : {len(df_eval):,}")

    sample = stratified_sample(df_eval, n=min(sample_size, len(df_eval)), stratify_col="IsHelpful", random_state=config.RANDOM_STATE)
    train_df, test_df = temporal_split(sample, config.TRAIN_END_YEAR, config.TEST_START_YEAR)

    print("Extraction des features (TF-IDF + SVD, texte, métadonnées)...")
    tfidf_svd = build_tfidf_pipeline(max_features=2000, n_components=30)
    X_tfidf_tr = tfidf_svd.fit_transform(train_df["Text"].fillna(""))
    X_tfidf_te = tfidf_svd.transform(test_df["Text"].fillna(""))

    tr_text = compute_text_features(train_df)
    te_text = compute_text_features(test_df)
    tr_text["readability"] = compute_readability(train_df["Text"])
    te_text["readability"] = compute_readability(test_df["Text"])

    tr_meta, prod_means = compute_metadata_features(train_df)
    te_meta, _ = compute_metadata_features(test_df, train_product_means=prod_means)

    X_train, _ = build_feature_matrix(train_df, X_tfidf_tr, None, tr_text, tr_meta)
    X_test, _ = build_feature_matrix(test_df, X_tfidf_te, None, te_text, te_meta)

    y_train = train_df["IsHelpful"].values
    y_test = test_df["IsHelpful"].values

    print(f"X_train : {X_train.shape} | X_test : {X_test.shape}")
    results_df, _ = compare_models(X_train, y_train, X_test, y_test)
    print("\nRésumé de la comparaison des modèles :")
    print(results_df)
    print("[OK] Module 2 terminé.")

def run_module_3(df=None, sample_size=3000):
    print("\n" + "="*70)
    print("MODULE 3 : Aspects et Sentiment par Aspect")
    print("="*70)
    import pandas as pd
    from src.aspects import build_aspect_matrix, summarize_aspects

    clean_path = config.DATA_PROCESSED / "reviews_clean.csv"
    if df is None:
        if not clean_path.exists():
            df = run_module_0()
        else:
            df = pd.read_csv(clean_path)

    sample = df.sample(min(sample_size, len(df)), random_state=config.RANDOM_STATE).copy()
    print("Construction de la matrice Avis x Aspect x Sentiment...")
    aspect_df = build_aspect_matrix(sample, method="keywords")
    print(f"Mentions d'aspects trouvées : {len(aspect_df):,}")
    summary = summarize_aspects(aspect_df)
    print("\nRésumé par aspect :")
    print(summary)
    aspect_path = config.DATA_PROCESSED / "aspects_sample.csv"
    aspect_df.to_csv(aspect_path, index=False)
    print(f"[OK] Module 3 terminé. Matrice sauvegardée dans {aspect_path}")
    return aspect_df

def run_module_4():
    print("\n" + "="*70)
    print("MODULE 4 : Découverte Non Supervisée (Clustering & Règles d'Association)")
    print("="*70)
    import pandas as pd
    from src.association_rules import analyze_negative_patterns

    aspect_path = config.DATA_PROCESSED / "aspects_sample.csv"
    if not aspect_path.exists():
        aspect_df = run_module_3()
    else:
        aspect_df = pd.read_csv(aspect_path)

    print("Recherche de règles d'association sur les aspects négatifs (FP-Growth)...")
    frequent, rules = analyze_negative_patterns(aspect_df, min_support=0.01, min_confidence=0.3)
    if rules is not None and not rules.empty:
        print("\nTop 5 règles d'association par lift :")
        print(rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(5))
    else:
        print("Aucune règle trouvée avec ces seuils ou support trop élevé.")
    print("[OK] Module 4 terminé.")

def run_module_6():
    print("\n" + "="*70)
    print("MODULE 6 : Score de Priorité & Livrable Vendeur")
    print("="*70)
    import pandas as pd
    import numpy as np
    from src.priority_scorer import compute_priority_score, get_top_reviews, evaluate_precision_at_k

    clean_path = config.DATA_PROCESSED / "reviews_clean.csv"
    if not clean_path.exists():
        df = run_module_0()
    else:
        df = pd.read_csv(clean_path)

    # Prendre le produit le plus commenté
    top_prod = df["ProductId"].value_counts().index[0]
    prod_df = df[df["ProductId"] == top_prod].copy().reset_index(drop=True)
    print(f"Produit ciblé : {top_prod} ({len(prod_df)} avis)")

    # Calcul signaux
    helpfulness_proba = prod_df["HelpfulnessRatio"].fillna(0.5).values
    incoherence_score = np.abs(prod_df["Score"] - 3.0) / 2.0
    neg_pattern = "bad|terrible|awful|broken|stale|damaged|poor|slow"
    neg_aspect_count = prod_df["Text"].str.lower().str.count(neg_pattern).values

    scores = compute_priority_score(
        prod_df,
        helpfulness_proba=helpfulness_proba,
        incoherence_score=incoherence_score,
        negative_aspect_count=neg_aspect_count
    )
    prod_df["priority_score"] = scores

    top5 = get_top_reviews(prod_df, scores, n=5)
    print("\n=== TOP 5 AVIS PRIORITAIRES POUR LE VENDEUR ===")
    for rank, (_, row) in enumerate(top5.iterrows(), 1):
        print(f"\n[Rang {rank}] Score: {row['priority_score']:.3f} | Note: {row['Score']}/5 | Utilité: {row['HelpfulnessNumerator']}/{row['HelpfulnessDenominator']}")
        print(f"Résumé: {row['Summary']}")
        print(f"Extrait: {row['Text'][:150]}...")

    p_at_k = evaluate_precision_at_k(prod_df["IsHelpful"], scores.values, k_values=(3, 5))
    print(f"\nPrécision@k sur votes réels : {p_at_k}")
    print("[OK] Module 6 terminé.")

def main():
    parser = argparse.ArgumentParser(description="ReviewScope - Pipeline ML complet")
    parser.add_argument("--all", action="store_true", help="Exécuter tous les modules")
    parser.add_argument("--module", type=int, choices=[0, 1, 2, 3, 4, 6], help="Exécuter un module spécifique")
    parser.add_argument("--sample", type=int, default=10000, help="Taille d'échantillon pour tests rapides")

    args = parser.parse_args()

    if args.all:
        df = run_module_0()
        run_module_1(df, sample_size=args.sample)
        run_module_2(df, sample_size=args.sample)
        run_module_3(df, sample_size=min(args.sample, 5000))
        run_module_4()
        run_module_6()
        print("\n" + "="*70)
        print("PIPELINE COMPLET EXÉCUTÉ AVEC SUCCÈS !")
        print("="*70)
    elif args.module is not None:
        if args.module == 0:
            run_module_0()
        elif args.module == 1:
            run_module_1(sample_size=args.sample)
        elif args.module == 2:
            run_module_2(sample_size=args.sample)
        elif args.module == 3:
            run_module_3(sample_size=args.sample)
        elif args.module == 4:
            run_module_4()
        elif args.module == 6:
            run_module_6()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
