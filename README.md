# ReviewScope 🔍

**Quels avis un vendeur doit-il lire en priorité ?**  
*Projet final 8INF867 — Fondamentaux de l'apprentissage automatique (UQAC)*

---

## Question centrale

À partir du texte des avis Amazon Fine Food (~568 000 avis), peut-on repérer ceux qui sont **fiables**, **utiles** et **révélateurs de problèmes précis**, puis les classer par ordre de priorité pour un vendeur ?

Ce projet combine les trois grandes familles du cours d'apprentissage automatique :
1. **Supervisé** : Cohérence texte/note (anomalies) et prédiction d'utilité avec gestion du déséquilibre de classes.
2. **Non supervisé** : Clustering (k-moyennes, DBSCAN) des plaintes négatives types et de niche.
3. **Règles d'association** : Apriori / FP-Growth sur les combinaisons d'aspects négatifs (ex: `{livraison-, emballage-} → 1 étoile`).
4. **Croisement de sources & Qualité** : Analyse formelle des données manquantes (MCAR / MAR / MNAR) lors de la jointure avec le catalogue Amazon 2023.

---

## Structure du projet

```text
aa_projet_final/
├── data/
│   ├── raw/                       # Reviews.csv brut (~568k lignes)
│   ├── processed/                 # reviews_clean.csv, matrices d'aspects
│   └── external/                  # Métadonnées Amazon 2023, Open Food Facts
├── notebooks/                     # 7 Notebooks Jupyter détaillés & documentés
│   ├── 00_audit_nettoyage.ipynb        # Module 0 : Audit, doublons métier, MCAR/MAR/MNAR
│   ├── 01_coherence_texte_note.ipynb   # Module 1 : Prédictions hors pli, anomalies signalées
│   ├── 02_prediction_utilite.ipynb     # Module 2 : Classification supervisée & ablations
│   ├── 03_aspects_sentiment.ipynb      # Module 3 : Dictionnaire vs Embeddings, matrice avis×aspect
│   ├── 04_non_supervise.ipynb          # Module 4 : k-moyennes, DBSCAN, FP-Growth
│   ├── 05_enrichissement_croise.ipynb  # Module 5 : Jointure catalogue 2023, analyse MNAR
│   └── 06_score_priorite.ipynb         # Module 6 : Livrable final & évaluation précision@k
├── src/                           # Code source modulaire et réutilisable
│   ├── __init__.py
│   ├── data_loader.py             # Chargement, dédoublonnage métier, split temporel
│   ├── text_processing.py         # Pipeline TF-IDF+SVD, embeddings, lisibilité
│   ├── feature_engineering.py     # Assemblage features texte + métadonnées sans fuite
│   ├── models.py                  # Zoo de modèles (LR, SVC, RF, GB, MLP, Stacking)
│   ├── aspects.py                 # Segmentation de phrases, détection d'aspects
│   ├── association_rules.py       # Règles d'association avec mlxtend
│   ├── priority_scorer.py         # Algorithme du score de priorité final
│   └── utils/
│       ├── __init__.py
│       └── visualization.py       # Fonctions graphiques et courbes PR
├── models/                        # Modèles et checkpoints sauvegardés
├── reports/
│   └── figures/                   # Graphiques générés pour le rapport
├── config.py                      # Configuration globale (seuils, graines, chemins)
├── run_pipeline.py                # Script CLI pour exécuter le pipeline d'un coup
├── requirements.txt               # Dépendances Python
└── README.md
```

---

## Installation

### 1. Cloner ou ouvrir le dossier du projet
```bash
cd aa_projet_final
```

### 2. Créer un environnement virtuel (recommandé)
```bash
python -m venv venv
# Sous Windows PowerShell :
.\venv\Scripts\Activate.ps1
# Ou sous Linux/macOS :
# source venv/bin/activate
```

### 3. Installer les dépendances
```bash
python -m pip install -r requirements.txt
```

---

## Comment lancer le projet ?

Vous avez **deux façons** d'exécuter le projet :

### Option A : Via la ligne de commande (Rapide & Automatisé)

Un script orchestrateur `run_pipeline.py` permet d'exécuter n'importe quel module ou l'ensemble du pipeline :

- **Exécuter tout le pipeline de bout en bout** :
  ```bash
  python run_pipeline.py --all --sample 15000
  ```

- **Exécuter un module spécifique** :
  ```bash
  # Module 0 : Audit et nettoyage des données (génère reviews_clean.csv)
  python run_pipeline.py --module 0

  # Module 1 : Détection des anomalies de cohérence texte/note
  python run_pipeline.py --module 1 --sample 10000

  # Module 2 : Comparaison des modèles de prédiction d'utilité
  python run_pipeline.py --module 2 --sample 20000

  # Module 3 : Extraction des aspects et du sentiment
  python run_pipeline.py --module 3 --sample 5000

  # Module 4 : Clustering des avis négatifs et règles d'association
  python run_pipeline.py --module 4

  # Module 6 : Génération de la liste prioritaire pour le vendeur
  python run_pipeline.py --module 6
  ```

---

### Option B : Via les Notebooks Jupyter (Interactif & Visuel)

Pour explorer les visualisations et analyses pas à pas :

1. Démarrez Jupyter Notebook ou JupyterLab :
   ```bash
   jupyter notebook
   ```
2. Ouvrez le dossier `notebooks/` et exécutez les notebooks dans l'ordre :
   - `00_audit_nettoyage.ipynb` : Détection des doublons métier, correction des ratios, découpage temporel anti-fuite.
   - `01_coherence_texte_note.ipynb` : Classifieur de sentiment et détection des anomalies (clic raté, sarcasme).
   - `02_prediction_utilite.ipynb` : Benchmarking (LogReg, SVM, RF, Gradient Boosting, MLP, Stacking) et ablations.
   - `03_aspects_sentiment.ipynb` : Extraction des thèmes (prix, emballage, goût, livraison...) et sentiment.
   - `04_non_supervise.ipynb` : k-moyennes, DBSCAN et règles d'association FP-Growth.
   - `05_enrichissement_croise.ipynb` : Jointure catalogue 2023 et analyse MCAR/MAR/MNAR.
   - `06_score_priorite.ipynb` : Calcul du score final pondéré et évaluation par Précision@k.

---

## Correspondance avec les notions du cours 8INF867

| Notion du cours | Emplacement dans le projet | Description |
|-----------------|---------------------------|-------------|
| **Fuite de données & Pipelines** | `src/data_loader.py`, `src/feature_engineering.py` | Découpage temporel strict (Train $\le$ 2010, Test 2011-2012) ; moyennes produit et vectoriseurs appris sur le train uniquement. |
| **Données manquantes MCAR/MAR/MNAR** | Module 0 & Module 5 | Traitement des votes à 0 (avis non évalués $\neq$ inutiles) et analyse des produits retirés du catalogue 2023. |
| **Anomalies comme signal** | Module 1 (`src/text_processing.py`) | Utilisation de `cross_val_predict` pour détecter les avis où le texte contredit la note. |
| **Normalisation des données** | Module 2 & Module 4 | Normalisation StandardScaler et $L_2$ avant SVM, réseaux de neurones, k-moyennes et DBSCAN. |
| **Réduction de dimension** | Module 2 & Module 4 | SVD (TruncatedSVD) sur TF-IDF et PCA 2D pour la visualisation. |
| **Modèles supervisés variés** | Module 2 (`src/models.py`) | Comparaison de modèles linéaires, à noyaux, ensemblistes (RF, GB, Stacking) et neuronaux (MLP). |
| **Déséquilibre de classes & Métriques** | Module 2 | Priorité donnée au PR-AUC et au score F1-macro plutôt qu'à l'exactitude brute. |
| **Non supervisé & Association** | Module 4 (`src/association_rules.py`) | k-moyennes (Elbow/Silhouette), DBSCAN (plaintes de niche) et règles d'association (Apriori/FP-Growth). |
