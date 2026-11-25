"""Zoo de modèles supervisés et évaluation pour la prédiction d'utilité (Module 2)."""

import time
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import config


def get_models():
    """Retourne le dictionnaire des pipelines à comparer."""
    rs = config.RANDOM_STATE

    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=rs)),
    ])

    svc = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", random_state=rs, max_iter=2000),
            cv=3,
        )),
    ])

    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        class_weight="balanced",
        random_state=rs,
        n_jobs=-1,
    )

    gb = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=4,
        random_state=rs,
    )

    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=30,
            early_stopping=True,
            random_state=rs,
        )),
    ])

    stacking = StackingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("gb", gb)],
        final_estimator=LogisticRegression(class_weight="balanced", random_state=rs),
        cv=3,
        n_jobs=-1,
    )

    return {
        "LogisticRegression": lr,
        "LinearSVC": svc,
        "RandomForest": rf,
        "GradientBoosting": gb,
        "MLP": mlp,
        "Stacking": stacking,
    }


def compare_models(X_train, y_train, X_test, y_test, models=None):
    """Entraîne et évalue tous les modèles selon les métriques adaptées au déséquilibre."""
    if models is None:
        models = get_models()

    results = []
    trained_models = {}

    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        results.append({
            "Model": name,
            "PR-AUC": round(average_precision_score(y_test, y_proba), 4),
            "ROC-AUC": round(roc_auc_score(y_test, y_proba), 4),
            "F1-macro": round(f1_score(y_test, y_pred, average="macro"), 4),
            "F1-helpful": round(f1_score(y_test, y_pred, pos_label=1), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "Time (s)": round(train_time, 2),
        })
        trained_models[name] = model

    results_df = pd.DataFrame(results).sort_values(by="PR-AUC", ascending=False).reset_index(drop=True)
    return results_df, trained_models