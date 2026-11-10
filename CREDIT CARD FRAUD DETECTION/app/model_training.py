import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
from .config import (
    DATASET_PATH, FEATURE_COLUMNS, TARGET, RANDOM_STATE, TEST_SIZE,
    MODEL_DIR, REPORT_DIR, METADATA_PATH, METRICS_PATH
)
from .data_loader import load_dataset
from .preprocessing import build_pipeline

def _evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
    }

def train_and_evaluate(dataset_path=DATASET_PATH):
    df = load_dataset(dataset_path)
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    candidates = {
        "random_forest": RandomForestClassifier(
            n_estimators=250, max_depth=None, min_samples_split=2,
            class_weight="balanced_subsample", n_jobs=-1,
            random_state=RANDOM_STATE
        ),
        "adaboost": AdaBoostClassifier(
            n_estimators=150, learning_rate=0.5, random_state=RANDOM_STATE
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    fitted = {}

    for name, estimator in candidates.items():
        print(f"Training {name}...")
        pipeline = build_pipeline(estimator)
        pipeline.fit(X_train, y_train)  # SMOTE is applied to training folds only.
        results[name] = _evaluate(pipeline, X_test, y_test)
        model_path = MODEL_DIR / f"{name}.joblib"
        joblib.dump(pipeline, model_path)
        fitted[name] = pipeline
        print(f"{name}: F1={results[name]['f1_score']:.4f}, "
              f"Recall={results[name]['recall']:.4f}, "
              f"PR-AUC={results[name]['pr_auc']:.4f}")

    # Select by PR-AUC, a useful metric for highly imbalanced fraud data.
    best_name = max(results, key=lambda key: results[key]["pr_auc"])
    metadata = {
        "best_model": best_name,
        "model_file": f"{best_name}.joblib",
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET,
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "fraud_rows": int(y.sum()),
        "legitimate_rows": int((y == 0).sum()),
        "selection_metric": "pr_auc",
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump({"models": results, "metadata": metadata}, f, indent=2)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    _save_plots(results)
    print(f"Best model: {best_name}")
    print(f"Metrics saved to {METRICS_PATH}")
    return results, metadata

def _save_plots(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    names = list(results)
    for metric in ("precision", "recall", "f1_score", "roc_auc", "pr_auc"):
        plt.figure(figsize=(8, 5))
        plt.bar(names, [results[n][metric] for n in names])
        plt.title(f"Model comparison: {metric}")
        plt.ylabel(metric.replace("_", " ").title())
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(REPORT_DIR / f"{metric}_comparison.png", dpi=150)
        plt.close()

    for name in names:
        plt.figure(figsize=(5, 4))
        sns.heatmap(results[name]["confusion_matrix"], annot=True, fmt="d",
                    cmap="Blues", cbar=False,
                    xticklabels=["Legitimate", "Fraud"],
                    yticklabels=["Legitimate", "Fraud"])
        plt.title(f"{name.replace('_', ' ').title()} confusion matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(REPORT_DIR / f"{name}_confusion_matrix.png", dpi=150)
        plt.close()

if __name__ == "__main__":
    train_and_evaluate()
